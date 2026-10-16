import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import tabulate as tb

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from matplotlib.ticker import MultipleLocator

from config import CONFIG
from index_creation import get_season_days
from objective_func import objective_function
from simulation.simulation import run_simulation

PROJECT_ROOT = Path(__file__).resolve().parent


def log_generation_metrics(
    population,
    hof,
    record,
    nevals,
    gen,
    design_var,
    operational_var,
    season_days,
    simulation_successful,
    simulation_penalized,
    run_successful_total,
    run_penalized_total,
    logger,
):
    """Log generation statistics and every gene of the generation/global best."""

    generation_best = tools.selBest(population, k=1)[0]
    global_best = hof[0]

    metrics = {
        "fitness/avg": float(record["avg"]),
        "fitness/std": float(record["std"]),
        "fitness/min": float(record["min"]),
        "fitness/max": float(record["max"]),
        "fitness/nevals": float(nevals),
        "fitness/generation_best": float(generation_best.fitness.values[0]),
        "fitness/global_best": float(global_best.fitness.values[0]),
        # Preserve the existing metric name for continuity.
        "best_fitness": float(global_best.fitness.values[0]),
    }
    simulation_total = simulation_successful + simulation_penalized

    simulation_success_rate = (
        100.0 * simulation_successful / simulation_total
        if simulation_total > 0
        else 0.0
    )

    metrics.update(
        {
            "simulations/successful": float(simulation_successful),
            "simulations/penalized": float(simulation_penalized),
            "simulations/total": float(simulation_total),
            "simulations/success_rate_pct": float(simulation_success_rate),
            "simulations/run_successful_total": float(run_successful_total),
            "simulations/run_penalized_total": float(run_penalized_total),
        }
    )

    def safe_name(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")

    # Log each design gene individually.
    for i, name in enumerate(design_var):
        key = safe_name(name)

        metrics[f"generation_best/design/{key}"] = float(generation_best[i])
        metrics[f"global_best/design/{key}"] = float(global_best[i])

    # Log all operational genes for every representative day.
    num_operational_vars = len(operational_var)

    for day_info in season_days:
        local_index = day_info["local_index"]

        day_label = (
            f"day_{local_index + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        )

        for j, name in enumerate(operational_var):
            gene_index = len(design_var) + local_index * num_operational_vars + j

            key = safe_name(name)

            metrics[f"generation_best/operational/{day_label}/{key}"] = float(
                generation_best[gene_index]
            )

            metrics[f"global_best/operational/{day_label}/{key}"] = float(
                global_best[gene_index]
            )

    # Avoid sending NaN or infinite values to MLflow.
    metrics = {key: value for key, value in metrics.items() if np.isfinite(value)}

    # Every metric is associated with this generation.
    mlflow.log_metrics(metrics, step=gen)

    logger.info(
        "Generation %d | best fitness: %.6f | global best fitness: %.6f",
        gen,
        generation_best.fitness.values[0],
        global_best.fitness.values[0],
    )


def worker_loop(season, comm, rank, logger):
    """
    Worker MPI rank.

    Each worker rank creates a local multiprocessing pool,
    evaluates batches, and reports success or failure to rank 0.

    A Python exception during batch evaluation is reported to rank 0.
    If evaluation fails, the worker terminates its local pool before
    exiting instead of potentially hanging while joining it.
    """

    logger.info(
        "Starting worker | Season: %s | MPI rank: %s",
        season,
        rank,
    )

    num_workers = int(os.environ.get("LOCAL_WORKERS", "4"))

    if num_workers < 1:
        raise ValueError(f"LOCAL_WORKERS must be >= 1, got {num_workers}")

    mp_context = mp.get_context()
    pool_failed = False

    try:
        with mp_context.Pool(
            processes=num_workers,
            maxtasksperchild=100,
        ) as pool:
            logger.info(
                "Multiprocessing pool started | MPI rank: %s | Workers: %s",
                rank,
                num_workers,
            )

            while True:
                # Wait for a batch or shutdown signal from rank 0.
                batch = comm.recv(source=0, tag=1)

                if batch is None:
                    logger.info(
                        "Received shutdown signal | MPI rank: %s",
                        rank,
                    )
                    break

                logger.info(
                    "Received batch | MPI rank: %s | Individuals: %d",
                    rank,
                    len(batch),
                )

                try:
                    batch_result = evaluate_batch(
                        batch=batch,
                        pool=pool,
                        season=season,
                        rank=rank,
                        logger=logger,
                    )

                    # Validate the result before reporting success.
                    if not isinstance(batch_result, dict):
                        raise TypeError("evaluate_batch() must return a dictionary")

                    required_keys = {
                        "fitnesses",
                        "successful",
                        "penalized",
                    }

                    missing_keys = required_keys - batch_result.keys()

                    if missing_keys:
                        raise KeyError(
                            f"evaluate_batch() result missing keys: {sorted(missing_keys)}"
                        )

                    response = {
                        "ok": True,
                        "result": batch_result,
                        "error": None,
                    }

                    logger.info(
                        "Batch evaluated successfully | MPI rank: %s | "
                        "Individuals: %d | Successful simulations: %s | "
                        "Penalized simulations: %s",
                        rank,
                        len(batch),
                        batch_result["successful"],
                        batch_result["penalized"],
                    )

                except Exception as exc:
                    pool_failed = True

                    logger.exception(
                        "Batch evaluation failed | MPI rank: %s",
                        rank,
                    )

                    response = {
                        "ok": False,
                        "result": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                # Reply to rank 0 for every batch received, including failures.
                comm.send(response, dest=0, tag=2)

                # Do not keep using a potentially unhealthy pool after failure.
                if pool_failed:
                    logger.error(
                        "Stopping worker after batch failure | MPI rank: %s",
                        rank,
                    )
                    break

    except Exception:
        logger.exception(
            "Worker loop encountered an unexpected error | MPI rank: %s",
            rank,
        )
        raise

    finally:
        logger.info(
            "Worker stopped | MPI rank: %s",
            rank,
        )


def batch_individuals(individuals, batch_size=16):
    """
    Split individuals into batches
    """
    return [
        individuals[i : i + batch_size] for i in range(0, len(individuals), batch_size)
    ]


def evaluate_batch(batch, pool, season, rank, logger):
    """
    Evaluate a batch of individuals across all representative days.

    Returns:
        fitnesses: DEAP-compatible fitness tuples
        successful: number of successful simulations
        penalized: number of penalized simulations
    """

    season_days = get_season_days(season)
    tasks = []

    for individual_id, individual in enumerate(batch):
        for day_info in season_days:
            tasks.append(
                (
                    individual_id,
                    day_info["local_index"],
                    day_info["month"],
                    day_info["day"],
                    day_info["day_index"],
                    individual,
                )
            )

    logger.info(
        f"[Rank {rank}] Created {len(tasks)} simulation tasks "
        f"for {len(batch)} individuals"
    )

    results = pool.starmap(run_one_simulation, tasks)

    # One fitness total per individual in this batch
    fitnesses = [0.0] * len(batch)

    successful = 0
    penalized = 0

    for (
        individual_id,
        local_day_index,
        month,
        day,
        day_index,
        objective,
        was_penalized,
    ) in results:
        fitnesses[individual_id] += objective

        if was_penalized:
            penalized += 1
        else:
            successful += 1

    logger.info(
        f"[Rank {rank}] Batch completed | "
        f"successful={successful} | penalized={penalized} | "
        f"total={successful + penalized}"
    )

    return {
        "fitnesses": [(fitness,) for fitness in fitnesses],
        "successful": successful,
        "penalized": penalized,
    }


def distribute_batches(
    batches,
    local_pool,
    season,
    comm,
    rank,
    mpi_size,
    logger,
):
    workers = list(range(1, mpi_size))

    all_fitnesses = []
    total_successful = 0
    total_penalized = 0

    for wave_start in range(0, len(batches), mpi_size):
        wave = batches[wave_start : wave_start + mpi_size]

        logger.info(
            f"[Rank {rank}] Processing batches "
            f"{wave_start + 1} - {wave_start + len(wave)}"
        )

        remote_batches = wave[: len(workers)]

        for worker, batch in zip(workers, remote_batches):
            comm.send(batch, dest=worker, tag=1)

        local_result = None

        if len(wave) > len(workers):
            local_batch = wave[len(workers)]

            local_result = evaluate_batch(
                batch=local_batch,
                pool=local_pool,
                season=season,
                rank=rank,
                logger=logger,
            )

        # Receive the remote results in the same order
        # that the corresponding batches were dispatched.

        # Receive all responses expected for this wave.
        remote_responses = []

        for worker in workers[: len(remote_batches)]:
            response = comm.recv(source=worker, tag=2)
            remote_responses.append((worker, response))

        # Check for batch failures after collecting all responses.
        failures = []

        for worker, response in remote_responses:
            if not isinstance(response, dict):
                failures.append(
                    (worker, f"Unexpected response type: {type(response).__name__}")
                )
            elif not response.get("ok", False):
                failures.append((worker, response.get("error", "Unknown worker error")))
            elif not isinstance(response.get("result"), dict):
                failures.append(
                    (worker, "Worker response is missing its result dictionary")
                )

        if failures:
            details = "; ".join(f"rank {worker}: {error}" for worker, error in failures)

            logger.error("Remote batch evaluation failed: %s", details)

            raise RuntimeError(f"MPI batch evaluation failed: {details}")

        # Aggregate successful results in dispatch order.
        for worker, response in remote_responses:
            result = response["result"]

            all_fitnesses.extend(result["fitnesses"])
            total_successful += result["successful"]
            total_penalized += result["penalized"]

        # Rank 0's local result retains the original format.
        if local_result is not None:
            all_fitnesses.extend(local_result["fitnesses"])
            total_successful += local_result["successful"]
            total_penalized += local_result["penalized"]

    return {
        "fitnesses": all_fitnesses,
        "successful": total_successful,
        "penalized": total_penalized,
    }


def evaluate_population(
    population,
    local_pool,
    season,
    comm,
    rank,
    mpi_size,
    logger,
):
    batches = batch_individuals(
        population,
        batch_size=16,
    )

    evaluation_result = distribute_batches(
        batches=batches,
        local_pool=local_pool,
        season=season,
        comm=comm,
        rank=rank,
        mpi_size=mpi_size,
        logger=logger,
    )

    fitnesses = evaluation_result["fitnesses"]

    if len(fitnesses) != len(population):
        raise RuntimeError(
            f"Expected {len(population)} fitness results, received {len(fitnesses)}"
        )

    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness

    return {
        "nevals": len(population),
        "successful": evaluation_result["successful"],
        "penalized": evaluation_result["penalized"],
    }


def run_one_simulation(
    individual_id,
    local_day_index,
    month,
    day,
    day_index,
    individual,
):
    # -------------------------------------------------
    # Design parameters
    # -------------------------------------------------

    design_params = individual[:5]

    # -------------------------------------------------
    # Operational parameters
    # -------------------------------------------------

    operational_params = individual[5:]

    # local_day_index = 0, 1, ..., 6
    #
    # This determines which 3 operational genes belong
    # to this representative day.

    start = local_day_index * 3
    end = start + 3

    daily_operational = operational_params[start:end]

    # -------------------------------------------------
    # Combine inputs
    # -------------------------------------------------

    simulation_input = design_params + daily_operational

    # -------------------------------------------------
    # Convert to PySAM overrides
    # -------------------------------------------------

    var_names = CONFIG["design"]["overrides"] + CONFIG["operational"]["overrides"]

    var_types = CONFIG["design"]["types"] + CONFIG["operational"]["types"]

    overrides = {
        var_names[j]: var_types[j](simulation_input[j]) for j in range(len(var_names))
    }

    # Ensure shutdown temperature never exceeds startup temperature
    if "T_shutdown" in overrides and "T_startup" in overrides:
        overrides["T_shutdown"] = min(overrides["T_shutdown"], overrides["T_startup"])

    # -------------------------------------------------
    # Run PySAM
    # -------------------------------------------------

    sim_result, penalty_flag = run_simulation(overrides, day_index=day_index)

    # -------------------------------------------------
    # Penalty
    # -------------------------------------------------

    if penalty_flag or sim_result is None:
        return (
            individual_id,
            local_day_index,
            month,
            day,
            day_index,
            float(CONFIG["penalty"]),
            True,  # penalized
        )

    # -------------------------------------------------
    # Daily objective
    # -------------------------------------------------

    objective = objective_function(
        hourly_energy=sim_result["hourly_energy"],
        field_htf_pump_power=sim_result["field_htf_pump_power"],
        pc_htf_pump_power=sim_result["pc_htf_pump_power"],
        field_collector_tracking_power=sim_result["field_collector_tracking_power"],
        pc_startup_thermal_power=sim_result["pc_startup_thermal_power"],
        field_piping_thermal_loss=sim_result["field_piping_thermal_loss"],
        receiver_thermal_loss=sim_result["receiver_thermal_loss"],
        day_index=day_index,
    )

    # -------------------------------------------------
    # Validate objective
    # -------------------------------------------------

    # Validate objective

    penalized = objective is None or not np.isfinite(objective)

    if penalized:
        objective = float(CONFIG["penalty"])

    return (
        individual_id,
        local_day_index,
        month,
        day,
        day_index,
        float(objective),
        penalized,
    )


def init_fresh_ga(toolbox, pop_size, logger):
    """Encapsulates the logic for starting a brand-new evolution."""
    logger.info("Starting fresh GA run")

    # Set seeds from CONFIG for reproducibility
    random.seed(CONFIG.get("random_seed"))
    np.random.seed(CONFIG.get("random_seed"))

    pop = toolbox.population(n=pop_size)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "max", "min"]

    hof = tools.HallOfFame(maxsize=CONFIG.get("hall_of_fame_size"))

    start_gen = 0
    return pop, logbook, hof, start_gen


# serialise population object to be pickle-safe
def serialize_population(pop):
    return [(list(ind), ind.fitness.values) for ind in pop]


# de-serialise population object from pickle one
def deserialize_population(serialized, toolbox):
    pop = []
    for genome, fitness in serialized:
        ind = toolbox.individual()
        ind[:] = genome
        ind.fitness.values = fitness
        pop.append(ind)
    return pop


# -------- MAIN GA TASK ----------------------
def run_deap_ga_optimisation(
    season: str,
    local_pool,
    comm,
    rank,
    mpi_size,
    logger,
):
    if season not in CONFIG["SEASONS"]:
        raise ValueError(
            f"Unknown season: {season}. "
            f"Available seasons: "
            f"{list(CONFIG['SEASONS'].keys())}"
        )

    season_days = get_season_days(season)

    num_days = len(season_days)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Starting GA optimisation for season: {season}")

    logger.info(f"Representative days: {season_days}")

    logger.info(f"Number of representative days: {num_days}")
    # Read configuration
    design_lb, design_ub = (
        CONFIG.get("design")["lb"],
        CONFIG.get("design")["ub"],
    )
    design_var = CONFIG.get("design")["overrides"]
    design_var_types = CONFIG.get("design")["types"]
    operational_lb, operational_ub = (
        CONFIG.get("operational")["lb"],
        CONFIG.get("operational")["ub"],
    )
    operational_var = CONFIG.get("operational")["overrides"]
    operational_var_types = CONFIG.get("operational")["types"]

    pop_size = CONFIG.get("pop_size")
    num_generations = CONFIG.get("num_generations")
    cxpb = CONFIG.get("cxpb")
    mutpb = CONFIG.get("mutpb")
    indpb = CONFIG.get("indpb")

    logger.info(f"Design Bounds: {list(zip(design_lb, design_ub))}")
    logger.info(f"Operational Bounds: {list(zip(operational_lb, operational_ub))}")

    # DEAP setup
    if not hasattr(creator, "FitnessMax"):
        creator.create(
            "FitnessMax", base.Fitness, weights=(1.0,)
        )  # 1.0 for maximising objective func
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # toolbox init
    toolbox = base.Toolbox()

    # individual generation
    def gen_individual():
        individual = []
        # DESIGN VARIABLES
        for lb, ub in zip(
            design_lb,
            design_ub,
        ):
            individual.append(random.randint(lb, ub))

        # OPERATIONAL VARIABLES
        for _ in range(num_days):
            for lb, ub in zip(
                operational_lb,
                operational_ub,
            ):
                individual.append(random.randint(lb, ub))
        # Sanity check
        assert len(individual) == 5 + 3 * num_days
        return individual

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        gen_individual,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register(
    #     "evaluate",
    #     deap_fitness,
    #     day_index=current_day,
    #     var_names=design_var + operational_var,
    #     var_types=design_var_types + operational_var_types,
    # )

    tourn_size = CONFIG.get("tournament_size", 7)
    if pop_size is not None and tourn_size > pop_size:
        tourn_size = max(2, pop_size // 2) if pop_size >= 2 else 1

    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=tourn_size,
    )
    toolbox.register("mate", tools.cxTwoPoint)

    def custom_mutation(individual):
        """
        Mutate an individual consisting of:
            5 design parameters
            num_days × 3 operational parameters
        All variables are integer-valued.
        """
        # MUTATE DESIGN PARAMETERS
        for i in range(5):
            if random.random() < indpb:
                sigma = 0.1 * (design_ub[i] - design_lb[i])
                mutated_value = individual[i] + random.gauss(0, sigma)
                mutated_value = round(mutated_value)
                individual[i] = int(max(design_lb[i], min(design_ub[i], mutated_value)))
        # MUTATE OPERATIONAL PARAMETERS
        for day in range(num_days):
            # Starting index of this day's 3 variables
            start = 5 + day * 3
            for j in range(3):
                if random.random() < indpb:
                    sigma = 0.1 * (operational_ub[j] - operational_lb[j])
                    mutated_value = individual[start + j] + random.gauss(0, sigma)
                    mutated_value = round(mutated_value)
                    individual[start + j] = int(
                        max(
                            operational_lb[j],
                            min(operational_ub[j], mutated_value),
                        )
                    )

        return (individual,)

    toolbox.register("mutate", custom_mutation)

    # ------- Stable checkpoint key ------------------
    ckpt_key = hashlib.sha256(
        json.dumps(
            {
                "season": season,
                "vars": design_var + operational_var,
                "types": [t.__name__ for t in design_var_types + operational_var_types],
                "lb": design_lb + operational_lb,
                "ub": design_ub + operational_ub,
                "pop": pop_size,
                "gens": num_generations,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "indpb": indpb,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]

    def safe_pickle_save(data, file_path):
        tmp_path = file_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f)
            tmp_path.replace(file_path)  # atomic on same filesystem
            return True
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
            return False

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "GA" / season / ckpt_key
    resume_file = checkpoint_dir / "checkpoint_latest.pkl"
    resume_file.parent.mkdir(parents=True, exist_ok=True)

    # Check for legacy top-level checkpoints if season-specific directory is empty
    if not resume_file.exists() and CONFIG.get("resume_from_checkpoint", False):
        legacy_dir = PROJECT_ROOT / "checkpoints" / "GA" / "66eb56f64973"
        legacy_file = legacy_dir / "checkpoint_latest.pkl"
        if season == "summer" and legacy_file.exists():
            logger.info("Found legacy Summer checkpoint in %s; linking for resumption.", legacy_file)
            resume_file = legacy_file

    # ------- Resume or fresh start -----------------------
    run_successful_total = 0
    run_penalized_total = 0

    if resume_file.exists() and CONFIG.get("resume_from_checkpoint", False):
        try:
            with open(resume_file, "rb") as f:
                cp = pickle.load(f)

            # Validate that the checkpoint matches or has compatible chromosome dimensions
            valid_key = (cp.get("ckpt_key") == ckpt_key)
            if not valid_key:
                pop_data = cp.get("pop", [])
                if len(pop_data) == pop_size and len(pop_data[0][0]) == (5 + 3 * num_days):
                    logger.info(
                        "Accepting compatible checkpoint (key: %s, gen: %d, pop: %d).",
                        cp.get("ckpt_key"),
                        cp.get("generation"),
                        len(pop_data),
                    )
                    valid_key = True

            if not valid_key:
                logger.warning(
                    "Checkpoint key mismatch! Starting fresh to avoid Gene corruption."
                )
                run_successful_total = 0
                run_penalized_total = 0
                pop, logbook, hof, start_gen = init_fresh_ga(toolbox, pop_size, logger)
            else:
                # Restore cumulative simulation counters.
                run_successful_total = cp.get("run_successful_total", 0)
                run_penalized_total = cp.get("run_penalized_total", 0)

                logger.info(
                    f"Resuming from {resume_file} at generation {cp['generation']}"
                )

                # Restore random states before continuing evolution.
                random.setstate(cp["rndstate"])
                np.random.set_state(cp["np_rndstate"])

                # IMPORTANT: use the same key as the checkpoint writer.
                pop = deserialize_population(cp["pop"], toolbox)

                logbook = cp["logbook"]

                hof = tools.HallOfFame(maxsize=CONFIG.get("hall_of_fame_size"))
                hof.update(deserialize_population(cp["hof"], toolbox))

                start_gen = cp["generation"] + 1

        except Exception as e:
            logger.error(f"Checkpoint corrupted: {e}. Starting fresh.")

            run_successful_total = 0
            run_penalized_total = 0

            pop, logbook, hof, start_gen = init_fresh_ga(toolbox, pop_size, logger)

    else:
        pop, logbook, hof, start_gen = init_fresh_ga(toolbox, pop_size, logger)

    stats = tools.Statistics(
        lambda ind: ind.fitness.values[0]
    )  # take 0th-index because deap supports multi-objective function
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    gens_log = [entry["gen"] for entry in logbook] if logbook else []
    max_fitness_log = [entry["max"] for entry in logbook] if logbook else []
    avg_fitness_log = [entry["avg"] for entry in logbook] if logbook else []

    plot_path = PROJECT_ROOT / "plots" / "GA_plots" / f"{season}_fitness_vs_gen_{timestamp}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if start_gen == 0:
        initial_stats = evaluate_population(
            population=pop,
            local_pool=local_pool,
            season=season,
            comm=comm,
            rank=rank,
            mpi_size=mpi_size,
            logger=logger,
        )

        run_successful_total += initial_stats["successful"]
        run_penalized_total += initial_stats["penalized"]

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)

        log_generation_metrics(
            population=pop,
            hof=hof,
            record=record,
            nevals=len(pop),
            gen=0,
            design_var=design_var,
            operational_var=operational_var,
            season_days=season_days,
            simulation_successful=initial_stats["successful"],
            simulation_penalized=initial_stats["penalized"],
            run_successful_total=run_successful_total,
            run_penalized_total=run_penalized_total,
            logger=logger,
        )

        gens_log.append(0)
        max_fitness_log.append(record["max"])
        avg_fitness_log.append(record["avg"])

    # ----------- GA loop ------------------------------------
    for gen in range(max(1, start_gen), num_generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        evaluation_stats = evaluate_population(
            population=offspring,
            local_pool=local_pool,
            season=season,
            comm=comm,
            rank=rank,
            mpi_size=mpi_size,
            logger=logger,
        )

        nevals = evaluation_stats["nevals"]

        run_successful_total += evaluation_stats["successful"]
        run_penalized_total += evaluation_stats["penalized"]

        # True elitism: preserve up to len(hof) elites, selecting remaining from offspring
        elite_count = min(len(hof), max(1, pop_size // 2)) if pop_size <= len(hof) else len(hof)
        pop = toolbox.select(offspring, k=pop_size - elite_count)
        pop.extend([toolbox.clone(ind) for ind in hof[:elite_count]])
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=nevals, **record)

        log_generation_metrics(
            population=pop,
            hof=hof,
            record=record,
            nevals=nevals,
            gen=gen,
            design_var=design_var,
            operational_var=operational_var,
            season_days=season_days,
            simulation_successful=evaluation_stats["successful"],
            simulation_penalized=evaluation_stats["penalized"],
            run_successful_total=run_successful_total,
            run_penalized_total=run_penalized_total,
            logger=logger,
        )

        gens_log.append(gen)
        max_fitness_log.append(record["max"])
        avg_fitness_log.append(record["avg"])

        if gen % CONFIG.get("checkpoint_interval", 5) == 0 or gen == num_generations - 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(gens_log, max_fitness_log, "r-", label="Max Fitness", lw=2)
            ax.plot(gens_log, avg_fitness_log, "b--", label="Avg Fitness", alpha=0.7)

            ax.set_title("Fitness vs Generation")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)

            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        best_ind = hof[0]

        cp_data = {
            "season": season,
            "var_name": design_var + operational_var,
            "var_types": design_var_types + operational_var_types,
            "lb": design_lb + operational_lb,
            "ub": design_ub + operational_ub,
            "pop": serialize_population(pop),
            "logbook": logbook,
            "hof": serialize_population(hof),
            "ckpt_key": ckpt_key,
            "generation": gen,
            "rndstate": random.getstate(),
            "np_rndstate": np.random.get_state(),
            "run_successful_total": run_successful_total,
            "run_penalized_total": run_penalized_total,
        }

        try:
            # Save the latest and generation-specific checkpoints.
            latest_saved = safe_pickle_save(
                cp_data,
                resume_file,
            )

            gen_file = checkpoint_dir / f"checkpoint_gen_{gen}.pkl"

            generation_saved = safe_pickle_save(
                cp_data,
                gen_file,
            )

            if not latest_saved or not generation_saved:
                logger.error(
                    "Checkpoint save incomplete at generation %d "
                    "(latest=%s, generation=%s)",
                    gen,
                    latest_saved,
                    generation_saved,
                )
            else:
                logger.info(
                    "Saved checkpoints for generation %d",
                    gen,
                )

                # Upload every generation's checkpoint to MLflow.
                try:
                    mlflow.log_artifact(
                        str(gen_file),
                        artifact_path="checkpoints/history",
                    )
                except Exception:
                    logger.exception(
                        "Failed to upload generation %d checkpoint "
                        "to MLflow; local checkpoint is preserved.",
                        gen,
                    )
        except Exception as e:
            logger.warning(f"Checkpoint write failed (Generation {gen}): {e}")

    # =========================================================
    # FINAL RESULT
    # =========================================================

    best_ind = hof[0]

    best_fitness = float(best_ind.fitness.values[0])

    best_solution = {
        "season": season,
        "fitness": best_fitness,
        "individual": list(best_ind),
    }

    # ---------------------------------------------------------
    # Log final solution to MLflow
    # ---------------------------------------------------------

    mlflow.log_params(
        {
            "checkpoint_key": ckpt_key,
            "final_best_fitness": best_fitness,
        }
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    res_dict = {
        "season": season,
        "best_fitness": best_fitness,
    }

    for i, name in enumerate(design_var):
        res_dict[f"design_{name}"] = best_ind[i]

    num_operational_vars = len(operational_var)
    for day_idx, day_info in enumerate(season_days):
        day_label = (
            f"day_{day_idx + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        )
        for op_idx, name in enumerate(operational_var):
            gene_idx = len(design_var) + day_idx * num_operational_vars + op_idx
            res_dict[f"{day_label}_{name}"] = best_ind[gene_idx]

    res_table = tb.tabulate(
        res_dict.items(),
        tablefmt="grid",
    )

    logger.info(
        f"\n{'-' * 40}\n"
        f"GA Optimal solution ({season})\n"
        f"{'-' * 40}\n"
        f"Final Best Results\n"
        f"{res_table}"
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    result_logbook = pd.DataFrame([res_dict])

    result_logbook.index = result_logbook.index + 1

    result_logbook.index.name = "serial"

    file_name = PROJECT_ROOT / "results" / "GA_results" / f"{season}_{timestamp}.csv"

    file_name.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_logbook.to_csv(
        file_name,
        index=True,
    )

    # ---------------------------------------------------------
    # Save best solution JSON
    # ---------------------------------------------------------

    best_solution_path = PROJECT_ROOT / "results" / "GA_results" / f"best_solution_{season}.json"

    with open(
        best_solution_path,
        "w",
    ) as f:
        json.dump(
            best_solution,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # MLflow artifacts
    # ---------------------------------------------------------

    mlflow.log_artifact(
        str(file_name),
        artifact_path="results",
    )

    mlflow.log_artifact(
        str(best_solution_path),
        artifact_path="results",
    )

    # Save GA logbook
    logbook_path = PROJECT_ROOT / "results" / "GA_results" / f"{season}_logbook.csv"

    pd.DataFrame(logbook).to_csv(
        logbook_path,
        index=False,
    )

    mlflow.log_artifact(
        str(logbook_path),
        artifact_path="ga_history",
    )

    # Log final plot
    if plot_path.exists():
        mlflow.log_artifact(
            str(plot_path),
            artifact_path="plots",
        )

    logger.info(f"Best solution: {best_solution}")

    logger.info(f"Best fitness: {best_fitness}")

    return (
        best_solution,
        best_fitness,
        {
            "pop": pop,
            "logbook": logbook,
            "hof": hof,
        },
    )


if __name__ == "__main__":
    from main import main

    main()

