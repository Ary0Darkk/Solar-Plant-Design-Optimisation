"""
Genetic Algorithm optimization engine with True Elitism and memory optimization.
"""

import gc
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from deap import algorithms, base, tools

from .checkpoint import (
    compute_signature,
    find_compatible_checkpoint,
    save_compressed_checkpoint,
)
from .chromosome import (
    build_gene_generator,
    deserialize_pop,
    init_deap_types,
    serialize_pop,
)
from .config import PROJECT_ROOT, PipelineConfig
from .objective import compute_daily_objective_value
from .operators import build_bounded_mutation
from .parallel import Communicator, partition_batches
from .simulation import get_simulator, validate_constraints
from .tariff import get_shared_tariff
from .tracker import log_step_metrics, plot_and_export
from .weather import get_season_days

logger = logging.getLogger("Solar_Optimisation")


def evaluate_individual_season(
    individual: Sequence[int],
    season_days: Sequence[Dict[str, Any]],
    config: PipelineConfig,
    tariff_array: Optional[np.ndarray] = None,
) -> Tuple[Tuple[float], bool]:
    """
    Simulate PySAM across 7 representative days for an individual chromosome.

    Returns:
        ((total_fitness,), is_penalized: bool)
    """
    design_names = config.design.names
    op_names = config.operational.names
    n_design = len(design_names)
    n_op = len(op_names)

    # 1. Extract physical design overrides
    design_overrides = {design_names[i]: individual[i] for i in range(n_design)}
    if not validate_constraints(design_overrides):
        return (config.simulation.penalty,), True

    sim = get_simulator(config.simulation)
    tariff = tariff_array if tariff_array is not None else get_shared_tariff(config.tariff)

    total_fitness = 0.0

    # 2. Simulate each representative operational day
    for day_info in season_days:
        l_idx = day_info["local_index"]
        d_idx = day_info["day_index"]

        offset = n_design + l_idx * n_op
        daily_overrides = dict(design_overrides)
        for j in range(n_op):
            daily_overrides[op_names[j]] = individual[offset + j]

        if not validate_constraints(daily_overrides):
            return (config.simulation.penalty,), True

        sim_slice, penalized = sim.run_daily_simulation(daily_overrides, day_index=d_idx)
        if penalized or sim_slice is None:
            return (config.simulation.penalty,), True

        daily_revenue = compute_daily_objective_value(
            sim_slice, day_index=d_idx, shared_tariff_8760=tariff
        )
        total_fitness += daily_revenue

    return (total_fitness,), False


def _worker_eval_task(args: Tuple[Any, ...]) -> Tuple[Tuple[float], bool]:
    """Top-level helper function for multiprocessing pool execution."""
    ind_genes, season_days, config = args
    return evaluate_individual_season(ind_genes, season_days, config)


def evaluate_batch_memory_optimized(
    individuals: Sequence[Any],
    pool: Any,
    season: str,
    config: PipelineConfig,
    rank: int = 0,
    custom_logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], Dict[str, int]]:
    """Evaluate batch of individuals using the multiprocessing pool."""
    season_days = get_season_days(season, config.seasons)

    # Pre-cache tariff in parent before distributing
    get_shared_tariff(config.tariff)

    tasks = [(list(ind), season_days, config) for ind in individuals]
    if pool is not None:
        results = pool.map(_worker_eval_task, tasks)
    else:
        results = [_worker_eval_task(t) for t in tasks]

    succ = 0
    pen = 0
    for ind, (fit_val, is_penalized) in zip(individuals, results):
        ind.fitness.values = fit_val
        if is_penalized:
            pen += 1
        else:
            succ += 1

    return list(individuals), {"successful": succ, "penalized": pen}


def worker_event_loop(
    season: str,
    comm: Communicator,
    rank: int,
    config: PipelineConfig,
    workers: int = 4,
    custom_logger: Optional[logging.Logger] = None,
) -> None:
    """Event loop for remote MPI worker ranks awaiting candidate batches from Master Rank 0."""
    import multiprocessing as mp

    if rank == 0:
        raise ValueError("Rank 0 is designated as Master Rank and cannot run worker loop")

    log = custom_logger or logger
    log.info("[Rank %d] Worker initialized with %d processes.", rank, workers)

    ctx = mp.get_context()
    with ctx.Pool(processes=workers, maxtasksperchild=config.ga.max_tasks_per_child) as pool:
        while True:
            batch = comm.recv(source=0, tag=0)
            if batch is None:
                log.info("[Rank %d] Received termination signal. Exiting worker loop.", rank)
                break

            evaluated_batch, stats = evaluate_batch_memory_optimized(
                batch, pool, season, config, rank=rank, custom_logger=log
            )
            comm.send((evaluated_batch, stats), dest=0, tag=0)


def run_optimization(
    season: str,
    config: PipelineConfig,
    pool: Any,
    comm: Communicator,
    rank: int = 0,
    mpi_size: int = 1,
    job_id: Optional[str] = None,
    custom_logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Execute Genetic Algorithm optimization for the target season.

    Features:
    - True Elitism preserving Hall of Fame best individuals across generations
    - 24-hour slice PySAM simulation to eliminate C-level memory footprint
    - Atomic gzip-compressed state checkpointing and restoration
    - Generation-level progress logging and artifact generation
    """
    if rank != 0:
        raise ValueError("0th rank must be the master rank")

    log = custom_logger or logger
    init_deap_types()
    season_days = get_season_days(season, config.seasons)
    ga = config.ga
    pop_size = ga.pop_size
    num_generations = ga.num_generations
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    toolbox = base.Toolbox()
    gene_gen = build_gene_generator(
        design_lb=config.design.lb,
        design_ub=config.design.ub,
        operational_lb=config.operational.lb,
        operational_ub=config.operational.ub,
        num_days=len(season_days),
    )
    from deap import creator

    toolbox.register("individual", tools.initIterate, creator.Individual, gene_gen)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate",
        build_bounded_mutation(
            design_lb=config.design.lb,
            design_ub=config.design.ub,
            operational_lb=config.operational.lb,
            operational_ub=config.operational.ub,
            num_days=len(season_days),
            indpb=ga.indpb,
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=ga.tournament_size)

    # Statistics compiler
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Checkpoint signature & paths
    ckpt_sig = compute_signature(
        season=season,
        design_names=config.design.names,
        operational_names=config.operational.names,
        design_lb=config.design.lb,
        design_ub=config.design.ub,
        operational_lb=config.operational.lb,
        operational_ub=config.operational.ub,
        pop_size=pop_size,
        num_generations=num_generations,
        cxpb=ga.cxpb,
        mutpb=ga.mutpb,
        indpb=ga.indpb,
    )
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "GA_optimized" / season / ckpt_sig
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_file = ckpt_dir / "checkpoint_latest.pkl.gz"
    best_ckpt_file = ckpt_dir / "checkpoint_best.pkl.gz"

    expected_chrom_len = len(config.design.names) + len(season_days) * len(config.operational.names)

    # Checkpoint restoration
    cp = None
    if ga.resume_from_checkpoint:
        season_base_dir = PROJECT_ROOT / "checkpoints" / "GA_optimized" / season
        discovered = find_compatible_checkpoint(
            season_dir=season_base_dir,
            expected_key=ckpt_sig,
            pop_size=pop_size,
            chromosome_len=expected_chrom_len,
        )
        if discovered is not None:
            _, cp = discovered

    total_successful = 0
    total_penalized = 0
    best_fitness_ever = -float("inf")
    stagnant_generations = 0

    if cp is not None:
        log.info(
            "Resuming optimization from Gen %d (Best fitness: %.2f)",
            cp.get("generation", 0),
            cp.get("best_fitness_ever", -float("inf")),
        )
        total_successful = cp.get("total_successful", 0)
        total_penalized = cp.get("total_penalized", 0)
        best_fitness_ever = cp.get("best_fitness_ever", -float("inf"))
        stagnant_generations = cp.get("stagnant_generations", 0)
        random.setstate(cp["rndstate"])
        np.random.set_state(cp["np_rndstate"])
        pop = deserialize_pop(cp["pop"], toolbox)
        logbook = cp["logbook"]
        hof = tools.HallOfFame(ga.hall_of_fame_size)
        if "hof" in cp:
            hof.update(deserialize_pop(cp["hof"], toolbox))
        start_gen = cp["generation"] + 1
    else:
        log.info("Starting fresh GA optimization (pop=%d, gens=%d)", pop_size, num_generations)
        random.seed(ga.random_seed)
        np.random.seed(ga.random_seed)
        pop = toolbox.population(n=pop_size)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]
        hof = tools.HallOfFame(ga.hall_of_fame_size)
        start_gen = 0

    gens_log = [entry["gen"] for entry in logbook] if logbook else []
    max_log = [entry["max"] for entry in logbook] if logbook else []
    avg_log = [entry["avg"] for entry in logbook] if logbook else []

    def _evaluate_all(candidates):
        nonlocal total_successful, total_penalized
        batches = partition_batches(
            candidates, batch_size=max(4, len(candidates) // (mpi_size * 2))
        )
        evaluated_all = []
        wave_succ = 0
        wave_pen = 0

        if mpi_size <= 1:
            for b in batches:
                eval_b, b_stats = evaluate_batch_memory_optimized(
                    b, pool, season, config, rank, custom_logger=log
                )
                evaluated_all.extend(eval_b)
                wave_succ += b_stats["successful"]
                wave_pen += b_stats["penalized"]
        else:
            worker_ids = list(range(1, mpi_size))
            b_idx = 0
            w_counts = {w: 0 for w in worker_ids}
            for w in worker_ids:
                if b_idx < len(batches):
                    comm.send(batches[b_idx], dest=w, tag=0)
                    w_counts[w] += 1
                    b_idx += 1

            while b_idx < len(batches):
                eval_b, b_stats = evaluate_batch_memory_optimized(
                    batches[b_idx], pool, season, config, rank, custom_logger=log
                )
                evaluated_all.extend(eval_b)
                wave_succ += b_stats["successful"]
                wave_pen += b_stats["penalized"]
                b_idx += 1

            for w in worker_ids:
                for _ in range(w_counts[w]):
                    eval_b, b_stats = comm.recv(source=w, tag=0)
                    evaluated_all.extend(eval_b)
                    wave_succ += b_stats["successful"]
                    wave_pen += b_stats["penalized"]

        total_successful += wave_succ
        total_penalized += wave_pen
        return evaluated_all, wave_succ, wave_pen

    # Generation 0 Evaluation
    if start_gen == 0:
        pop, succ, pen = _evaluate_all(pop)
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        log_step_metrics(
            pop,
            hof,
            record,
            0,
            config.design.names,
            config.operational.names,
            season_days,
            succ,
            pen,
            total_successful,
            total_penalized,
        )
        gens_log.append(0)
        max_log.append(record["max"])
        avg_log.append(record["avg"])
        best_fitness_ever = record["max"]
        log.info("Gen 0 complete | Best Fitness: %.2f | Avg: %.2f", record["max"], record["avg"])

        if ga.checkpoint_interval == 1 or best_fitness_ever > -float("inf"):
            payload = {
                "season": season,
                "ckpt_key": ckpt_sig,
                "generation": 0,
                "best_fitness_ever": best_fitness_ever,
                "pop": serialize_pop(pop),
                "hof": serialize_pop(hof),
                "logbook": logbook,
                "rndstate": random.getstate(),
                "np_rndstate": np.random.get_state(),
                "total_successful": total_successful,
                "total_penalized": total_penalized,
                "stagnant_generations": stagnant_generations,
            }
            save_compressed_checkpoint(payload, latest_ckpt_file)
            save_compressed_checkpoint(payload, ckpt_dir / "checkpoint_gen_0.pkl.gz")
            save_compressed_checkpoint(payload, best_ckpt_file)
            del payload

    # Evolutionary Loop
    for gen in range(max(1, start_gen), num_generations):
        offspring = algorithms.varAnd(pop, toolbox, ga.cxpb, ga.mutpb)
        offspring, succ, pen = _evaluate_all(offspring)

        # True Elitism: preserve Hall of Fame best individuals
        elite_count = min(len(hof), max(1, pop_size // 2)) if pop_size <= len(hof) else len(hof)
        pop = toolbox.select(offspring, k=pop_size - elite_count)
        pop.extend([toolbox.clone(ind) for ind in hof[:elite_count]])
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        log_step_metrics(
            pop,
            hof,
            record,
            gen,
            config.design.names,
            config.operational.names,
            season_days,
            succ,
            pen,
            total_successful,
            total_penalized,
        )

        gens_log.append(gen)
        max_log.append(record["max"])
        avg_log.append(record["avg"])

        log.info(
            "Gen %d/%d | Best: %.2f | Avg: %.2f | Valid: %d, Pen: %d",
            gen,
            num_generations,
            record["max"],
            record["avg"],
            succ,
            pen,
        )

        current_best = record["max"]
        is_global_best = current_best > best_fitness_ever
        if is_global_best:
            best_fitness_ever = current_best
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if ga.early_stopping_patience > 0 and stagnant_generations >= ga.early_stopping_patience:
            log.info(
                "Early stopping triggered: no fitness improvement for %d consecutive generations.",
                stagnant_generations,
            )
            break

        if gen % ga.checkpoint_interval == 0 or gen == num_generations - 1 or is_global_best:
            payload = {
                "season": season,
                "ckpt_key": ckpt_sig,
                "generation": gen,
                "best_fitness_ever": best_fitness_ever,
                "pop": serialize_pop(pop),
                "hof": serialize_pop(hof),
                "logbook": logbook,
                "rndstate": random.getstate(),
                "np_rndstate": np.random.get_state(),
                "total_successful": total_successful,
                "total_penalized": total_penalized,
                "stagnant_generations": stagnant_generations,
            }
            save_compressed_checkpoint(payload, latest_ckpt_file)
            save_compressed_checkpoint(payload, ckpt_dir / f"checkpoint_gen_{gen}.pkl.gz")
            if is_global_best:
                save_compressed_checkpoint(payload, best_ckpt_file)
            del payload

        if ga.enable_gc_collection:
            del offspring
            gc.collect()

    best_ind = hof[0]
    best_fit = float(best_ind.fitness.values[0])

    sol_dict = plot_and_export(
        season=season,
        timestamp=timestamp,
        best_ind=best_ind,
        best_fitness=best_fit,
        design_names=config.design.names,
        operational_names=config.operational.names,
        season_days=season_days,
        logbook=logbook,
        gens=gens_log,
        max_fitness=max_log,
        avg_fitness=avg_log,
        job_id=job_id,
    )

    return sol_dict, best_fit, {"pop": pop, "logbook": logbook, "hof": hof}
