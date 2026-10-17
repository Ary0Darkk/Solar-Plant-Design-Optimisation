"""
Memory-efficient DEAP Genetic Algorithm evolutionary search engine and MPI coordinator.
"""

import gc
import logging
import multiprocessing as mp
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from .checkpoint import (
    compute_signature,
    load_compressed_checkpoint,
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
from .simulation import get_simulator
from .tariff import get_shared_tariff
from .tracker import log_step_metrics, plot_and_export
from .weather import get_season_days

logger = logging.getLogger("NYS_Optimisation")


def _eval_single_task(
    args: Tuple[int, int, Dict[str, Any], int],
) -> Tuple[int, int, Optional[float], bool]:
    """
    Worker task: execute PySAM 24h simulation, compute scalar objective, and discard simulation arrays.
    Returns: (ind_idx, local_day_idx, scalar_objective or None, is_penalized)
    """
    ind_idx, local_day_idx, overrides, day_index = args

    sim = get_simulator()
    sim_slice, is_penalized = sim.run_daily_simulation(overrides, day_index=day_index)

    if is_penalized or sim_slice is None:
        return ind_idx, local_day_idx, None, True

    try:
        obj = compute_daily_objective_value(sim_slice, day_index=day_index)
        return ind_idx, local_day_idx, obj, False
    except Exception:
        return ind_idx, local_day_idx, None, True
    finally:
        del sim_slice


def evaluate_batch_memory_optimized(
    batch: Sequence[Any],
    pool: Any,
    season: str,
    config: PipelineConfig,
    rank: int = 0,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Evaluate batch using streaming imap_unordered to minimize memory retention.
    """
    season_days = get_season_days(season, config.seasons)
    num_days = len(season_days)
    n_design = len(config.design.names)
    n_op = len(config.operational.names)

    tasks: List[Tuple[int, int, Dict[str, Any], int]] = []
    for ind_idx, ind in enumerate(batch):
        design_dict = {name: ind[i] for i, name in enumerate(config.design.names)}
        for day_info in season_days:
            l_idx = day_info["local_index"]
            offset = n_design + l_idx * n_op
            op_dict = {name: ind[offset + j] for j, name in enumerate(config.operational.names)}
            overrides = {**design_dict, **op_dict}
            tasks.append((ind_idx, l_idx, overrides, day_info["day_index"]))

    # Stream task results generator
    # Stream task results generator with low chunksize for immediate responsive streaming
    chunksize = min(4, max(1, len(tasks) // (getattr(pool, "_processes", 4) * 4)))
    results_by_ind: Dict[int, List[Tuple[int, Optional[float], bool]]] = {
        i: [] for i in range(len(batch))
    }
    total_tasks = len(tasks)
    done_tasks = 0
    log_interval = max(5, min(25, total_tasks // 10))

    for ind_idx, l_idx, obj, pen in pool.imap_unordered(
        _eval_single_task, tasks, chunksize=chunksize
    ):
        results_by_ind[ind_idx].append((l_idx, obj, pen))
        done_tasks += 1
        if done_tasks == 1 or done_tasks % log_interval == 0 or done_tasks == total_tasks:
            logger.info(
                "[Rank %d] Simulation progress: %d/%d tasks evaluated (%.1f%%)",
                rank,
                done_tasks,
                total_tasks,
                100.0 * done_tasks / total_tasks,
            )

    successful = 0
    penalized = 0

    for ind_idx, ind in enumerate(batch):
        day_results = results_by_ind[ind_idx]
        any_pen = any(pen for _, _, pen in day_results)

        if any_pen or len(day_results) < num_days:
            ind.fitness.values = (config.simulation.penalty,)
            penalized += len(day_results)
        else:
            total_fit = sum(val for _, val, _ in day_results if val is not None)
            ind.fitness.values = (float(total_fit),)
            successful += len(day_results)

    # Free task dictionaries and temporary structures
    del tasks
    del results_by_ind

    return list(batch), {
        "successful": successful,
        "penalized": penalized,
        "total": successful + penalized,
    }


def worker_event_loop(
    season: str,
    comm: Communicator,
    rank: int,
    config: PipelineConfig,
    workers: int = 4,
) -> None:
    """Worker loop listening for batches from Master Rank 0."""
    logger.info("[Rank %d] Worker loop active | Season: %s", rank, season)
    ctx = mp.get_context()

    with ctx.Pool(processes=workers, maxtasksperchild=config.ga.max_tasks_per_child) as pool:
        try:
            while True:
                batch = comm.recv(source=0, tag=0)
                if batch is None:
                    break

                eval_b, stats = evaluate_batch_memory_optimized(
                    batch=batch,
                    pool=pool,
                    season=season,
                    config=config,
                    rank=rank,
                )
                comm.send((eval_b, stats), dest=0, tag=0)

                # Explicit wave garbage collection
                if config.ga.enable_gc_collection:
                    gc.collect()
        finally:
            logger.info("[Rank %d] Worker loop shutdown.", rank)


def run_optimization(
    season: str,
    config: PipelineConfig,
    pool: Any,
    comm: Communicator,
    rank: int = 0,
    mpi_size: int = 1,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """Master optimization orchestration loop with memory recycling."""
    init_deap_types()
    season_days = get_season_days(season, config.seasons)
    num_days = len(season_days)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pre-load shared tariff into read-only memory
    get_shared_tariff(config.tariff)

    ga = config.ga
    pop_size = ga.pop_size
    num_generations = ga.num_generations
    chromosome_len = len(config.design.names) + num_days * len(config.operational.names)

    toolbox = base.Toolbox()
    gen_fn = build_gene_generator(
        design_lb=config.design.lb,
        design_ub=config.design.ub,
        operational_lb=config.operational.lb,
        operational_ub=config.operational.ub,
        num_days=num_days,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, gen_fn)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    tourn_size = min(ga.tournament_size, max(2, pop_size // 2)) if pop_size > 2 else 1
    toolbox.register("select", tools.selTournament, tournsize=tourn_size)
    toolbox.register("mate", tools.cxTwoPoint)

    mut_fn = build_bounded_mutation(
        design_lb=config.design.lb,
        design_ub=config.design.ub,
        operational_lb=config.operational.lb,
        operational_ub=config.operational.ub,
        num_days=num_days,
        indpb=ga.indpb,
    )
    toolbox.register("mutate", mut_fn)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Checkpoint configuration
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
    latest_ckpt_file = ckpt_dir / "checkpoint_latest.pkl.gz"
    best_ckpt_file = ckpt_dir / "checkpoint_best.pkl.gz"

    cp = None
    if ga.resume_from_checkpoint:
        cp = load_compressed_checkpoint(latest_ckpt_file, ckpt_sig, pop_size, chromosome_len)

    total_successful = 0
    total_penalized = 0
    best_fitness_ever = -float("inf")
    stagnant_generations = 0

    if cp is not None:
        logger.info("Resuming from compressed checkpoint at gen %d", cp["generation"])
        total_successful = cp.get("total_successful", 0)
        total_penalized = cp.get("total_penalized", 0)
        best_fitness_ever = cp.get("best_fitness_ever", -float("inf"))
        random.setstate(cp["rndstate"])
        np.random.set_state(cp["np_rndstate"])
        pop = deserialize_pop(cp["pop"], toolbox)
        logbook = cp["logbook"]
        hof = tools.HallOfFame(ga.hall_of_fame_size)
        if "hof" in cp:
            hof.update(deserialize_pop(cp["hof"], toolbox))
        start_gen = cp["generation"] + 1
    else:
        logger.info("Starting fresh GA optimization (pop=%d, gens=%d)", pop_size, num_generations)
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
                eval_b, b_stats = evaluate_batch_memory_optimized(b, pool, season, config, rank)
                evaluated_all.extend(eval_b)
                wave_succ += b_stats["successful"]
                wave_pen += b_stats["penalized"]
        else:
            # Distribute across workers
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
                    batches[b_idx], pool, season, config, rank
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

    # Initial evaluation
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
        logger.info("Gen 0 complete | Best Fitness: %.2f | Avg: %.2f", record["max"], record["avg"])

    # Evolutionary Loop
    for gen in range(max(1, start_gen), num_generations):
        offspring = algorithms.varAnd(pop, toolbox, ga.cxpb, ga.mutpb)
        offspring, succ, pen = _evaluate_all(offspring)

        # True Elitism: retain Hall of Fame elites, fill rest from offspring
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

        logger.info(
            "Gen %d/%d | Best: %.2f | Avg: %.2f | Valid: %d, Pen: %d",
            gen,
            num_generations,
            record["max"],
            record["avg"],
            succ,
            pen,
        )

        # Compressed checkpointing: keep latest + global best exclusively
        current_best = record["max"]
        is_global_best = current_best > best_fitness_ever
        if is_global_best:
            best_fitness_ever = current_best
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if ga.early_stopping_patience > 0 and stagnant_generations >= ga.early_stopping_patience:
            logger.info(
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
            }
            save_compressed_checkpoint(payload, latest_ckpt_file)
            if is_global_best:
                save_compressed_checkpoint(payload, best_ckpt_file)
            del payload

        # Garbage collection per wave
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
    )

    return sol_dict, best_fit, {"pop": pop, "logbook": logbook, "hof": hof}
