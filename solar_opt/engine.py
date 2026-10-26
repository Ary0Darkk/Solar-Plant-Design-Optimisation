"""
Core Genetic Algorithm Optimization Engine with True Elitism & Checkpointing.
"""

import logging
import multiprocessing as mp
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import tools

from .checkpoint import compute_config_signature, save_compressed_checkpoint
from .chromosome import (
    build_toolbox,
    decode_individual,
    init_deap_types,
)
from .config import PipelineConfig
from .objective import compute_daily_objective_value
from .operators import build_bounded_crossover, build_bounded_mutation
from .simulation import get_simulator
from .tariff import get_shared_tariff
from .tracker import log_step_metrics, plot_and_export, setup_logger
from .weather import get_season_days


def evaluate_individual_season(
    ind: List[float],
    season: str,
    config: PipelineConfig,
    simulator: Any,
    tariff: np.ndarray,
    season_days: List[Dict[str, int]],
) -> Tuple[float, bool]:
    """
    Evaluate 26-gene individual across 7 representative days of the season.
    Returns: (total_net_profit, is_successful)
    """
    design_names = config.design.names
    op_names = config.operational.names
    penalty = getattr(config.simulation, "penalty", -1e6)

    total_net_profit = 0.0

    for d_idx, day_meta in enumerate(season_days):
        day_index = day_meta["day_index"]
        params = decode_individual(ind, design_names, op_names, day_offset=d_idx)

        sim_slice, penalized = simulator.run_daily_simulation(params, day_index=day_index)
        if penalized or sim_slice is None:
            return penalty, False

        start_h = day_index * 24
        tariff_slice = tariff[start_h : start_h + 24]
        net_profit, _, _, _ = compute_daily_objective_value(
            sim_slice, tariff_slice, land_area=params.get("specified_total_aperture", 10000.0)
        )
        total_net_profit += net_profit

    return total_net_profit, True


def _eval_worker(
    args: Tuple[List[float], str, Any, Any, List[Dict[str, int]]],
) -> Tuple[float, bool]:
    """Top-level worker function for multiprocessing pool."""
    ind, season, config, tariff, season_days = args
    sim = get_simulator(config.simulation)
    return evaluate_individual_season(ind, season, config, sim, tariff, season_days)


def _eval_worker_indexed(
    args: Tuple[int, List[float], str, Any, Any, List[Dict[str, int]]],
) -> Tuple[int, float, bool]:
    """Worker task returning (index, fitness, is_successful) to preserve ordering in async streams."""
    idx, ind, season, config, tariff, season_days = args
    sim = get_simulator(config.simulation)
    profit, succ = evaluate_individual_season(ind, season, config, sim, tariff, season_days)
    return idx, profit, succ


def _evaluate_population_parallel(
    candidates: List[Any],
    season: str,
    cfg: PipelineConfig,
    tariff: np.ndarray,
    season_days: List[Dict[str, int]],
    pool: Any,
    comm: Optional[Any] = None,
    logger: Optional[Any] = None,
    gen: int = 0,
) -> Tuple[List[Tuple[float, bool]], int, int]:
    """Evaluate candidate individuals across local multiprocessing pool or distributed MPI cluster."""
    if not candidates:
        return [], 0, 0

    size = comm.Get_size() if comm is not None else 1
    total = len(candidates)
    log_interval = 1 if total <= 10 else max(2, total // 10)

    if size <= 1:
        indexed_args = [
            (idx, list(ind), season, cfg, tariff, season_days) for idx, ind in enumerate(candidates)
        ]
        results_ordered: List[Optional[Tuple[float, bool]]] = [None] * total
        succ_count = 0
        pen_count = 0

        if logger:
            worker_count = getattr(pool, "_processes", "pool")
            logger.info(
                f"[Gen {gen:3d}] Evaluating {total} candidates across {worker_count} local worker processes..."
            )

        start_time = time.time()
        for completed, (idx, profit, succ) in enumerate(
            pool.imap_unordered(_eval_worker_indexed, indexed_args), 1
        ):
            results_ordered[idx] = (profit, succ)
            if succ:
                succ_count += 1
            else:
                pen_count += 1

            if logger and (completed == 1 or completed % log_interval == 0 or completed == total):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    f"[Gen {gen:3d}] Progress: {completed}/{total} ({completed / total * 100:.1f}%) | "
                    f"Valid: {succ_count}/{completed} | Speed: {rate:.2f} ind/s | ETA: {eta:.0f}s"
                )

        return results_ordered, succ_count, pen_count

    # Multi-node MPI distribution
    from .parallel import partition_batches

    batches = partition_batches(candidates, size)

    # Dispatch sub-batches of plain gene lists to remote MPI worker ranks (1 .. size-1)
    for worker_rank in range(1, size):
        gene_batch = [list(ind) for ind in batches[worker_rank]]
        comm.send(gene_batch, dest=worker_rank, tag=1)

    # Master node (Rank 0) evaluates its assigned sub-batch locally with live progress
    master_batch = batches[0]
    master_indexed = [
        (idx, list(ind), season, cfg, tariff, season_days) for idx, ind in enumerate(master_batch)
    ]
    master_results: List[Optional[Tuple[float, bool]]] = [None] * len(master_batch)

    if logger:
        logger.info(
            f"[Gen {gen:3d}] Master Rank 0 evaluating {len(master_batch)} candidates locally..."
        )

    for idx, profit, succ in pool.imap_unordered(_eval_worker_indexed, master_indexed):
        master_results[idx] = (profit, succ)

    all_results = list(master_results)

    # Collect evaluated results in deterministic rank order from remote workers
    for worker_rank in range(1, size):
        if logger:
            logger.info(
                f"[Gen {gen:3d}] Awaiting results from Worker Rank {worker_rank}/{size - 1}..."
            )
        worker_results = comm.recv(source=worker_rank, tag=2)
        all_results.extend(worker_results)
        if logger:
            logger.info(
                f"[Gen {gen:3d}] Received {len(worker_results)} evaluations from Worker Rank {worker_rank}"
            )

    succ = sum(1 for _, s in all_results if s)
    pen = len(all_results) - succ
    return all_results, succ, pen


def run_optimization(
    season: str,
    config: Optional[PipelineConfig] = None,
    workers: int = 4,
    comm: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
    custom_logger: Optional[logging.Logger] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute season-wise Genetic Algorithm optimization loop with Hall of Fame elitism.
    """
    if comm is not None and comm.Get_rank() != 0:
        raise RuntimeError("run_optimization must be called on rank 0")

    cfg = config or PipelineConfig()
    random.seed(cfg.ga.random_seed)
    np.random.seed(cfg.ga.random_seed)

    ts_str = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = custom_logger or setup_logger(rank=0, timestamp=ts_str)
    logger.info(f"Starting CSP GA Optimization | Season: {season.upper()} | Workers: {workers}")

    season_days = get_season_days(season, cfg.seasons)
    tariff = get_shared_tariff(cfg.tariff)
    init_deap_types()

    toolbox = build_toolbox(cfg.design, cfg.operational, num_days=len(season_days))

    # Register GA evolutionary operators
    mutate_op = build_bounded_mutation(
        cfg.design, cfg.operational, eta_m=cfg.ga.eta_m, indpb=0.1, num_days=len(season_days)
    )
    mate_op = build_bounded_crossover(
        cfg.design, cfg.operational, eta_c=cfg.ga.eta_c, num_days=len(season_days)
    )
    toolbox.register("mutate", mutate_op)
    toolbox.register("mate", mate_op)
    toolbox.register("select", tools.selTournament, tournsize=cfg.ga.tournament_size)

    # Initialize Population & Elitism Hall of Fame
    pop = toolbox.population(n=cfg.ga.pop_size)
    hof = tools.HallOfFame(cfg.ga.elitism_count)

    # Statistics & Logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]

    gens_history = []
    max_history = []
    avg_history = []

    cum_success = 0
    cum_penalized = 0

    job_id = os.environ.get("PBS_JOBID", "local")
    cfg_sig = compute_config_signature(cfg)

    # Multiprocessing Pool
    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
    pool = ctx.Pool(
        processes=workers,
        maxtasksperchild=cfg.ga.max_tasks_per_child,
    )

    try:
        # Generation 0 Evaluation
        results, gen_success, gen_penalized = _evaluate_population_parallel(
            pop, season, cfg, tariff, season_days, pool, comm, logger=logger, gen=0
        )

        for ind, (fit, _) in zip(pop, results):
            ind.fitness.values = (fit,)

        cum_success += gen_success
        cum_penalized += gen_penalized

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        logger.info(
            f"Gen 0 | Max: INR {record['max']:,.2f} | Avg: INR {record['avg']:,.2f} | Succ: {gen_success}/{len(pop)}"
        )

        gens_history.append(0)
        max_history.append(float(record["max"]))
        avg_history.append(float(record["avg"]))

        log_step_metrics(
            pop=pop,
            hof=hof,
            record=record,
            gen=0,
            design_names=cfg.design.names,
            operational_names=cfg.operational.names,
            season_days=season_days,
            sim_successful=gen_success,
            sim_penalized=gen_penalized,
            total_successful=cum_success,
            total_penalized=cum_penalized,
        )

        # Generational Evolution Loop
        for gen in range(1, cfg.ga.num_generations + 1):
            # Select next generation individuals (leaving room for elites)
            offspring = toolbox.select(pop, len(pop) - cfg.ga.elitism_count)
            offspring = list(map(toolbox.clone, offspring))

            # Apply Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cfg.ga.cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply Mutation
            for mutant in offspring:
                if random.random() < cfg.ga.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                results, gen_success, gen_penalized = _evaluate_population_parallel(
                    invalid_ind,
                    season,
                    cfg,
                    tariff,
                    season_days,
                    pool,
                    comm,
                    logger=logger,
                    gen=gen,
                )

                for ind, (fit, _) in zip(invalid_ind, results):
                    ind.fitness.values = (fit,)

                cum_success += gen_success
                cum_penalized += gen_penalized

            # Inject preserved elites (Hall of Fame)
            elites = [toolbox.clone(e) for e in hof]
            pop[:] = offspring + elites

            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            gens_history.append(gen)
            max_history.append(float(record["max"]))
            avg_history.append(float(record["avg"]))

            if gen % 10 == 0 or gen == cfg.ga.num_generations:
                logger.info(
                    f"Gen {gen:3d}/{cfg.ga.num_generations} | Max: INR {record['max']:,.2f} | Avg: INR {record['avg']:,.2f} | Valid: {gen_success}"
                )

            log_step_metrics(
                pop=pop,
                hof=hof,
                record=record,
                gen=gen,
                design_names=cfg.design.names,
                operational_names=cfg.operational.names,
                season_days=season_days,
                sim_successful=gen_success,
                sim_penalized=gen_penalized,
                total_successful=cum_success,
                total_penalized=cum_penalized,
            )

            # Periodic Checkpointing
            if checkpoint_dir and (gen % 25 == 0 or gen == cfg.ga.num_generations):
                ckpt_path = Path(checkpoint_dir) / f"checkpoint_{season}_gen_{gen}.pkl.gz"
                save_compressed_checkpoint(
                    ckpt_path,
                    population=pop,
                    generation=gen,
                    hall_of_fame=list(hof),
                    logbook=logbook,
                    signature=cfg_sig,
                )

    finally:
        pool.close()
        pool.join()
        # Gracefully signal all remote MPI worker ranks to terminate
        if comm is not None and comm.Get_size() > 1:
            for worker_rank in range(1, comm.Get_size()):
                comm.send(None, dest=worker_rank, tag=1)

    best_individual = list(hof[0])
    best_fitness = float(hof[0].fitness.values[0])

    logger.info(
        f"Optimization Completed | Best Fitness: INR {best_fitness:,.2f} | Total Successful Sims: {cum_success}"
    )

    # Export Artifacts (Plots, Excel, Report, JSON)
    artifacts = plot_and_export(
        season=season,
        timestamp=ts_str,
        best_ind=best_individual,
        best_fitness=best_fitness,
        design_names=cfg.design.names,
        operational_names=cfg.operational.names,
        season_days=season_days,
        logbook=logbook,
        gens=gens_history,
        max_fitness=max_history,
        avg_fitness=avg_history,
        job_id=job_id,
    )

    return artifacts


def worker_event_loop(
    comm: Any,
    season: str = "summer",
    config: Optional[PipelineConfig] = None,
    workers: int = 4,
) -> None:
    """
    Event loop for MPI worker ranks on multi-node HPC clusters.
    Awaits chromosome batches from Master Rank 0, evaluates them using local cores,
    and transmits evaluated results back to Rank 0.
    """
    if comm is None or comm.Get_rank() == 0:
        raise RuntimeError("worker_event_loop must only be executed by rank > 0")

    cfg = config or PipelineConfig()
    season_days = get_season_days(season, cfg.seasons)
    tariff = get_shared_tariff(cfg.tariff)

    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
    pool = ctx.Pool(
        processes=workers,
        maxtasksperchild=cfg.ga.max_tasks_per_child,
    )

    try:
        while True:
            batch = comm.recv(source=0, tag=1)
            if batch is None:
                # Received termination sentinel from Master Rank 0
                break

            if batch:
                indexed_args = [
                    (idx, list(ind), season, cfg, tariff, season_days)
                    for idx, ind in enumerate(batch)
                ]
                results_ordered: List[Optional[Tuple[float, bool]]] = [None] * len(batch)
                for idx, profit, succ in pool.imap_unordered(_eval_worker_indexed, indexed_args):
                    results_ordered[idx] = (profit, succ)
            else:
                results_ordered = []

            comm.send(results_ordered, dest=0, tag=2)
    finally:
        pool.close()
        pool.join()
