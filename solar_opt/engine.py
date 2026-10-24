"""
Core Genetic Algorithm Optimization Engine with True Elitism & Checkpointing.
"""

import multiprocessing as mp
import os
import random
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


def run_optimization(
    season: str,
    config: Optional[PipelineConfig] = None,
    workers: int = 4,
    comm: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute season-wise Genetic Algorithm optimization loop with Hall of Fame elitism.
    """
    if comm is not None and comm.Get_rank() != 0:
        raise RuntimeError("run_optimization must be called on rank 0")

    cfg = config or PipelineConfig()
    random.seed(cfg.ga.random_seed)
    np.random.seed(cfg.ga.random_seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = setup_logger(rank=0, timestamp=timestamp)
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

    job_id = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or "local"
    cfg_sig = compute_config_signature(cfg)

    # Multiprocessing Pool
    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
    pool = ctx.Pool(
        processes=workers,
        maxtasksperchild=cfg.ga.max_tasks_per_child,
    )

    try:
        # Generation 0 Evaluation
        eval_args = [(list(ind), season, cfg, tariff, season_days) for ind in pop]
        results = pool.map(_eval_worker, eval_args)

        gen_success = 0
        gen_penalized = 0
        for ind, (fit, succ) in zip(pop, results):
            ind.fitness.values = (fit,)
            if succ:
                gen_success += 1
            else:
                gen_penalized += 1

        cum_success += gen_success
        cum_penalized += gen_penalized

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        logger.info(
            f"Gen 0 | Max: ₹{record['max']:,.2f} | Avg: ₹{record['avg']:,.2f} | Succ: {gen_success}/{len(pop)}"
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
                eval_args = [(list(ind), season, cfg, tariff, season_days) for ind in invalid_ind]
                results = pool.map(_eval_worker, eval_args)

                gen_success = 0
                gen_penalized = 0
                for ind, (fit, succ) in zip(invalid_ind, results):
                    ind.fitness.values = (fit,)
                    if succ:
                        gen_success += 1
                    else:
                        gen_penalized += 1
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
                    f"Gen {gen:3d}/{cfg.ga.num_generations} | Max: ₹{record['max']:,.2f} | Avg: ₹{record['avg']:,.2f} | Valid: {gen_success}"
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

    best_individual = list(hof[0])
    best_fitness = float(hof[0].fitness.values[0])

    logger.info(
        f"Optimization Completed | Best Fitness: ₹{best_fitness:,.2f} | Total Successful Sims: {cum_success}"
    )

    # Export Artifacts (Plots, Excel, Report, JSON)
    artifacts = plot_and_export(
        season=season,
        timestamp=timestamp,
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


def worker_event_loop(comm: Any) -> None:
    """Event loop for MPI worker ranks (non-zero ranks)."""
    if comm.Get_rank() == 0:
        raise RuntimeError("worker_event_loop must only be executed by rank > 0")
    comm.Barrier()
