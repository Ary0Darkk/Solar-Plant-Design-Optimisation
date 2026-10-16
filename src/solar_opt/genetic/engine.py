"""
DEAP Genetic Algorithm evolutionary search engine and MPI master/worker coordination.
"""

import logging
import multiprocessing as mp
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deap import algorithms, base, creator, tools
import mlflow
import numpy as np

from ..config.defaults import PROJECT_ROOT
from ..config.models import OptimizationConfig
from ..parallel.batching import batch_items
from ..parallel.communicator import Communicator
from ..simulation.weather import get_season_days
from ..tracking.exporter import (
    export_optimization_results,
    format_solution_table,
    plot_convergence,
)
from ..tracking.mlflow_tracker import log_generation_metrics
from .checkpoint import (
    compute_checkpoint_key,
    load_checkpoint,
    safe_pickle_save,
)
from .chromosome import (
    create_individual_generator,
    deserialize_population,
    serialize_population,
    setup_deap_primitives,
)
from .evaluator import evaluate_batch
from .operators import create_bounded_integer_mutation

logger = logging.getLogger("NYS_Optimisation")


def run_worker_loop(
    season: str,
    comm: Communicator,
    rank: int,
    config: OptimizationConfig,
    local_pool: Optional[Any] = None,
) -> None:
    """Worker event loop processing candidate batches sent from Master rank 0."""
    logger.info("[Rank %d] Worker initialized for season %s", rank, season)

    pool = local_pool
    owns_pool = False
    if pool is None:
        workers = int(getattr(config.ga, "workers_per_rank", 4))
        pool = mp.get_context().Pool(processes=workers, maxtasksperchild=100)
        owns_pool = True

    try:
        while True:
            # tag 0: batch payload, tag 1: termination signal
            batch = comm.recv(source=0, tag=0)
            if batch is None:
                logger.info("[Rank %d] Received termination signal. Exiting worker loop.", rank)
                break

            evaluated_batch, stats = evaluate_batch(
                batch=batch,
                local_pool=pool,
                season=season,
                design_vars=config.design.names,
                operational_vars=config.operational.names,
                penalty_value=config.simulation.penalty,
                min_price=config.tariff.min_price,
                max_price=config.tariff.max_price,
                rank=rank,
            )

            comm.send((evaluated_batch, stats), dest=0, tag=0)
    finally:
        if owns_pool and pool is not None:
            pool.close()
            pool.join()


def distribute_and_evaluate(
    population: Sequence[Any],
    local_pool: Any,
    season: str,
    comm: Communicator,
    config: OptimizationConfig,
    rank: int = 0,
    mpi_size: int = 1,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Distribute batches across MPI ranks and evaluate Rank 0's assigned portion locally.
    """
    batch_size = max(4, len(population) // (mpi_size * 2)) if len(population) > 4 else len(population)
    batches = batch_items(population, batch_size=batch_size)
    num_batches = len(batches)

    # In standalone single-rank mode
    if mpi_size <= 1:
        all_evaluated: List[Any] = []
        total_successful = 0
        total_penalized = 0

        for b in batches:
            eval_b, stats = evaluate_batch(
                batch=b,
                local_pool=local_pool,
                season=season,
                design_vars=config.design.names,
                operational_vars=config.operational.names,
                penalty_value=config.simulation.penalty,
                min_price=config.tariff.min_price,
                max_price=config.tariff.max_price,
                rank=rank,
            )
            all_evaluated.extend(eval_b)
            total_successful += stats["successful"]
            total_penalized += stats["penalized"]

        return all_evaluated, {
            "successful": total_successful,
            "penalized": total_penalized,
            "total": total_successful + total_penalized,
        }

    # In MPI multi-rank mode
    # Dispatch batches round-robin to workers (ranks 1 .. mpi_size-1)
    remote_results: List[Any] = []
    worker_batch_count: Dict[int, int] = {w: 0 for w in range(1, mpi_size)}

    # Send initial round
    batch_idx = 0
    for w in range(1, mpi_size):
        if batch_idx < num_batches:
            comm.send(batches[batch_idx], dest=w, tag=0)
            worker_batch_count[w] += 1
            batch_idx += 1

    # Local processing on Rank 0 while workers evaluate
    local_evaluated: List[Any] = []
    local_successful = 0
    local_penalized = 0

    while batch_idx < num_batches:
        eval_b, stats = evaluate_batch(
            batch=batches[batch_idx],
            local_pool=local_pool,
            season=season,
            design_vars=config.design.names,
            operational_vars=config.operational.names,
            penalty_value=config.simulation.penalty,
            min_price=config.tariff.min_price,
            max_price=config.tariff.max_price,
            rank=rank,
        )
        local_evaluated.extend(eval_b)
        local_successful += stats["successful"]
        local_penalized += stats["penalized"]
        batch_idx += 1

    # Collect responses from all worker ranks
    for w in range(1, mpi_size):
        for _ in range(worker_batch_count[w]):
            eval_b, stats = comm.recv(source=w, tag=0)
            remote_results.extend(eval_b)
            local_successful += stats["successful"]
            local_penalized += stats["penalized"]

    all_evaluated = local_evaluated + remote_results
    return all_evaluated, {
        "successful": local_successful,
        "penalized": local_penalized,
        "total": local_successful + local_penalized,
    }


def run_seasonal_optimization(
    season: str,
    config: OptimizationConfig,
    local_pool: Any,
    comm: Communicator,
    rank: int = 0,
    mpi_size: int = 1,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Run the season-wise genetic algorithm optimization pipeline.
    """
    setup_deap_primitives()

    season_days = get_season_days(season, seasons_dict=config.seasons)
    num_days = len(season_days)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    design_names = config.design.names
    design_lb = config.design.lb
    design_ub = config.design.ub
    design_types = config.design.types

    op_names = config.operational.names
    op_lb = config.operational.lb
    op_ub = config.operational.ub
    op_types = config.operational.types

    ga = config.ga
    pop_size = ga.pop_size
    num_generations = ga.num_generations

    chromosome_length = len(design_names) + num_days * len(op_names)

    # Initialize DEAP Toolbox
    toolbox = base.Toolbox()
    gen_ind = create_individual_generator(
        design_lb=design_lb,
        design_ub=design_ub,
        operational_lb=op_lb,
        operational_ub=op_ub,
        num_days=num_days,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, gen_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    tourn_size = ga.tournament_size
    if tourn_size > pop_size:
        tourn_size = max(2, pop_size // 2) if pop_size >= 2 else 1

    toolbox.register("select", tools.selTournament, tournsize=tourn_size)
    toolbox.register("mate", tools.cxTwoPoint)

    mutation_op = create_bounded_integer_mutation(
        design_lb=design_lb,
        design_ub=design_ub,
        operational_lb=op_lb,
        operational_ub=op_ub,
        num_days=num_days,
        indpb=ga.indpb,
    )
    toolbox.register("mutate", mutation_op)

    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Checkpoint configuration
    ckpt_key = compute_checkpoint_key(
        season=season,
        design_vars=design_names,
        operational_vars=op_names,
        design_types=design_types,
        operational_types=op_types,
        design_lb=design_lb,
        design_ub=design_ub,
        operational_lb=op_lb,
        operational_ub=op_ub,
        pop_size=pop_size,
        num_generations=num_generations,
        cxpb=ga.cxpb,
        mutpb=ga.mutpb,
        indpb=ga.indpb,
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "GA" / season / ckpt_key
    latest_checkpoint = checkpoint_dir / "checkpoint_latest.pkl"

    run_successful_total = 0
    run_penalized_total = 0
    start_gen = 0

    cp = None
    if ga.resume_from_checkpoint:
        cp = load_checkpoint(latest_checkpoint, ckpt_key, pop_size, chromosome_length)

    if cp is not None:
        logger.info("Resuming from checkpoint %s at generation %d", latest_checkpoint, cp["generation"])
        run_successful_total = cp.get("run_successful_total", 0)
        run_penalized_total = cp.get("run_penalized_total", 0)
        random.setstate(cp["rndstate"])
        np.random.set_state(cp["np_rndstate"])
        pop = deserialize_population(cp["pop"], toolbox)
        logbook = cp["logbook"]
        hof = tools.HallOfFame(ga.hall_of_fame_size)
        if "hof" in cp:
            hof.update(deserialize_population(cp["hof"], toolbox))
        start_gen = cp["generation"] + 1
    else:
        logger.info("Starting fresh GA optimization run (pop_size=%d, generations=%d)", pop_size, num_generations)
        random.seed(ga.random_seed)
        np.random.seed(ga.random_seed)
        pop = toolbox.population(n=pop_size)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]
        hof = tools.HallOfFame(ga.hall_of_fame_size)
        start_gen = 0

    gens_log = [entry["gen"] for entry in logbook] if logbook else []
    max_fitness_log = [entry["max"] for entry in logbook] if logbook else []
    avg_fitness_log = [entry["avg"] for entry in logbook] if logbook else []

    plot_path = PROJECT_ROOT / "plots" / "GA_plots" / f"{season}_fitness_vs_gen_{timestamp}.png"

    # Evaluate initial generation if starting fresh
    if start_gen == 0:
        pop, initial_stats = distribute_and_evaluate(
            population=pop,
            local_pool=local_pool,
            season=season,
            comm=comm,
            config=config,
            rank=rank,
            mpi_size=mpi_size,
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
            design_var=design_names,
            operational_var=op_names,
            season_days=season_days,
            simulation_successful=initial_stats["successful"],
            simulation_penalized=initial_stats["penalized"],
            run_successful_total=run_successful_total,
            run_penalized_total=run_penalized_total,
        )

        gens_log.append(0)
        max_fitness_log.append(record["max"])
        avg_fitness_log.append(record["avg"])
        logger.info("Gen 0 complete | Best Fitness: %.2f | Avg Fitness: %.2f", record["max"], record["avg"])

    # GA Evolutionary Generations Loop
    for gen in range(max(1, start_gen), num_generations):
        offspring = algorithms.varAnd(pop, toolbox, ga.cxpb, ga.mutpb)

        offspring, eval_stats = distribute_and_evaluate(
            population=offspring,
            local_pool=local_pool,
            season=season,
            comm=comm,
            config=config,
            rank=rank,
            mpi_size=mpi_size,
        )

        run_successful_total += eval_stats["successful"]
        run_penalized_total += eval_stats["penalized"]

        # Elitism: preserve Hall of Fame individuals, select remaining from offspring
        elite_count = min(len(hof), max(1, pop_size // 2)) if pop_size <= len(hof) else len(hof)
        pop = toolbox.select(offspring, k=pop_size - elite_count)
        pop.extend([toolbox.clone(ind) for ind in hof[:elite_count]])
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(offspring), **record)

        log_generation_metrics(
            population=pop,
            hof=hof,
            record=record,
            nevals=len(offspring),
            gen=gen,
            design_var=design_names,
            operational_var=op_names,
            season_days=season_days,
            simulation_successful=eval_stats["successful"],
            simulation_penalized=eval_stats["penalized"],
            run_successful_total=run_successful_total,
            run_penalized_total=run_penalized_total,
        )

        gens_log.append(gen)
        max_fitness_log.append(record["max"])
        avg_fitness_log.append(record["avg"])

        logger.info(
            "Gen %d/%d complete | Best Fitness: %.2f | Avg: %.2f | Success: %d, Penalized: %d",
            gen,
            num_generations,
            record["max"],
            record["avg"],
            eval_stats["successful"],
            eval_stats["penalized"],
        )

        # Periodic atomic checkpoint save
        if gen % ga.checkpoint_interval == 0 or gen == num_generations - 1:
            cp_payload = {
                "season": season,
                "ckpt_key": ckpt_key,
                "generation": gen,
                "pop": serialize_population(pop),
                "hof": serialize_population(hof),
                "logbook": logbook,
                "rndstate": random.getstate(),
                "np_rndstate": np.random.get_state(),
                "run_successful_total": run_successful_total,
                "run_penalized_total": run_penalized_total,
            }
            safe_pickle_save(cp_payload, latest_checkpoint)
            gen_ckpt = checkpoint_dir / f"checkpoint_gen_{gen}.pkl"
            safe_pickle_save(cp_payload, gen_ckpt)

    # Plot convergence
    plot_convergence(
        gens=gens_log,
        max_fitness=max_fitness_log,
        avg_fitness=avg_fitness_log,
        season=season,
        output_path=plot_path,
    )

    best_ind = hof[0]
    best_fitness = float(best_ind.fitness.values[0])

    sol_dict, csv_path, json_path, logbook_path = export_optimization_results(
        season=season,
        timestamp=timestamp,
        best_ind=best_ind,
        best_fitness=best_fitness,
        design_vars=design_names,
        operational_vars=op_names,
        season_days=season_days,
        logbook=logbook,
        plot_path=plot_path,
    )

    table_str = format_solution_table(sol_dict)
    logger.info("\n%s\nOptimal Solution (%s):\n%s\n%s", "=" * 40, season, table_str, "=" * 40)

    return sol_dict, best_fitness, {
        "pop": pop,
        "logbook": logbook,
        "hof": hof,
        "csv_path": csv_path,
        "json_path": json_path,
    }
