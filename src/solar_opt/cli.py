"""
Command Line Interface (CLI) for season-wise CSP genetic algorithm optimization.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime

import mlflow

from .config.defaults import get_default_config
from .genetic.engine import run_seasonal_optimization, run_worker_loop
from .parallel.communicator import get_communicator
from .tracking.logger import setup_rank_logger
from .tracking.mlflow_tracker import init_tracking

# Ensure Windows consoles encode UTF-8 to prevent emoji / unicode crashes
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="solar-opt",
        description="Season-wise Concentrated Solar Power (CSP) genetic algorithm optimization",
    )
    parser.add_argument(
        "--season",
        default="summer",
        choices=["winter", "summer", "monsoon", "post"],
        help="Seasonal optimization case to execute (default: summer)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Optional override for number of GA generations",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Optional override for population size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Multiprocessing worker count per rank (default: from LOCAL_WORKERS or 4)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable remote DagsHub experiment tracking and use local MLflow only",
    )
    return parser.parse_args(args)


def main(args=None):
    """Main CLI entrypoint."""
    parsed = parse_args(args)
    season = parsed.season

    config = get_default_config()

    if parsed.generations is not None:
        config.ga.num_generations = parsed.generations
    if parsed.pop_size is not None:
        config.ga.pop_size = parsed.pop_size
        if config.ga.tournament_size > parsed.pop_size:
            config.ga.tournament_size = max(2, parsed.pop_size // 2)
        if config.ga.hall_of_fame_size > parsed.pop_size:
            config.ga.hall_of_fame_size = max(1, parsed.pop_size // 2)

    workers = parsed.workers
    if workers is None:
        workers = int(os.environ.get("LOCAL_WORKERS", "4"))

    comm = get_communicator()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Synchronize session timestamp from Rank 0
    session_timestamp = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if rank == 0 else None
    )
    session_timestamp = comm.bcast(session_timestamp, root=0)

    logger = setup_rank_logger(rank=rank, timestamp=session_timestamp)
    logger.info("NYS-Optimisation started!")
    logger.info("[Rank %d/%d] Starting optimization for season: %s", rank, size, season)

    # ---------------------------------------------------------
    # MASTER (Rank 0)
    # ---------------------------------------------------------
    if rank == 0:
        if not parsed.no_tracking:
            init_tracking()

        mlflow.set_experiment("CSP_Seasonal_GA")

        with mlflow.start_run(run_name=f"GA_{season}"):
            mlflow.set_tags(
                {
                    "project": "CSP Plant Optimisation",
                    "season": season,
                    "parallelization": "MPI + multiprocessing",
                    "algorithm": "DEAP Genetic Algorithm",
                    "simulation": "PySAM",
                }
            )

            mlflow.log_params(
                {
                    "season": season,
                    "random_seed": config.ga.random_seed,
                    "penalty": config.simulation.penalty,
                    "population_size": config.ga.pop_size,
                    "num_generations": config.ga.num_generations,
                    "cxpb": config.ga.cxpb,
                    "mutpb": config.ga.mutpb,
                    "indpb": config.ga.indpb,
                    "tournament_size": config.ga.tournament_size,
                    "hall_of_fame_size": config.ga.hall_of_fame_size,
                    "num_days": len(config.seasons[season]),
                    "mpi_ranks": size,
                    "workers_per_rank": workers,
                    "configured_total_workers": size * workers,
                    "design_variables": json.dumps(list(config.design.names)),
                    "operational_variables": json.dumps(list(config.operational.names)),
                    "design_lower_bounds": json.dumps(list(config.design.lb)),
                    "design_upper_bounds": json.dumps(list(config.design.ub)),
                    "operational_lower_bounds": json.dumps(list(config.operational.lb)),
                    "operational_upper_bounds": json.dumps(list(config.operational.ub)),
                    "electricity_min_price": config.tariff.min_price,
                    "electricity_max_price": config.tariff.max_price,
                }
            )

            logger.info("[Rank 0] MLflow run started: %s", mlflow.active_run().info.run_id)

            mp_context = mp.get_context()
            try:
                with mp_context.Pool(processes=workers, maxtasksperchild=100) as local_pool:
                    run_seasonal_optimization(
                        season=season,
                        config=config,
                        local_pool=local_pool,
                        comm=comm,
                        rank=rank,
                        mpi_size=size,
                    )
            finally:
                # Signal remote worker ranks to stop
                for w in range(1, size):
                    comm.send(None, dest=w, tag=0)

            logger.info("[Rank 0] MLflow run completed")

    # ---------------------------------------------------------
    # WORKERS (Rank > 0)
    # ---------------------------------------------------------
    else:
        mp_context = mp.get_context()
        with mp_context.Pool(processes=workers, maxtasksperchild=100) as worker_pool:
            run_worker_loop(
                season=season,
                comm=comm,
                rank=rank,
                config=config,
                local_pool=worker_pool,
            )

    logger.info("Optimisation Completed!")


if __name__ == "__main__":
    main()
