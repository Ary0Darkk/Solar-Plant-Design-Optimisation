"""
Command Line Interface (CLI) for optimized_pipeline.
"""

import argparse
import multiprocessing as mp
import os
import sys
from datetime import datetime

import mlflow

from .config import get_pipeline_config
from .engine import run_optimization, worker_event_loop
from .parallel import get_communicator
from .tracker import init_tracking, setup_logger

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_arguments(argv=None):
    """Parse CLI options with memory-control flags."""
    parser = argparse.ArgumentParser(
        prog="optimized_pipeline",
        description="High-throughput, memory-efficient CSP genetic algorithm optimizer",
    )
    parser.add_argument(
        "--season",
        default="summer",
        choices=["winter", "summer", "monsoon", "post"],
        help="Season to optimize (default: summer)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of GA generations override",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Population size override",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes per rank (default: from LOCAL_WORKERS or 4)",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=None,
        help="Recycle worker process after N simulations (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="Minimum electricity tariff in INR/kWh (default: 3.0)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Maximum electricity tariff in INR/kWh (default: 8.0)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable remote DagsHub/MLflow experiment tracking",
    )
    return parser.parse_args(argv)



def main(argv=None):
    """Main execution entry point."""
    args = parse_arguments(argv)
    config = get_pipeline_config()

    if args.generations is not None:
        config.ga.num_generations = args.generations
    if args.pop_size is not None:
        config.ga.pop_size = args.pop_size
    if args.max_tasks_per_child is not None:
        config.ga.max_tasks_per_child = args.max_tasks_per_child
    if args.seed is not None:
        config.ga.random_seed = args.seed
    if args.min_price is not None:
        config.tariff.min_price = args.min_price
    if args.max_price is not None:
        config.tariff.max_price = args.max_price


    workers = args.workers
    if workers is None:
        workers = int(os.environ.get("LOCAL_WORKERS", "4"))

    comm = get_communicator()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Synchronize timestamp from Rank 0
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if rank == 0 else None
    ts = comm.bcast(ts, root=0)

    logger = setup_logger(rank=rank, timestamp=ts)
    logger.info("[Rank %d/%d] Starting optimized pipeline for season: %s", rank, size, args.season)

    if rank == 0:
        tracking_active = False
        if not args.no_tracking:
            tracking_active = init_tracking()

        try:
            with mlflow.start_run(run_name=f"GA_Opt_{args.season}"):
                mlflow.set_tags(
                    {
                        "pipeline": "optimized_pipeline",
                        "season": args.season,
                        "algorithm": "DEAP GA",
                        "memory_mode": "24h_slice_recycle",
                    }
                )
                mlflow.log_params(
                    {
                        "season": args.season,
                        "pop_size": config.ga.pop_size,
                        "generations": config.ga.num_generations,
                        "workers_per_rank": workers,
                        "max_tasks_per_child": config.ga.max_tasks_per_child,
                    }
                )

                ctx = mp.get_context()
                try:
                    with ctx.Pool(processes=workers, maxtasksperchild=config.ga.max_tasks_per_child) as pool:
                        run_optimization(
                            season=args.season,
                            config=config,
                            pool=pool,
                            comm=comm,
                            rank=rank,
                            mpi_size=size,
                        )
                finally:
                    # Notify remote MPI ranks to stop
                    for w in range(1, size):
                        comm.send(None, dest=w, tag=0)

        except Exception:
            # Fallback run without MLflow context if tracking fails
            ctx = mp.get_context()
            try:
                with ctx.Pool(processes=workers, maxtasksperchild=config.ga.max_tasks_per_child) as pool:
                    run_optimization(
                        season=args.season,
                        config=config,
                        pool=pool,
                        comm=comm,
                        rank=rank,
                        mpi_size=size,
                    )
            finally:
                for w in range(1, size):
                    comm.send(None, dest=w, tag=0)

    else:
        # Worker ranks
        worker_event_loop(
            season=args.season,
            comm=comm,
            rank=rank,
            config=config,
            workers=workers,
        )

    logger.info("Optimized Pipeline Run Completed successfully!")


if __name__ == "__main__":
    main()
