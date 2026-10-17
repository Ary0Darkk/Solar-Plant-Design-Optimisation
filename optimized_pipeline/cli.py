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
        "--job-id",
        type=str,
        default=None,
        help="HPC scheduler or run Job ID for MLflow tracking (defaults to PBS_JOBID / SLURM_JOB_ID)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable remote DagsHub/MLflow experiment tracking",
    )
    return parser.parse_args(argv)


def collect_run_tags(args, rank: int, size: int, workers: int) -> dict:
    """Collect scheduler metadata, job_id, and system properties for MLflow."""
    import socket
    import subprocess

    job_id = (
        getattr(args, "job_id", None)
        or os.environ.get("PBS_JOBID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or f"local_{os.getpid()}"
    )
    job_name = (
        os.environ.get("PBS_JOBNAME") or os.environ.get("SLURM_JOB_NAME") or f"GA_Opt_{args.season}"
    )
    queue = os.environ.get("PBS_QUEUE") or os.environ.get("SLURM_PARTITION") or "local"
    total_cpus = (
        os.environ.get("PBS_NCPUS") or os.environ.get("APP_CPU_COUNT") or str(workers * size)
    )
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous"
    hostname = socket.gethostname()

    tags = {
        "pipeline": "optimized_pipeline",
        "season": str(args.season),
        "algorithm": "DEAP GA",
        "memory_mode": "24h_slice_recycle",
        "job_id": str(job_id),
        "job_name": str(job_name),
        "queue": str(queue),
        "scheduler": "PBS"
        if "PBS_JOBID" in os.environ
        else "Slurm"
        if "SLURM_JOB_ID" in os.environ
        else "Local",
        "mpi_size": str(size),
        "workers_per_rank": str(workers),
        "total_cpus": str(total_cpus),
        "hostname": str(hostname),
        "user": str(user),
    }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
        if commit:
            tags["git_commit"] = commit
    except Exception:
        pass

    return tags


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
    run_tags = collect_run_tags(args, rank=rank, size=size, workers=workers)
    job_id = run_tags.get("job_id")

    logger.info("=" * 60)
    logger.info("Concentrated Solar Power (CSP) Genetic Algorithm Optimization")
    logger.info(
        "Season: %s | Pop: %d | Gens: %d | Workers: %d | Rank: %d/%d",
        args.season,
        config.ga.pop_size,
        config.ga.num_generations,
        workers,
        rank,
        size,
    )
    logger.info(
        "Scheduler Job ID: %s | Queue: %s | Host: %s",
        job_id,
        run_tags.get("queue"),
        run_tags.get("hostname"),
    )
    logger.info(
        "Hint: For a faster run, use: python deap_ga_optimiser.py --pop-size 14 --generations 5"
    )
    logger.info("=" * 60)

    if rank == 0:
        if not args.no_tracking:
            init_tracking()

        try:
            run_name = f"{run_tags.get('job_name', f'GA_Opt_{args.season}')}"
            if job_id and not job_id.startswith("local_"):
                run_name = f"{run_name}_{job_id}"

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(run_tags)
                mlflow.log_params(
                    {
                        "season": args.season,
                        "pop_size": config.ga.pop_size,
                        "generations": config.ga.num_generations,
                        "workers_per_rank": workers,
                        "max_tasks_per_child": config.ga.max_tasks_per_child,
                        "job_id": str(job_id),
                    }
                )

                ctx = mp.get_context()
                try:
                    with ctx.Pool(
                        processes=workers, maxtasksperchild=config.ga.max_tasks_per_child
                    ) as pool:
                        run_optimization(
                            season=args.season,
                            config=config,
                            pool=pool,
                            comm=comm,
                            rank=rank,
                            mpi_size=size,
                            job_id=job_id,
                        )
                finally:
                    # Notify remote MPI ranks to stop
                    for w in range(1, size):
                        comm.send(None, dest=w, tag=0)

        except Exception:
            # Fallback run without MLflow context if tracking fails
            ctx = mp.get_context()
            try:
                with ctx.Pool(
                    processes=workers, maxtasksperchild=config.ga.max_tasks_per_child
                ) as pool:
                    run_optimization(
                        season=args.season,
                        config=config,
                        pool=pool,
                        comm=comm,
                        rank=rank,
                        mpi_size=size,
                        job_id=job_id,
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
