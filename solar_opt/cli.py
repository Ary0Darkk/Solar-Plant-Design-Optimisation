"""
Command-Line Interface (CLI) and execution orchestration.
"""

import argparse
import multiprocessing as mp
import os
import sys
from datetime import datetime
from typing import List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mlflow

from .config import get_pipeline_config
from .engine import run_optimization, worker_event_loop
from .parallel import get_communicator
from .tracker import init_tracking, log_run_params, setup_logger


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line flags."""
    parser = argparse.ArgumentParser(
        description="Concentrated Solar Power (CSP) Genetic Algorithm Optimization using PySAM and DEAP."
    )
    parser.add_argument(
        "--season",
        type=str,
        default="summer",
        choices=["winter", "summer", "monsoon", "post"],
        help="Season to optimize (default: summer)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of GA generations override (default: from config, usually 100)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Population size override (default: from config, usually 112)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes per rank (default: 112 on HPC, automatically falls back to 4 locally)",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=None,
        help="Recycle worker process after N simulations to clear PySAM leaks (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 41)",
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
        help="Scheduler Job ID (auto-detected from PBS_JOBID or SLURM_JOB_ID)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="CSP_Seasonal_GA",
        help="MLflow / DagsHub experiment name (default: CSP_Seasonal_GA)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable remote DagsHub/MLflow experiment tracking",
    )
    return parser.parse_args(argv)


def resolve_worker_count(args_workers: Optional[int] = None) -> int:
    """
    Resolve number of multiprocessing workers per rank.

    Defaults to 112 (or LOCAL_WORKERS environment variable) on HPC clusters,
    but safely falls back to 4 (or available CPU count) when executed locally.
    """
    if args_workers is not None:
        return max(1, args_workers)

    target_workers = int(os.environ.get("LOCAL_WORKERS", "112"))
    is_hpc = bool(
        os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    )
    cpu_count = os.cpu_count() or 4

    if not is_hpc:
        return min(4, cpu_count)

    return target_workers


def collect_run_tags(args: argparse.Namespace, rank: int, size: int, workers: int) -> dict:
    """Collect scheduler metadata, job_id, and system properties for tracking."""
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
        "pipeline": "solar_opt",
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


def main(argv: Optional[List[str]] = None) -> None:
    """Main execution entry point."""
    args = parse_arguments(argv)
    config = get_pipeline_config(args.config)

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

    workers = resolve_worker_count(args.workers)

    comm = get_communicator()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Synchronize timestamp across MPI ranks from Rank 0
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if rank == 0 else None
    ts = comm.bcast(ts, root=0)

    logger = setup_logger(rank=rank, timestamp=ts)
    run_tags = collect_run_tags(args, rank=rank, size=size, workers=workers)
    job_id = run_tags.get("job_id")

    logger.info("=" * 60)
    logger.info("Concentrated Solar Power (CSP) Genetic Algorithm Optimization")
    role_str = "Master" if rank == 0 else f"Worker {rank}"
    logger.info(
        "Season: %s | Pop: %d | Gens: %d | Workers: %d | Rank: %d/%d (%s)",
        args.season,
        config.ga.pop_size,
        config.ga.num_generations,
        workers,
        rank,
        size,
        role_str,
    )

    is_hpc = bool(
        os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    )
    target_env = int(os.environ.get("LOCAL_WORKERS", "112"))
    if args.workers is None and not is_hpc and target_env > workers:
        logger.info(
            "Local workstation detected (no HPC scheduler). Safe fallback: using %d workers.",
            workers,
        )

    logger.info(
        "Scheduler Job ID: %s | Queue: %s | Host: %s",
        job_id,
        run_tags.get("queue"),
        run_tags.get("hostname"),
    )
    logger.info("=" * 60)

    if rank == 0:
        logger.info("[Rank 0: Master] Initializing Master GA optimization...")
        if not args.no_tracking:
            init_tracking(experiment_name=args.experiment)

        try:
            run_name = f"{run_tags.get('job_name', f'GA_Opt_{args.season}')}"
            if job_id and not job_id.startswith("local_"):
                run_name = f"{run_name}_{job_id}"

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(run_tags)
                log_run_params(config=config, season=args.season, workers=workers, job_id=job_id)

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
                            custom_logger=logger,
                        )
                finally:
                    for w in range(1, size):
                        comm.send(None, dest=w, tag=0)

        except Exception as exc:
            logger.warning("Running without MLflow tracking due to: %s", exc)
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
                        custom_logger=logger,
                    )
            finally:
                for w in range(1, size):
                    comm.send(None, dest=w, tag=0)

    else:
        logger.info(
            "[Rank %d: Worker] Entering worker event loop (listening to Master Rank 0)...", rank
        )
        worker_event_loop(
            season=args.season,
            comm=comm,
            rank=rank,
            config=config,
            workers=workers,
            custom_logger=logger,
        )

    logger.info("Solar Optimization Run Completed successfully!")


if __name__ == "__main__":
    main()
