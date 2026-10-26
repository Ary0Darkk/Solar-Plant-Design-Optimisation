#!/usr/bin/env python3
"""
Primary entry point for Concentrated Solar Power (CSP) GA Optimization.
Supports multi-node HPC execution (Hybrid MPI + Multiprocessing) and local workstations.
"""

import argparse
import multiprocessing
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from solar_opt.chromosome import init_deap_types  # noqa: E402
from solar_opt.config import get_pipeline_config  # noqa: E402
from solar_opt.engine import run_optimization, worker_event_loop  # noqa: E402
from solar_opt.parallel import get_communicator  # noqa: E402
from solar_opt.tracker import init_tracking, log_run_params, setup_logger  # noqa: E402


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(
        description="Concentrated Solar Power (CSP) Genetic Algorithm Optimization using PySAM and DEAP."
    )
    parser.add_argument(
        "--season",
        type=str,
        default="summer",
        choices=["winter", "summer", "monsoon", "post", "all"],
        help="Season to optimize: winter, summer, monsoon, post, or all (default: summer)",
    )
    return parser.parse_args(argv)


def resolve_worker_count(config_workers: int = 112) -> int:
    """
    Detect available CPU cores and allocate appropriate worker concurrency per node.
    Prioritizes LOCAL_WORKERS and PBS_NCPUS on PBS Pro HPC clusters.
    """
    local_workers = os.environ.get("LOCAL_WORKERS")
    if local_workers:
        try:
            return max(1, int(local_workers))
        except ValueError:
            pass

    pbs_cpus = os.environ.get("PBS_NCPUS")
    if pbs_cpus:
        try:
            return max(1, int(pbs_cpus))
        except ValueError:
            pass

    cpu_count = multiprocessing.cpu_count()
    is_hpc = bool(
        os.environ.get("PBS_JOBID")
        or os.environ.get("MPI_LOCALRANKID")
        or os.environ.get("OMPI_COMM_WORLD_SIZE")
    )
    if not is_hpc:
        return min(4, cpu_count)
    return min(config_workers, cpu_count)


def main(argv: Optional[List[str]] = None) -> None:
    """Main execution entry point with multi-node HPC MPI coordination."""
    args = parse_arguments(argv)
    config = get_pipeline_config()

    init_deap_types()
    workers = resolve_worker_count(config.simulation.max_simulation_workers)
    comm = get_communicator()
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = setup_logger(rank=rank, timestamp=timestamp)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Concentrated Solar Power (CSP) GA Optimization")
        if size > 1:
            logger.info(
                "Multi-Node HPC Topology: Master Rank 0 + %d Worker Ranks (Total Nodes: %d)",
                size - 1,
                size,
            )
        else:
            logger.info("Single-Node Topology (Standalone execution)")
        logger.info("Master Host: %s | Workers per node: %d", hostname, workers)
        if size == 1:
            logger.info(
                "Note: Running locally with pop_size=%d across %d worker processes.",
                config.ga.pop_size,
                workers,
            )
        logger.info("=" * 60)
    else:
        logger.info(
            "[Rank %d/%d] Worker node initialized on host %s (%d local worker processes)",
            rank,
            size,
            hostname,
            workers,
        )

    seasons = ["winter", "summer", "monsoon", "post"] if args.season == "all" else [args.season]

    for season in seasons:
        if rank == 0:
            if config.tracking.enabled:
                init_tracking(experiment_name=config.tracking.experiment_name)

            run_name = f"CSP_GA_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with (
                mlflow.start_run(run_name=run_name) if config.tracking.enabled else open(os.devnull)
            ):
                log_run_params(
                    {
                        "season": season,
                        "pop_size": config.ga.pop_size,
                        "generations": config.ga.num_generations,
                        "workers_per_node": workers,
                        "mpi_nodes": size,
                    }
                )
                run_optimization(
                    season,
                    config=config,
                    workers=workers,
                    comm=comm,
                    custom_logger=logger,
                    timestamp=timestamp,
                )
        else:
            worker_event_loop(comm, season=season, config=config, workers=workers)

    if rank == 0:
        logger.info("All requested optimization seasons completed successfully!")


if __name__ == "__main__":
    main()
