#!/usr/bin/env python3
"""
DEAP Genetic Algorithm Optimiser for Concentrated Solar Power (CSP) Plant Design.
Powered by the memory-optimized pipeline (`optimized_pipeline`).

Usage:
    python deap_ga_optimiser.py
    python deap_ga_optimiser.py --season winter --pop-size 28 --generations 10
"""

import logging
from typing import Any, Dict, Optional, Tuple

from config import CONFIG
from optimized_pipeline.cli import main, parse_arguments
from optimized_pipeline.config import PipelineConfig, get_pipeline_config
from optimized_pipeline.engine import run_optimization
from optimized_pipeline.parallel import Communicator, get_communicator
from optimized_pipeline.tracker import setup_logger
from optimized_pipeline.weather import get_season_days


def run_deap_ga_optimisation(
    season: str = "winter",
    local_pool: Any = None,
    comm: Optional[Any] = None,
    rank: int = 0,
    mpi_size: int = 1,
    logger: Optional[logging.Logger] = None,
    config: Optional[PipelineConfig] = None,
    pop_size: Optional[int] = None,
    num_generations: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Run Genetic Algorithm optimization for the specified season.

    Backward-compatible wrapper around `optimized_pipeline.engine.run_optimization`.
    """
    cfg = config or get_pipeline_config()

    if pop_size is not None:
        cfg.ga.pop_size = pop_size
    if num_generations is not None:
        cfg.ga.num_generations = num_generations

    import multiprocessing as mp

    from optimized_pipeline.cli import resolve_worker_count

    active_comm: Optional[Communicator] = comm
    if active_comm is None:
        active_comm = get_communicator()

    active_logger = logger or setup_logger(rank=rank)

    if local_pool is not None:
        return run_optimization(
            season=season,
            config=cfg,
            pool=local_pool,
            comm=active_comm,
            rank=rank,
            mpi_size=mpi_size,
            custom_logger=active_logger,
        )
    else:
        workers = resolve_worker_count(kwargs.get("workers"))
        ctx = mp.get_context()
        with ctx.Pool(processes=workers, maxtasksperchild=cfg.ga.max_tasks_per_child) as pool:
            return run_optimization(
                season=season,
                config=cfg,
                pool=pool,
                comm=active_comm,
                rank=rank,
                mpi_size=mpi_size,
                custom_logger=active_logger,
            )


__all__ = [
    "run_deap_ga_optimisation",
    "get_season_days",
    "CONFIG",
    "main",
    "parse_arguments",
]

if __name__ == "__main__":
    main()
