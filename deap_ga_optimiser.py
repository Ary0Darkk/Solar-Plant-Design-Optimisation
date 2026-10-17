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
from optimized_pipeline.engine import run_optimization, worker_event_loop
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

    Rank 0 acts as the Master Rank, coordinating the DEAP GA loop and dispatching
    evaluation batches. Ranks > 0 act as Worker Ranks running the worker event loop.

    Backward-compatible wrapper around `optimized_pipeline.engine`.
    """
    cfg = config or get_pipeline_config()

    if pop_size is not None:
        cfg.ga.pop_size = pop_size
    if num_generations is not None:
        cfg.ga.num_generations = num_generations

    import multiprocessing as mp

    from optimized_pipeline.cli import resolve_worker_count

    active_comm: Communicator = comm if comm is not None else get_communicator()
    effective_rank = rank if rank != 0 else active_comm.Get_rank()
    effective_size = mpi_size if mpi_size > 1 else active_comm.Get_size()
    workers = resolve_worker_count(kwargs.get("workers"))

    active_logger = logger or setup_logger(rank=effective_rank)

    if effective_rank == 0:
        # Master Rank
        active_logger.info(
            "[Rank 0: Master] Running GA optimization as Master Rank (mpi_size=%d)...",
            effective_size,
        )
        try:
            if local_pool is not None:
                return run_optimization(
                    season=season,
                    config=cfg,
                    pool=local_pool,
                    comm=active_comm,
                    rank=0,
                    mpi_size=effective_size,
                    custom_logger=active_logger,
                )
            else:
                ctx = mp.get_context()
                with ctx.Pool(
                    processes=workers, maxtasksperchild=cfg.ga.max_tasks_per_child
                ) as pool:
                    return run_optimization(
                        season=season,
                        config=cfg,
                        pool=pool,
                        comm=active_comm,
                        rank=0,
                        mpi_size=effective_size,
                        custom_logger=active_logger,
                    )
        finally:
            if effective_size > 1:
                for w in range(1, effective_size):
                    active_comm.send(None, dest=w, tag=0)
    else:
        # Worker Rank
        active_logger.info(
            "[Rank %d: Worker] Running worker event loop (listening to Master Rank 0)...",
            effective_rank,
        )
        worker_event_loop(
            season=season,
            comm=active_comm,
            rank=effective_rank,
            config=cfg,
            workers=workers,
            pool=local_pool,
        )
        return {}, 0.0, {}


__all__ = [
    "run_deap_ga_optimisation",
    "get_season_days",
    "CONFIG",
    "main",
    "parse_arguments",
]

if __name__ == "__main__":
    main()
