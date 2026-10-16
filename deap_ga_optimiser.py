"""
DEAP Genetic Algorithm Optimiser for Concentrated Solar Power (CSP).
Now powered by the memory-optimized pipeline (`optimized_pipeline`).

This module replaces the legacy monolithic optimizer with high-performance,
memory-efficient components (24-hour simulation slicing, process recycling,
pre-computed tariffs, gzip checkpointing, and MPI coordination).
"""

import logging
from typing import Any, Dict, Optional, Tuple

from config import CONFIG
from optimized_pipeline.cli import main
from optimized_pipeline.config import PipelineConfig, get_pipeline_config
from optimized_pipeline.engine import run_optimization
from optimized_pipeline.parallel import Communicator, StandaloneComm, get_communicator
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

    # Use existing communicator if provided, otherwise auto-detect
    active_comm: Optional[Communicator] = comm
    if active_comm is None:
        active_comm = get_communicator()

    return run_optimization(
        season=season,
        config=cfg,
        comm=active_comm,
        custom_logger=logger,
    )


__all__ = [
    "run_deap_ga_optimisation",
    "get_season_days",
    "CONFIG",
    "main",
]

if __name__ == "__main__":
    main()
