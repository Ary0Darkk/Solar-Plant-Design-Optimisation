"""
Solar Plant Design & Operation Optimization Package.
"""

from .config import (
    GAParameters,
    PipelineConfig,
    SimulationConfig,
    TariffConfig,
    get_pipeline_config,
)
from .engine import run_optimization
from .simulation import CompactSimulator, DailySimulationSlice, get_simulator

__version__ = "2.0.0"

__all__ = [
    "GAParameters",
    "PipelineConfig",
    "SimulationConfig",
    "TariffConfig",
    "get_pipeline_config",
    "run_optimization",
    "CompactSimulator",
    "DailySimulationSlice",
    "get_simulator",
]
