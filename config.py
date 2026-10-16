"""
Global configuration file for the Concentrated Solar Power (CSP) optimization project.
Powered by the memory-optimized pipeline (`optimized_pipeline.config`).
"""

from optimized_pipeline.config import (
    DESIGN_BOUNDS,
    OPERATIONAL_BOUNDS,
    PROJECT_ROOT,
    SEASONS,
    GAParameters,
    PipelineConfig,
    SimulationConfig,
    TariffConfig,
    VariableBounds,
    get_pipeline_config,
)

# Canonical configuration instance
PIPELINE_CONFIG: PipelineConfig = get_pipeline_config()

# Backward-compatible dictionary view
CONFIG = PIPELINE_CONFIG.to_dict()

__all__ = [
    "CONFIG",
    "PIPELINE_CONFIG",
    "PipelineConfig",
    "get_pipeline_config",
    "DESIGN_BOUNDS",
    "OPERATIONAL_BOUNDS",
    "SEASONS",
    "VariableBounds",
    "GAParameters",
    "TariffConfig",
    "SimulationConfig",
    "PROJECT_ROOT",
]
