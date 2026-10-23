"""
Compatibility module re-exporting solar_opt.
"""

from solar_opt import (  # noqa: F401  # noqa: F401
    CompactSimulator,
    DailySimulationSlice,
    GAParameters,
    PipelineConfig,
    SimulationConfig,
    TariffConfig,
    checkpoint,
    chromosome,
    cli,
    config,
    engine,
    get_pipeline_config,
    get_simulator,
    objective,
    operators,
    parallel,
    run_optimization,
    simulation,
    tariff,
    tracker,
    weather,
)

__all__ = [
    "checkpoint",
    "chromosome",
    "cli",
    "config",
    "engine",
    "objective",
    "operators",
    "parallel",
    "simulation",
    "tariff",
    "tracker",
    "weather",
]
