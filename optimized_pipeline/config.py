"""
Memory-efficient configuration models and physical parameter boundaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------------------
# Representative Days (Month, Day) across 4 seasons (7 days each)
# -------------------------------------------------------------------------
SEASONS: Dict[str, List[Tuple[int, int]]] = {
    "winter": [(1, 8), (1, 13), (1, 16), (1, 28), (2, 21), (2, 22), (12, 12)],
    "summer": [(3, 9), (3, 16), (4, 12), (4, 17), (5, 7), (5, 16), (5, 21)],
    "monsoon": [(6, 13), (7, 6), (7, 15), (7, 22), (7, 23), (8, 4), (8, 11)],
    "post": [(9, 5), (9, 20), (10, 3), (10, 12), (11, 3), (11, 17), (11, 23)],
}


@dataclass(frozen=True)
class VariableBounds:
    """Defines variables, physical boundaries, and integer conversion functions."""

    names: Tuple[str, ...]
    lb: Tuple[int, ...]
    ub: Tuple[int, ...]
    types: Tuple[Callable[[Any], int], ...]

    def __post_init__(self):
        if not (len(self.names) == len(self.lb) == len(self.ub) == len(self.types)):
            raise ValueError(
                f"Bounds length mismatch: names={len(self.names)}, "
                f"lb={len(self.lb)}, ub={len(self.ub)}, types={len(self.types)}"
            )
        for name, low, high in zip(self.names, self.lb, self.ub):
            if low > high:
                raise ValueError(f"Lower bound ({low}) > upper bound ({high}) for '{name}'")


# Season-wide 5 design parameters
DESIGN_BOUNDS = VariableBounds(
    names=(
        "specified_total_aperture",  # reflective area [m²]
        "Row_Distance",  # pitch between rows [m] (guaranteed > W_aperture)
        "ColperSCA",  # modules per SCA
        "W_aperture",  # aperture width [m]
        "L_SCA",  # collector length [m]
    ),
    lb=(8000, 10, 2, 4, 50),
    ub=(12000, 25, 10, 8, 200),
    types=(int, int, int, int, int),
)

# Daily 3 operational parameters
OPERATIONAL_BOUNDS = VariableBounds(
    names=(
        "m_dot",  # mass-flow limit [kg/s]
        "T_startup",  # startup temperature [°C]
        "T_shutdown",  # shutdown temperature [°C] (guaranteed <= T_startup)
    ),
    lb=(2, 300, 250),
    ub=(12, 380, 300),
    types=(int, int, int),
)


@dataclass
class GAParameters:
    """Evolutionary algorithm parameters and memory recycling settings."""

    pop_size: int = 112
    num_generations: int = 100
    tournament_size: int = 7
    hall_of_fame_size: int = 5
    cxpb: float = 0.8
    mutpb: float = 0.2
    indpb: float = 0.1
    random_seed: int = 41
    resume_from_checkpoint: bool = True
    checkpoint_interval: int = 1
    # Aggressive child process recycling to clear PySAM C-memory leaks
    max_tasks_per_child: int = 25
    # Explicit garbage collection after each evaluation wave
    enable_gc_collection: bool = True
    # Stop early if global best fitness fails to improve for N consecutive generations (0 = disabled)
    early_stopping_patience: int = 15


@dataclass
class TariffConfig:
    """Tariff normalization and demand data paths."""

    min_price: float = 3.0
    max_price: float = 8.0
    demand_file: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT / "electricity_data" / "Yearly Demand Profile Haryana-Himanchal.xlsx"
        )
    )
    cache_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "electricity_data" / "dynamic_price_data_haryana.csv"
    )


@dataclass
class SimulationConfig:
    """PySAM simulation settings and physical limits."""

    json_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "simulation" / "Gurgaon_plant.json"
    )
    model_name: str = "PhysicalTroughNone"
    penalty: float = -1e13
    log_each_simulation: bool = False


@dataclass
class PipelineConfig:
    """Top-level configuration uniting all optimization sub-systems."""

    seasons: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: dict(SEASONS))
    design: VariableBounds = DESIGN_BOUNDS
    operational: VariableBounds = OPERATIONAL_BOUNDS
    ga: GAParameters = field(default_factory=GAParameters)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert typed configuration into legacy CONFIG dictionary format."""
        return {
            "demand_file_path": str(self.tariff.demand_file),
            "show_demand_plot": False,
            "show_price_plot": False,
            "min_price": self.tariff.min_price,
            "max_price": self.tariff.max_price,
            "json_file": str(self.simulation.json_file),
            "model": self.simulation.model_name,
            "log_each_simulation": self.simulation.log_each_simulation,
            "penalty": self.simulation.penalty,
            "SEASONS": self.seasons,
            "design": {
                "names": list(self.design.names),
                "lb": list(self.design.lb),
                "ub": list(self.design.ub),
                "types": list(self.design.types),
            },
            "operational": {
                "names": list(self.operational.names),
                "lb": list(self.operational.lb),
                "ub": list(self.operational.ub),
                "types": list(self.operational.types),
            },
            "pop_size": self.ga.pop_size,
            "num_generations": self.ga.num_generations,
            "tournament_size": self.ga.tournament_size,
            "hall_of_fame_size": self.ga.hall_of_fame_size,
            "cxpb": self.ga.cxpb,
            "mutpb": self.ga.mutpb,
            "indpb": self.ga.indpb,
            "random_seed": self.ga.random_seed,
            "resume_from_checkpoint": self.ga.resume_from_checkpoint,
            "checkpoint_interval": self.ga.checkpoint_interval,
            "max_tasks_per_child": self.ga.max_tasks_per_child,
            "enable_gc_collection": self.ga.enable_gc_collection,
        }

    to_legacy_dict = to_dict


def get_pipeline_config() -> PipelineConfig:
    """Return a fresh default configuration instance."""
    return PipelineConfig()
