"""
Typed configuration data classes for the CSP optimization pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


@dataclass(frozen=True)
class VariableBounds:
    """Defines variables, bounds, and scalar types for an optimization group."""

    names: Tuple[str, ...]
    lb: Tuple[int, ...]
    ub: Tuple[int, ...]
    types: Tuple[Callable[[Any], Any], ...]

    def __post_init__(self):
        if not (len(self.names) == len(self.lb) == len(self.ub) == len(self.types)):
            raise ValueError(
                f"Mismatch in variable bounds definitions: "
                f"names={len(self.names)}, lb={len(self.lb)}, ub={len(self.ub)}, types={len(self.types)}"
            )
        for name, low, high in zip(self.names, self.lb, self.ub):
            if low > high:
                raise ValueError(
                    f"Lower bound ({low}) exceeds upper bound ({high}) for variable '{name}'"
                )


@dataclass
class GAParameters:
    """Genetic Algorithm evolutionary hyperparameters."""

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
    verbose: bool = True


@dataclass
class TariffSettings:
    """Tariff scaling and electricity demand settings."""

    min_price: float = 3.0
    max_price: float = 8.0
    demand_file_path: Path = field(
        default_factory=lambda: Path("electricity_data/Yearly Demand Profile Haryana-Himanchal.xlsx")
    )
    show_demand_plot: bool = False
    show_price_plot: bool = False


@dataclass
class SimulationSettings:
    """PySAM simulation settings."""

    json_file: Path = field(default_factory=lambda: Path("simulation/Gurgaon_plant.json"))
    model: str = "PhysicalTroughNone"
    penalty: float = -1e13
    log_each_simulation: bool = False


@dataclass
class OptimizationConfig:
    """Master configuration encompassing all pipeline parameters."""

    seasons: Dict[str, List[Tuple[int, int]]]
    design: VariableBounds
    operational: VariableBounds
    ga: GAParameters = field(default_factory=GAParameters)
    tariff: TariffSettings = field(default_factory=TariffSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert typed config to backward-compatible dictionary for existing callers."""
        return {
            "demand_file_path": str(self.tariff.demand_file_path),
            "show_demand_plot": self.tariff.show_demand_plot,
            "show_price_plot": self.tariff.show_price_plot,
            "min_price": self.tariff.min_price,
            "max_price": self.tariff.max_price,
            "json_file": str(self.simulation.json_file),
            "model": self.simulation.model,
            "log_each_simulation": self.simulation.log_each_simulation,
            "penalty": self.simulation.penalty,
            "SEASONS": self.seasons,
            "design": {
                "overrides": list(self.design.names),
                "lb": list(self.design.lb),
                "ub": list(self.design.ub),
                "types": list(self.design.types),
            },
            "operational": {
                "overrides": list(self.operational.names),
                "lb": list(self.operational.lb),
                "ub": list(self.operational.ub),
                "types": list(self.operational.types),
            },
            "resume_from_checkpoint": self.ga.resume_from_checkpoint,
            "checkpoint_interval": self.ga.checkpoint_interval,
            "random_seed": self.ga.random_seed,
            "pop_size": self.ga.pop_size,
            "num_generations": self.ga.num_generations,
            "tournament_size": self.ga.tournament_size,
            "hall_of_fame_size": self.ga.hall_of_fame_size,
            "cxpb": self.ga.cxpb,
            "mutpb": self.ga.mutpb,
            "indpb": self.ga.indpb,
            "verbose": self.ga.verbose,
        }
