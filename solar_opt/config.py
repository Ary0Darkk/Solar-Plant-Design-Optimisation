"""
Global configuration dataclasses, bounds, and YAML loaders for the Solar Optimization pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------
# Default Bounds & Parameters
# ------------------------------------------------------------------------------

DEFAULT_DESIGN_BOUNDS: Dict[str, Tuple[int, int]] = {
    "specified_total_aperture": (8000, 12000),
    "Row_Distance": (10, 25),
    "ColperSCA": (2, 10),
    "W_aperture": (4, 8),
    "L_SCA": (50, 200),
}

DEFAULT_OPERATIONAL_BOUNDS: Dict[str, Tuple[int, int]] = {
    "m_dot": (2, 12),
    "T_startup": (300, 380),
    "T_shutdown": (250, 300),
}

DEFAULT_SEASONS: Dict[str, List[Tuple[int, int]]] = {
    "winter": [(1, 8), (1, 13), (1, 16), (1, 28), (2, 21), (2, 22), (12, 12)],
    "summer": [(3, 9), (3, 16), (4, 12), (4, 17), (5, 7), (5, 16), (5, 21)],
    "monsoon": [(6, 13), (7, 6), (7, 15), (7, 22), (7, 23), (8, 4), (8, 11)],
    "post": [(9, 5), (9, 20), (10, 3), (10, 12), (11, 3), (11, 17), (11, 23)],
}


@dataclass(frozen=True)
class VariableBounds:
    """Immutable collection of decision variable bounds and types."""

    names: Tuple[str, ...]
    lb: Tuple[int, ...]
    ub: Tuple[int, ...]
    types: Tuple[Type[int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.types:
            object.__setattr__(self, "types", tuple(int for _ in self.names))
        if not (len(self.names) == len(self.lb) == len(self.ub) == len(self.types)):
            raise ValueError(
                "All bound sequences (names, lb, ub, types) must have identical length."
            )


def _create_bounds(bounds_dict: Dict[str, Tuple[int, int]]) -> VariableBounds:
    names = tuple(bounds_dict.keys())
    lb = tuple(int(bounds_dict[k][0]) for k in names)
    ub = tuple(int(bounds_dict[k][1]) for k in names)
    return VariableBounds(names=names, lb=lb, ub=ub)


DESIGN_BOUNDS = _create_bounds(DEFAULT_DESIGN_BOUNDS)
OPERATIONAL_BOUNDS = _create_bounds(DEFAULT_OPERATIONAL_BOUNDS)
SEASONS = DEFAULT_SEASONS


@dataclass
class GAParameters:
    """Genetic Algorithm hyperparameters and execution settings."""

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
    max_tasks_per_child: int = 25
    enable_gc_collection: bool = True
    early_stopping_patience: int = 15


@dataclass
class TariffConfig:
    """Dynamic electricity tariff generation settings."""

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
    """PySAM concentrated solar power simulation parameters."""

    json_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "simulation" / "Gurgaon_plant.json"
    )
    model_name: str = "PhysicalTroughNone"
    penalty: float = -1.0e13
    log_each_simulation: bool = False


@dataclass
class PipelineConfig:
    """Consolidated configuration root for the entire optimization pipeline."""

    seasons: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: dict(SEASONS))
    design: VariableBounds = field(default_factory=lambda: DESIGN_BOUNDS)
    operational: VariableBounds = field(default_factory=lambda: OPERATIONAL_BOUNDS)
    ga: GAParameters = field(default_factory=GAParameters)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a flat dictionary for legacy compatibility."""
        return {
            "min_price": self.tariff.min_price,
            "max_price": self.tariff.max_price,
            "demand_file": str(self.tariff.demand_file),
            "cache_file": str(self.tariff.cache_file),
            "json_file": str(self.simulation.json_file),
            "model_name": self.simulation.model_name,
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
            "early_stopping_patience": self.ga.early_stopping_patience,
        }

    to_legacy_dict = to_dict


def load_pipeline_config(config_path: Optional[Union[Path, str]] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file or defaults.

    Args:
        config_path: Optional path to YAML file. Defaults to `config.yaml` in project root.
    """
    target_path = Path(config_path) if config_path else (PROJECT_ROOT / "config.yaml")
    if not target_path.exists():
        if config_path:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return PipelineConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 1. Seasons
    seasons = dict(SEASONS)
    if "seasons" in data and isinstance(data["seasons"], dict):
        seasons = {
            s: [tuple(item) for item in days]  # type: ignore
            for s, days in data["seasons"].items()
        }

    # 2. Design Bounds
    design_bounds = DESIGN_BOUNDS
    if "design_bounds" in data and isinstance(data["design_bounds"], dict):
        d_names = tuple(data["design_bounds"].keys())
        d_lb = tuple(int(data["design_bounds"][k][0]) for k in d_names)
        d_ub = tuple(int(data["design_bounds"][k][1]) for k in d_names)
        design_bounds = VariableBounds(
            names=d_names,
            lb=d_lb,
            ub=d_ub,
            types=tuple(int for _ in d_names),
        )

    # 3. Operational Bounds
    op_bounds = OPERATIONAL_BOUNDS
    if "operational_bounds" in data and isinstance(data["operational_bounds"], dict):
        o_names = tuple(data["operational_bounds"].keys())
        o_lb = tuple(int(data["operational_bounds"][k][0]) for k in o_names)
        o_ub = tuple(int(data["operational_bounds"][k][1]) for k in o_names)
        op_bounds = VariableBounds(
            names=o_names,
            lb=o_lb,
            ub=o_ub,
            types=tuple(int for _ in o_names),
        )

    # 4. GA Parameters
    ga_params = GAParameters()
    if "ga" in data and isinstance(data["ga"], dict):
        ga_dict = data["ga"]
        for key, val in ga_dict.items():
            if hasattr(ga_params, key):
                field_type = type(getattr(ga_params, key))
                setattr(ga_params, key, field_type(val))

    # 5. Tariff Config
    tariff_cfg = TariffConfig()
    if "tariff" in data and isinstance(data["tariff"], dict):
        t_dict = data["tariff"]
        if "min_price" in t_dict:
            tariff_cfg.min_price = float(t_dict["min_price"])
        if "max_price" in t_dict:
            tariff_cfg.max_price = float(t_dict["max_price"])
        if "demand_file" in t_dict:
            df_path = Path(t_dict["demand_file"])
            tariff_cfg.demand_file = df_path if df_path.is_absolute() else (PROJECT_ROOT / df_path)
        if "cache_file" in t_dict:
            cf_path = Path(t_dict["cache_file"])
            tariff_cfg.cache_file = cf_path if cf_path.is_absolute() else (PROJECT_ROOT / cf_path)

    # 6. Simulation Config
    sim_cfg = SimulationConfig()
    if "simulation" in data and isinstance(data["simulation"], dict):
        s_dict = data["simulation"]
        if "json_file" in s_dict:
            jf_path = Path(s_dict["json_file"])
            sim_cfg.json_file = jf_path if jf_path.is_absolute() else (PROJECT_ROOT / jf_path)
        if "model_name" in s_dict:
            sim_cfg.model_name = str(s_dict["model_name"])
        if "penalty" in s_dict:
            sim_cfg.penalty = float(s_dict["penalty"])
        if "log_each_simulation" in s_dict:
            sim_cfg.log_each_simulation = bool(s_dict["log_each_simulation"])

    return PipelineConfig(
        seasons=seasons,
        design=design_bounds,
        operational=op_bounds,
        ga=ga_params,
        tariff=tariff_cfg,
        simulation=sim_cfg,
    )


def save_pipeline_config_to_yaml(config: PipelineConfig, file_path: Union[Path, str]) -> None:
    """Serialize PipelineConfig instance to a YAML file."""
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "ga": {
            "pop_size": config.ga.pop_size,
            "num_generations": config.ga.num_generations,
            "tournament_size": config.ga.tournament_size,
            "hall_of_fame_size": config.ga.hall_of_fame_size,
            "cxpb": config.ga.cxpb,
            "mutpb": config.ga.mutpb,
            "indpb": config.ga.indpb,
            "random_seed": config.ga.random_seed,
            "resume_from_checkpoint": config.ga.resume_from_checkpoint,
            "checkpoint_interval": config.ga.checkpoint_interval,
            "max_tasks_per_child": config.ga.max_tasks_per_child,
            "enable_gc_collection": config.ga.enable_gc_collection,
            "early_stopping_patience": config.ga.early_stopping_patience,
        },
        "tariff": {
            "min_price": config.tariff.min_price,
            "max_price": config.tariff.max_price,
            "demand_file": (
                str(config.tariff.demand_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if config.tariff.demand_file.is_relative_to(PROJECT_ROOT)
                else str(config.tariff.demand_file).replace("\\", "/")
            ),
            "cache_file": (
                str(config.tariff.cache_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if config.tariff.cache_file.is_relative_to(PROJECT_ROOT)
                else str(config.tariff.cache_file).replace("\\", "/")
            ),
        },
        "simulation": {
            "json_file": (
                str(config.simulation.json_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if config.simulation.json_file.is_relative_to(PROJECT_ROOT)
                else str(config.simulation.json_file).replace("\\", "/")
            ),
            "model_name": config.simulation.model_name,
            "penalty": config.simulation.penalty,
            "log_each_simulation": config.simulation.log_each_simulation,
        },
        "design_bounds": {
            name: [int(low), int(high)]
            for name, low, high in zip(config.design.names, config.design.lb, config.design.ub)
        },
        "operational_bounds": {
            name: [int(low), int(high)]
            for name, low, high in zip(
                config.operational.names, config.operational.lb, config.operational.ub
            )
        },
        "seasons": {s: [list(d) for d in days] for s, days in config.seasons.items()},
    }

    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)


def get_pipeline_config(config_path: Optional[Union[Path, str]] = None) -> PipelineConfig:
    """Return a PipelineConfig instance, loaded from YAML or built-in defaults."""
    return load_pipeline_config(config_path)
