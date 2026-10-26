"""
Dataclass configuration models and YAML configuration loader.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class VariableBounds:
    """Strongly typed container for optimization decision variable bounds."""

    names: Tuple[str, ...]
    lb: Tuple[float, ...]
    ub: Tuple[float, ...]
    types: Tuple[type, ...]

    def __post_init__(self):
        n = len(self.names)
        if len(self.lb) != n or len(self.ub) != n or len(self.types) != n:
            raise ValueError(
                f"Dimension mismatch in VariableBounds: names={n}, lb={len(self.lb)}, ub={len(self.ub)}, types={len(self.types)}"
            )

    def to_bounds_list(self) -> List[Tuple[float, float]]:
        return list(zip(self.lb, self.ub))


@dataclass
class SystemConfig:
    project_name: str = "Solar-Plant-Design-Optimisation"
    log_level: str = "INFO"
    seed: int = 41
    scratch_dir: str = "scratch"


@dataclass
class SimulationConfig:
    model_name: str = "PhysicalTroughNone"
    weather_file: str = "simulation/28.428_77.159_28.428_77.159_himawari_60_2020.csv"
    plant_config_file: str = "simulation/Gurgaon_plant.json"
    max_simulation_workers: int = 112
    worker_recycle_rate: int = 25
    penalty: float = -1e6


@dataclass
class TariffConfig:
    source_file: str = "electricity_data/dynamic_price_data_haryana.csv"
    demand_file: str = "electricity_data/Yearly Demand Profile Haryana-Himanchal.xlsx"
    min_price: float = 3.0
    max_price: float = 8.0


@dataclass
class GAConfig:
    num_generations: int = 100
    pop_size: int = 112
    cxpb: float = 0.8
    mutpb: float = 0.2
    eta_c: float = 20.0
    eta_m: float = 20.0
    tournament_size: int = 3
    elitism_count: int = 2
    max_tasks_per_child: int = 25
    random_seed: int = 41


@dataclass
class TrackingConfig:
    enabled: bool = True
    experiment_name: str = "CSP_Seasonal_GA"
    dagshub_repo: str = "aryanvj787/NYS-Design-Optimisation-using-PySAM"
    dagshub_user: str = "aryanvj787"
    save_artifacts: bool = True
    dpi: int = 300


@dataclass
class BoundsConfig:
    design: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "specified_total_aperture": (8000.0, 12000.0),
            "Row_Distance": (10.0, 25.0),
            "ColperSCA": (2, 10),
            "W_aperture": (4.0, 8.0),
            "L_SCA": (50.0, 200.0),
        }
    )
    operational: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "m_dot": (2.0, 12.0),
            "T_startup": (300.0, 380.0),
            "T_shutdown": (250.0, 300.0),
        }
    )


@dataclass
class PipelineConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    seasons: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "winter": {"name": "Winter", "start_month": 1, "start_day": 8, "num_days": 7},
            "summer": {"name": "Summer", "start_month": 5, "start_day": 9, "num_days": 7},
            "monsoon": {"name": "Monsoon", "start_month": 7, "start_day": 15, "num_days": 7},
            "post": {"name": "Post-Monsoon", "start_month": 10, "start_day": 15, "num_days": 7},
        }
    )

    @property
    def design(self) -> VariableBounds:
        names = tuple(self.bounds.design.keys())
        lb = tuple(float(b[0]) for b in self.bounds.design.values())
        ub = tuple(float(b[1]) for b in self.bounds.design.values())
        types = tuple(int if k == "ColperSCA" else float for k in names)
        return VariableBounds(names=names, lb=lb, ub=ub, types=types)

    @property
    def operational(self) -> VariableBounds:
        names = tuple(self.bounds.operational.keys())
        lb = tuple(float(b[0]) for b in self.bounds.operational.values())
        ub = tuple(float(b[1]) for b in self.bounds.operational.values())
        types = tuple(float for _ in names)
        return VariableBounds(names=names, lb=lb, ub=ub, types=types)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for backward compatibility."""
        return {
            "min_price": self.tariff.min_price,
            "max_price": self.tariff.max_price,
            "pop_size": self.ga.pop_size,
            "generations": self.ga.num_generations,
            "crossover_prob": self.ga.cxpb,
            "mutation_prob": self.ga.mutpb,
            "seed": self.system.seed,
            "weather_file": self.simulation.weather_file,
            "plant_config_file": self.simulation.plant_config_file,
            "design_bounds": self.design.to_bounds_list(),
            "operational_bounds": self.operational.to_bounds_list(),
        }


def save_pipeline_config_to_yaml(config: PipelineConfig, filepath: Union[str, Path]) -> Path:
    """Serialize PipelineConfig to standard YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "system": {
            "project_name": config.system.project_name,
            "log_level": config.system.log_level,
            "seed": config.system.seed,
            "scratch_dir": config.system.scratch_dir,
        },
        "simulation": {
            "model_name": config.simulation.model_name,
            "weather_file": config.simulation.weather_file,
            "plant_config_file": config.simulation.plant_config_file,
            "max_simulation_workers": config.simulation.max_simulation_workers,
            "worker_recycle_rate": config.simulation.worker_recycle_rate,
        },
        "tariff": {
            "source_file": config.tariff.source_file,
            "demand_file": config.tariff.demand_file,
            "min_price": config.tariff.min_price,
            "max_price": config.tariff.max_price,
        },
        "seasons": config.seasons,
        "ga": {
            "num_generations": config.ga.num_generations,
            "pop_size": config.ga.pop_size,
            "cxpb": config.ga.cxpb,
            "mutpb": config.ga.mutpb,
            "eta_c": config.ga.eta_c,
            "eta_m": config.ga.eta_m,
            "tournament_size": config.ga.tournament_size,
            "elitism_count": config.ga.elitism_count,
            "max_tasks_per_child": config.ga.max_tasks_per_child,
        },
        "tracking": {
            "enabled": config.tracking.enabled,
            "experiment_name": config.tracking.experiment_name,
            "dagshub_repo": config.tracking.dagshub_repo,
            "dagshub_user": config.tracking.dagshub_user,
            "save_artifacts": config.tracking.save_artifacts,
            "dpi": config.tracking.dpi,
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return filepath


def load_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML file or defaults."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        if config_path:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return PipelineConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = PipelineConfig()

    if "system" in data:
        cfg.system = SystemConfig(
            **{k: v for k, v in data["system"].items() if hasattr(cfg.system, k)}
        )

    if "simulation" in data:
        cfg.simulation = SimulationConfig(
            **{k: v for k, v in data["simulation"].items() if hasattr(cfg.simulation, k)}
        )

    if "tariff" in data:
        cfg.tariff = TariffConfig(
            **{k: v for k, v in data["tariff"].items() if hasattr(cfg.tariff, k)}
        )

    if "seasons" in data and isinstance(data["seasons"], dict):
        cfg.seasons = data["seasons"]

    if "ga" in data:
        ga_dict = data["ga"]
        mapping = {
            "crossover_prob": "cxpb",
            "mutation_prob": "mutpb",
        }
        clean_ga = {}
        for k, v in ga_dict.items():
            target_key = mapping.get(k, k)
            if hasattr(cfg.ga, target_key):
                clean_ga[target_key] = v
        cfg.ga = GAConfig(**clean_ga)

    if "tracking" in data:
        cfg.tracking = TrackingConfig(
            **{k: v for k, v in data["tracking"].items() if hasattr(cfg.tracking, k)}
        )

    if "bounds" in data:
        if "design" in data["bounds"]:
            cfg.bounds.design = {k: tuple(v) for k, v in data["bounds"]["design"].items()}
        if "operational" in data["bounds"]:
            cfg.bounds.operational = {k: tuple(v) for k, v in data["bounds"]["operational"].items()}

    # Environment variable overrides (useful for CI, testing, and micro-runs)
    if "GA_NUM_GENERATIONS" in os.environ:
        try:
            cfg.ga.num_generations = int(os.environ["GA_NUM_GENERATIONS"])
        except ValueError:
            pass
    if "GA_POP_SIZE" in os.environ:
        try:
            cfg.ga.pop_size = int(os.environ["GA_POP_SIZE"])
        except ValueError:
            pass

    return cfg


def get_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Convenience getter for PipelineConfig."""
    return load_pipeline_config(config_path)
