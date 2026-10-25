"""
High-Performance 24-Hour PySAM Physical Trough Simulation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DailySimulationSlice:
    """Exact 24-hour simulation window outputs (~672 bytes)."""

    day_index: int
    hourly_energy: np.ndarray  # (24,) float32 [MWh]
    hourly_parasitic: np.ndarray  # (24,) float32 [MWh]
    hourly_heat_loss: np.ndarray  # (24,) float32 [MWt]


def validate_physical_constraints(params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate physical engineering feasibility guardrails:
    1. Maintenance corridor: Row_Distance - W_aperture >= 3.5 m
    2. Anti-chattering deadband: T_startup - T_shutdown >= 15 °C
    """
    row_distance = params.get("Row_Distance", 15.0)
    w_aperture = params.get("W_aperture", 5.0)
    if (row_distance - w_aperture) < 3.5:
        return (
            False,
            f"Corridor clearance violated: Row_Distance ({row_distance}) - W_aperture ({w_aperture}) < 3.5 m",
        )

    t_startup = params.get("T_startup")
    t_shutdown = params.get("T_shutdown")
    if t_startup is not None and t_shutdown is not None:
        if (t_startup - t_shutdown) < 15.0:
            return (
                False,
                f"Anti-chattering deadband violated: T_startup ({t_startup}) - T_shutdown ({t_shutdown}) < 15 °C",
            )

    return True, "Valid"


class CompactCSPPlantSimulator:
    """Production PySAM simulator isolating 24-hour daily output windows."""

    def __init__(self, config: Any):
        self.config = config
        self._model = None
        self._init_pysam_model()

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def _init_pysam_model(self):
        """Initialize PySAM TroughPhysical instance."""
        import PySAM.TroughPhysical as TP

        self._model = TP.default(getattr(self.config, "model_name", "PhysicalTroughNone"))

        plant_cfg_file = self._resolve_path(self.config.plant_config_file)
        if plant_cfg_file.exists():
            with open(plant_cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            for k, v in cfg_data.items():
                try:
                    self._model.value(k, v)
                except Exception:
                    pass

        weather_file = self._resolve_path(self.config.weather_file)
        if weather_file.exists():
            self._model.Weather.file_name = str(weather_file.resolve())

    def run_daily_simulation(
        self,
        design_overrides: Dict[str, Any],
        day_index: int,
    ) -> Tuple[Optional[DailySimulationSlice], bool]:
        """
        Execute PySAM simulation with parameter overrides and extract 24-hour window.
        Returns: (DailySimulationSlice, is_penalized)
        """
        valid, _ = validate_physical_constraints(design_overrides)
        if not valid:
            return None, True

        try:
            for k, v in design_overrides.items():
                try:
                    self._model.value(k, v)
                except Exception:
                    pass

            self._model.execute()

            start_hr = day_index * 24
            end_hr = start_hr + 24

            # Electrical generation (kW -> MWh)
            gen = np.array(self._model.Outputs.gen[start_hr:end_hr], dtype=np.float32) / 1000.0
            gen = np.clip(gen, 0.0, None)

            # Parasitic pumping loads (MWe)
            try:
                par_fp = (
                    np.array(self._model.Outputs.W_dot_pump[start_hr:end_hr], dtype=np.float32)
                    / 1000.0
                )
            except Exception:
                par_fp = np.zeros(24, dtype=np.float32)

            try:
                par_track = (
                    np.array(self._model.Outputs.W_dot_sca_track[start_hr:end_hr], dtype=np.float32)
                    / 1000.0
                )
            except Exception:
                par_track = np.zeros(24, dtype=np.float32)

            hourly_parasitic = np.clip(par_fp + par_track, 0.0, None)

            # Thermal losses (MWt)
            try:
                piping_loss = (
                    np.array(
                        self._model.Outputs.q_dot_piping_loss[start_hr:end_hr], dtype=np.float32
                    )
                    / 1000.0
                )
            except Exception:
                piping_loss = np.zeros(24, dtype=np.float32)

            try:
                rec_loss = (
                    np.array(self._model.Outputs.q_dot_rec_loss[start_hr:end_hr], dtype=np.float32)
                    / 1000.0
                )
            except Exception:
                rec_loss = np.zeros(24, dtype=np.float32)

            hourly_heat_loss = np.clip(piping_loss + rec_loss, 0.0, None)

            slice_out = DailySimulationSlice(
                day_index=day_index,
                hourly_energy=gen,
                hourly_parasitic=hourly_parasitic,
                hourly_heat_loss=hourly_heat_loss,
            )
            return slice_out, False

        except Exception:
            return None, True


def get_simulator(config: Any) -> CompactCSPPlantSimulator:
    """Simulator factory."""
    return CompactCSPPlantSimulator(config)
