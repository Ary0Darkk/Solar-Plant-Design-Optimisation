"""
High-efficiency, 24-hour slice PySAM Concentrated Solar Power simulator.
"""

import json
import logging
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PySAM.TroughPhysical as TP

from .config import PROJECT_ROOT, SimulationConfig

logger = logging.getLogger("NYS_Optimisation")


@dataclass(slots=True)
class DailySimulationSlice:
    """
    Ultra-compact 24-hour simulation output payload.
    Total memory footprint: ~672 bytes (vs ~490 KB in legacy 8,760-hour list representation).
    """

    hourly_energy: np.ndarray           # Shape: (24,), dtype: float32
    field_htf_pump_power: np.ndarray    # Shape: (24,), dtype: float32
    pc_htf_pump_power: np.ndarray       # Shape: (24,), dtype: float32
    collector_tracking_power: np.ndarray  # Shape: (24,), dtype: float32
    pc_startup_thermal: np.ndarray      # Shape: (24,), dtype: float32
    piping_loss: np.ndarray             # Shape: (24,), dtype: float32
    receiver_loss: np.ndarray           # Shape: (24,), dtype: float32
    annual_energy: float
    land_area: float
    total_installed_cost: float


def validate_constraints(overrides: Dict[str, Any]) -> bool:
    """
    Verify physical feasibility:
    1. Geometric clearance: Row_Distance - W_aperture >= 3.5 m (maintenance truck corridor)
    2. Operational temperature deadband: T_startup - T_shutdown >= 15°C (anti-chattering hysteresis)
    """
    row_dist = overrides.get("Row_Distance")
    w_aperture = overrides.get("W_aperture")
    if row_dist is not None and w_aperture is not None:
        if (row_dist - w_aperture) < 3.5:
            return False

    t_start = overrides.get("T_startup")
    t_shut = overrides.get("T_shutdown")
    if t_start is not None and t_shut is not None:
        if (t_start - t_shut) < 15:
            return False

    return True



class CompactSimulator:

    """PySAM model runner optimizing process memory and extracting strictly needed hourly slices."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.json_file = Path(self.config.json_file)
        if not self.json_file.is_absolute():
            self.json_file = PROJECT_ROOT / self.json_file

        self.valid_defaults, self.param_kinds = self._probe_model()

    def _probe_model(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        if not self.json_file.exists():
            raise FileNotFoundError(f"Missing PySAM JSON plant config at: {self.json_file}")

        with open(self.json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        probe = TP.default(self.config.model_name)
        defaults: Dict[str, Any] = {}
        for k, v in raw.items():
            if k == "number_inputs":
                continue
            try:
                probe.value(k, v)
                defaults[k] = v
            except Exception:
                pass

        kinds: Dict[str, str] = {}
        target_keys = {
            "specified_total_aperture",
            "Row_Distance",
            "ColperSCA",
            "W_aperture",
            "L_SCA",
            "m_dot",
            "m_dot_htfmin",
            "m_dot_htfmax",
            "T_startup",
            "T_shutdown",
        }
        for k in target_keys:
            try:
                val = probe.value(k)
                if isinstance(val, (Real, bool)):
                    kinds[k] = "scalar"
                elif isinstance(val, (list, tuple)):
                    kinds[k] = "sequence"
                else:
                    kinds[k] = "other"
            except Exception:
                kinds[k] = "unknown"

        return defaults, kinds

    def validate_constraints(self, overrides: Dict[str, Any]) -> bool:
        """Verify geometric non-collision (Row_Distance > W_aperture) and thermal limits."""
        return validate_constraints(overrides)

    def _apply_overrides(self, model: Any, overrides: Dict[str, Any]) -> None:
        for k, v in overrides.items():
            kind = self.param_kinds.get(k)
            if kind == "scalar":
                model.value(k, v)
            elif kind == "sequence":
                curr = model.value(k)
                new_v = (v,) + curr[1:] if isinstance(curr, tuple) else [v] + list(curr[1:])
                model.value(k, new_v)
            else:
                try:
                    curr = model.value(k)
                    if isinstance(curr, (Real, bool)):
                        model.value(k, v)
                    elif isinstance(curr, (list, tuple)):
                        new_v = (v,) + curr[1:] if isinstance(curr, tuple) else [v] + list(curr[1:])
                        model.value(k, new_v)
                except Exception:
                    pass

    def run_daily_simulation(
        self,
        overrides: Dict[str, Any],
        day_index: int,
    ) -> Tuple[Optional[DailySimulationSlice], bool]:
        """
        Execute PySAM for a single 24-hour window and return exact 24-element float32 slices.

        Returns:
            (DailySimulationSlice or None, is_penalized: bool)
        """
        if not self.validate_constraints(overrides):
            return None, True

        params = dict(overrides)
        if "m_dot" in params:
            m = params.pop("m_dot")
            params.setdefault("m_dot_htfmin", m)
            params.setdefault("m_dot_htfmax", m)

        # Enforce shutdown threshold <= startup threshold
        if "T_shutdown" in params and "T_startup" in params:
            params["T_shutdown"] = min(params["T_shutdown"], params["T_startup"])

        model = None
        try:
            model = TP.default(self.config.model_name)
            for k, v in self.valid_defaults.items():
                model.value(k, v)

            self._apply_overrides(model, params)

            # Limit simulation duration strictly to the target 24-hour day
            model.SystemControl.time_start = float(day_index * 86400)
            model.SystemControl.time_stop = float((day_index + 1) * 86400)

            model.execute()
            out = model.Outputs

            # Extract strictly 24 hours into contiguous float32 numpy buffers
            slice_result = DailySimulationSlice(
                hourly_energy=np.asarray(out.P_cycle[:24], dtype=np.float32),
                field_htf_pump_power=np.asarray(out.W_dot_field_pump[:24], dtype=np.float32),
                pc_htf_pump_power=np.asarray(out.cycle_htf_pump_power[:24], dtype=np.float32),
                collector_tracking_power=np.asarray(out.W_dot_sca_track[:24], dtype=np.float32),
                pc_startup_thermal=np.asarray(out.q_dot_pc_startup[:24], dtype=np.float32),
                piping_loss=np.asarray(out.q_dot_piping_loss[:24], dtype=np.float32),
                receiver_loss=np.asarray(out.q_dot_rec_thermal_loss[:24], dtype=np.float32),
                annual_energy=float(out.annual_energy),
                land_area=float(out.total_land_area),
                total_installed_cost=float(out.total_installed_cost),
            )
            return slice_result, False

        except Exception as exc:
            logger.debug("Simulation error on day %d: %s", day_index, exc)
            return None, True

        finally:
            # Release PySAM C++ allocation immediately
            del model


# Process-local simulator singleton to avoid rebuilding default model repeatedly
_LOCAL_SIMULATOR: Optional[CompactSimulator] = None


def get_simulator() -> CompactSimulator:
    """Retrieve or create the process-local simulator."""
    global _LOCAL_SIMULATOR
    if _LOCAL_SIMULATOR is None:
        _LOCAL_SIMULATOR = CompactSimulator()
    return _LOCAL_SIMULATOR


# Alias for compatibility
PySAMSimulator = CompactSimulator

