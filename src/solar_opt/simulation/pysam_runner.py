"""
PySAM Concentrated Solar Power (CSP) TroughPhysical simulation engine.
"""

import json
import logging
import time
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import PySAM.TroughPhysical as TP

from ..config.defaults import PROJECT_ROOT
from .models import SimulationOutputs

logger = logging.getLogger("NYS_Optimisation")


def replace_1st_order(data: Union[Sequence[Any], List[Any]], new_val: Any) -> Sequence[Any]:
    """Replace the first element of a list or tuple, preserving sequence type."""
    if isinstance(data, tuple):
        return (new_val,) + data[1:]
    return [new_val] + list(data[1:])


class PySAMSimulationRunner:
    """Manages PySAM model execution, parameter mapping, constraint checks, and results extraction."""

    def __init__(
        self,
        json_file: Union[str, Path] = "simulation/Gurgaon_plant.json",
        model_name: str = "PhysicalTroughNone",
        penalty: float = -1e13,
        log_each_simulation: bool = False,
    ):
        self.json_file = Path(json_file)
        if not self.json_file.is_absolute():
            self.json_file = PROJECT_ROOT / self.json_file

        self.model_name = model_name
        self.penalty = penalty
        self.log_each_simulation = log_each_simulation
        self.valid_defaults, self.param_kinds = self._load_and_probe_defaults()

    def _load_and_probe_defaults(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Load JSON defaults and probe valid keys and parameter types on startup."""
        if not self.json_file.exists():
            raise FileNotFoundError(f"PySAM configuration JSON file not found: {self.json_file}")

        with self.json_file.open("r", encoding="utf-8") as file:
            raw_defaults = json.load(file)

        probe_model = TP.default(self.model_name)
        valid_defaults: Dict[str, Any] = {}
        failed_keys: List[Tuple[str, str]] = []

        for key, value in raw_defaults.items():
            if key == "number_inputs":
                continue
            try:
                probe_model.value(key, value)
                valid_defaults[key] = value
            except Exception as exc:
                failed_keys.append((key, str(exc)))

        if failed_keys:
            logger.debug(
                "Ignored %d unsupported JSON defaults during probe: %s",
                len(failed_keys),
                [k for k, _ in failed_keys],
            )

        known_override_keys = {
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
        param_kinds: Dict[str, str] = {}
        for key in known_override_keys:
            try:
                val = probe_model.value(key)
                if isinstance(val, (Real, bool)):
                    param_kinds[key] = "scalar"
                elif isinstance(val, (list, tuple)):
                    param_kinds[key] = "sequence"
                else:
                    param_kinds[key] = "other"
            except Exception:
                param_kinds[key] = "unknown"

        return valid_defaults, param_kinds

    def validate_physical_constraints(self, overrides: Dict[str, Any]) -> bool:
        """Validate geometric feasibility (Row_Distance > W_aperture) before execution."""
        row_dist = overrides.get("Row_Distance")
        w_aperture = overrides.get("W_aperture")

        if row_dist is not None and w_aperture is not None and row_dist <= w_aperture:
            logger.warning(
                "Physical violation: Row_Distance (%.2f m) <= W_aperture (%.2f m). Collectors would collide.",
                float(row_dist),
                float(w_aperture),
            )
            return False
        return True

    def normalize_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize overrides, expanding 'm_dot' into min and max mass-flow parameters."""
        normalized = dict(overrides)
        if "m_dot" in normalized:
            m_dot = normalized.pop("m_dot")
            normalized.setdefault("m_dot_htfmin", m_dot)
            normalized.setdefault("m_dot_htfmax", m_dot)
        return normalized

    def initialize_model(self) -> Any:
        """Instantiate and populate a fresh PySAM model with validated defaults."""
        model = TP.default(self.model_name)
        for key, value in self.valid_defaults.items():
            model.value(key, value)
        return model

    def apply_overrides(self, model: Any, overrides: Dict[str, Any]) -> None:
        """Apply parameter overrides to the PySAM model instance."""
        for key, value in overrides.items():
            kind = self.param_kinds.get(key)
            if kind == "scalar":
                model.value(key, value)
            elif kind == "sequence":
                curr = model.value(key)
                new_val = (
                    replace_1st_order(curr, value)
                    if len(curr) > 0
                    else ((value,) if isinstance(curr, tuple) else [value])
                )
                model.value(key, new_val)
            else:
                try:
                    curr = model.value(key)
                except Exception as exc:
                    raise ValueError(f"Unknown PySAM parameter: {key}") from exc

                if isinstance(curr, (Real, bool)):
                    model.value(key, value)
                elif isinstance(curr, (list, tuple)):
                    new_val = (
                        replace_1st_order(curr, value)
                        if len(curr) > 0
                        else ((value,) if isinstance(curr, tuple) else [value])
                    )
                    model.value(key, new_val)
                else:
                    raise TypeError(f"Unsupported parameter type for '{key}': {type(curr).__name__}")

    def run_simulation(
        self,
        overrides: Dict[str, Any],
        day_index: Optional[int] = None,
    ) -> Tuple[Optional[SimulationOutputs], bool]:
        """
        Execute one PySAM simulation with constraint checks, timing, and penalty fallback.

        Returns:
            (SimulationOutputs or None, is_penalized: bool)
        """
        display_params = overrides or {"Mode": "Baseline"}

        if not self.validate_physical_constraints(overrides):
            logger.warning("Simulation skipped due to physical constraint violation.")
            return None, True

        start_time = time.perf_counter()
        try:
            normalized = self.normalize_overrides(overrides)
            model = self.initialize_model()
            self.apply_overrides(model, normalized)

            if day_index is not None:
                model.SystemControl.time_start = float(day_index * 86400)
                model.SystemControl.time_stop = float((day_index + 1) * 86400)

            model.execute()
            outputs = model.Outputs

            if day_index is not None:
                def _align(series):
                    arr = [0.0] * 8760
                    start = day_index * 24
                    arr[start : start + 24] = series[:24]
                    return arr
            else:
                def _align(series):
                    return series

            result: SimulationOutputs = {
                "hourly_energy": _align(outputs.P_cycle),
                "pc_htf_pump_power": _align(outputs.cycle_htf_pump_power),
                "field_htf_pump_power": _align(outputs.W_dot_field_pump),
                "field_collector_tracking_power": _align(outputs.W_dot_sca_track),
                "pc_startup_thermal_power": _align(outputs.q_dot_pc_startup),
                "field_piping_thermal_loss": _align(outputs.q_dot_piping_loss),
                "receiver_thermal_loss": _align(outputs.q_dot_rec_thermal_loss),
                "annual_energy": outputs.annual_energy,
                "gross_annual_energy": outputs.annual_W_cycle_gross,
                "land_area": outputs.total_land_area,
                "land_cost": outputs.csp_dtr_cost_plm_total,
                "total_installed_cost": outputs.total_installed_cost,
            }

            duration = time.perf_counter() - start_time
            if self.log_each_simulation:
                logger.info(
                    "Simulation completed in %.2f s | Parameters: %s",
                    duration,
                    display_params,
                )

            return result, False

        except Exception:
            duration = time.perf_counter() - start_time
            logger.exception(
                "Simulation failed after %.2f s | Parameters: %s",
                duration,
                display_params,
            )
            return None, True


# Global default runner instance for easy importing
_default_runner: Optional[PySAMSimulationRunner] = None


def get_default_runner() -> PySAMSimulationRunner:
    """Return a singleton instance of PySAMSimulationRunner."""
    global _default_runner
    if _default_runner is None:
        _default_runner = PySAMSimulationRunner()
    return _default_runner


def run_simulation(
    overrides: Dict[str, Any],
    day_index: Optional[int] = None,
) -> Tuple[Optional[SimulationOutputs], bool]:
    """Module-level simulation function delegating to the default runner."""
    return get_default_runner().run_simulation(overrides, day_index=day_index)
