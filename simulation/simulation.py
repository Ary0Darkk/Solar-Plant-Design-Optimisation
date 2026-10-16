# Solar plant simulation core module
import json
import logging
import sys
import time
from numbers import Real
from pathlib import Path
from typing import TypedDict

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import PySAM.TroughPhysical as TP

from config import CONFIG

# --------------------------------------------------
# Type definitions
# --------------------------------------------------


class SimulationOutputs(TypedDict):
    """Structured outputs returned from a successful PySAM simulation."""

    hourly_energy: tuple[float, ...] | list[float]
    pc_htf_pump_power: tuple[float, ...] | list[float]
    field_htf_pump_power: tuple[float, ...] | list[float]
    field_collector_tracking_power: tuple[float, ...] | list[float]
    pc_startup_thermal_power: tuple[float, ...] | list[float]
    field_piping_thermal_loss: tuple[float, ...] | list[float]
    receiver_thermal_loss: tuple[float, ...] | list[float]
    annual_energy: float
    gross_annual_energy: float
    land_area: float
    land_cost: float
    total_installed_cost: float


# --------------------------------------------------
# Sequence parameter helper
# --------------------------------------------------


def replace_1st_order(data, new_val):
    """Replace the first element of a list or tuple, preserving sequence type."""
    if isinstance(data, tuple):
        return (new_val,) + data[1:]
    return [new_val] + list(data[1:])


# --------------------------------------------------
# Logger setup
# --------------------------------------------------

LOGGER_NAME = "NYS_Optimisation"
logger = logging.getLogger(LOGGER_NAME)

if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())


# --------------------------------------------------
# Default parameters & model metadata probing
# --------------------------------------------------


def load_and_probe_defaults():
    """Load JSON defaults and probe valid keys and parameter types once at startup."""
    json_path = Path(CONFIG["json_file"])
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path

    with json_path.open("r", encoding="utf-8") as file:
        raw_defaults = json.load(file)

    probe_model = TP.default(CONFIG["model"])
    valid_defaults = {}
    failed_keys = []

    for key, value in raw_defaults.items():
        if key == "number_inputs":
            continue
        try:
            probe_model.value(key, value)
            valid_defaults[key] = value
        except Exception as exc:
            failed_keys.append((key, str(exc)))

    if failed_keys:
        logger.warning(
            "Ignored %d unsupported JSON defaults at startup: %s",
            len(failed_keys),
            [k for k, _ in failed_keys],
        )

    override_keys = set(
        CONFIG["design"]["overrides"]
        + CONFIG["operational"]["overrides"]
        + ["m_dot_htfmin", "m_dot_htfmax"]
    )
    param_kinds = {}
    for key in override_keys:
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


VALID_JSON_DEFAULTS, PARAMETER_KINDS = load_and_probe_defaults()


# --------------------------------------------------
# Physical constraint validation
# --------------------------------------------------


def validate_physical_constraints(overrides: dict) -> bool:
    """Validate geometric feasibility (Row_Distance > W_aperture) before simulation."""
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


# --------------------------------------------------
# Parameter preparation & model initialization
# --------------------------------------------------


def normalize_overrides(overrides: dict) -> dict:
    """Copy overrides and expand 'm_dot' into min and max HTF mass-flow parameters."""
    if not overrides:
        return {}

    normalized = dict(overrides)
    if "m_dot" in normalized:
        m_dot = normalized.pop("m_dot")
        normalized.setdefault("m_dot_htfmin", m_dot)
        normalized.setdefault("m_dot_htfmax", m_dot)

    return normalized


def initialize_model():
    """Create a fresh PySAM model instance populated with validated default parameters."""
    model = TP.default(CONFIG["model"])
    for key, value in VALID_JSON_DEFAULTS.items():
        model.value(key, value)
    return model


def apply_overrides(model, overrides: dict) -> None:
    """Apply parameter overrides to the PySAM model, handling scalar and sequence types."""
    for key, value in overrides.items():
        kind = PARAMETER_KINDS.get(key)

        if kind == "scalar":
            model.value(key, value)
        elif kind == "sequence":
            current_value = model.value(key)
            new_value = (
                replace_1st_order(current_value, value)
                if len(current_value) > 0
                else ((value,) if isinstance(current_value, tuple) else [value])
            )
            model.value(key, new_value)
        else:
            try:
                current_value = model.value(key)
            except Exception as exc:
                raise ValueError(
                    f"Unknown or inaccessible PySAM parameter: {key}"
                ) from exc

            if isinstance(current_value, (Real, bool)):
                model.value(key, value)
            elif isinstance(current_value, (list, tuple)):
                new_value = (
                    replace_1st_order(current_value, value)
                    if len(current_value) > 0
                    else ((value,) if isinstance(current_value, tuple) else [value])
                )
                model.value(key, new_value)
            else:
                raise TypeError(
                    f"Unsupported parameter type for '{key}': {type(current_value).__name__}"
                )


# --------------------------------------------------
# Simulation execution
# --------------------------------------------------


def _run_simulation_core(
    overrides: dict, day_index: int | None = None
) -> SimulationOutputs:
    """Execute a single PySAM simulation run and extract output metrics."""
    normalized = normalize_overrides(overrides)
    model = initialize_model()
    apply_overrides(model, normalized)

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

    return {
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


def run_simulation(
    overrides: dict,
    day_index: int | None = None,
) -> tuple[SimulationOutputs | None, bool]:
    """Run one PySAM simulation with constraint checks, timing, and penalty fallback."""
    display_params = overrides or {"Mode": "Baseline"}

    if not validate_physical_constraints(overrides or {}):
        logger.warning("Simulation skipped due to physical constraint violation.")
        return None, True

    start_time = time.perf_counter()
    try:
        result = _run_simulation_core(overrides or {}, day_index=day_index)
        duration = time.perf_counter() - start_time

        if CONFIG.get("log_each_simulation", False):
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
        logger.warning("Applying penalty: %.0e", CONFIG["penalty"])
        return None, True


# --------------------------------------------------
# Standalone execution demo
# --------------------------------------------------

if __name__ == "__main__":
    print("Testing PySAM single-day simulation (Day 140)...")
    test_overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
        "ColperSCA": 4,
        "L_SCA": 100,
    }
    result, penalty = run_simulation(test_overrides, day_index=140)
    if penalty or result is None:
        print("Simulation failed (penalty applied).")
    else:
        day_slice = result["hourly_energy"][140 * 24 : 141 * 24]
        peak_mw = max(day_slice)
        print("Simulation completed successfully!")
        print(f"Peak electric output on Day 140: {peak_mw:.4f} MW")
        print(f"Total land area: {result['land_area']:.2f} acres")
        print(f"Installed capital cost: ${result['total_installed_cost']:,.2f}")
