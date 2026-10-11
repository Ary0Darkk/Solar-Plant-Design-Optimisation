import json
import logging
import time
from pathlib import Path
from numbers import Real

import PySAM.TroughPhysical as TP

from config import CONFIG
from utilities.list_nesting import replace_1st_order
from utilities.setup_custom_logger import setup_custom_logger


# --------------------------------------------------
# Logger setup
# --------------------------------------------------

LOGGER_NAME = "NYS_Optimisation"
logger = logging.getLogger(LOGGER_NAME)

if not logger.hasHandlers():
    logger = setup_custom_logger()


# --------------------------------------------------
# Load JSON defaults once per Python process
# --------------------------------------------------


def load_json_defaults():
    """Load the PySAM model defaults from the configured JSON file."""

    json_path = Path(CONFIG["json_file"])

    with json_path.open("r", encoding="utf-8") as file:
        defaults = json.load(file)

    if not isinstance(defaults, dict):
        raise ValueError(
            f"Expected a JSON object in {json_path}, got {type(defaults).__name__}"
        )

    return defaults


JSON_DEFAULTS = load_json_defaults()


# --------------------------------------------------
# Prepare simulation overrides
# --------------------------------------------------


def normalize_overrides(overrides: dict) -> dict:
    """
    Copy overrides and expand m_dot into the minimum
    and maximum HTF mass-flow parameters.
    """

    if overrides is None:
        return {}

    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a dictionary.")

    normalized = dict(overrides)

    if "m_dot" in normalized:
        m_dot = normalized.pop("m_dot")

        # Preserve explicitly supplied bounds if present.
        normalized.setdefault("m_dot_htfmin", m_dot)
        normalized.setdefault("m_dot_htfmax", m_dot)

    return normalized


# --------------------------------------------------
# Initialize PySAM model
# --------------------------------------------------


def initialize_model():
    """
    Create a fresh PySAM model and apply JSON defaults.

    Unsupported JSON default keys are collected and reported
    rather than silently ignored.
    """

    model = TP.default(CONFIG["model"])
    failed_defaults = []

    for key, value in JSON_DEFAULTS.items():
        if key == "number_inputs":
            continue

        try:
            model.value(key, value)
        except Exception as exc:
            failed_defaults.append((key, str(exc)))

    if failed_defaults:
        failed_keys = [key for key, _ in failed_defaults]

        logger.warning(
            "Could not apply %d JSON defaults: %s",
            len(failed_keys),
            failed_keys,
        )

        if CONFIG.get("strict_json_defaults", False):
            raise ValueError(f"Failed to apply JSON defaults: {failed_keys}")

    return model


# --------------------------------------------------
# Apply optimization parameters
# --------------------------------------------------


def apply_overrides(model, overrides: dict):
    """
    Apply scalar overrides or replace the first element
    of an existing sequence, preserving the original behavior.
    """

    for key, value in overrides.items():
        try:
            current_value = model.value(key)
        except Exception as exc:
            raise ValueError(f"Unknown or inaccessible PySAM parameter: {key}") from exc

        # Scalar parameters
        if isinstance(current_value, (Real, bool)):
            model.value(key, value)

        # Sequence parameters: replace first element
        elif isinstance(current_value, (list, tuple)):
            if len(current_value) > 0:
                new_value = replace_1st_order(
                    data=current_value,
                    new_val=value,
                )
            elif isinstance(current_value, tuple):
                new_value = (value,)
            else:
                new_value = [value]

            model.value(key, new_value)

        else:
            raise TypeError(
                f"Unsupported parameter type for '{key}': "
                f"{type(current_value).__name__}"
            )


# --------------------------------------------------
# Execute PySAM simulation
# --------------------------------------------------


def _run_simulation_core(overrides: dict) -> dict:
    """Execute one fresh PySAM simulation."""

    normalized = normalize_overrides(overrides)

    model = initialize_model()
    apply_overrides(model, normalized)

    model.execute()

    outputs = model.Outputs

    # Preserve the output interface used by the optimizer.
    return {
        "hourly_energy": outputs.P_cycle,
        "pc_htf_pump_power": outputs.cycle_htf_pump_power,
        "field_htf_pump_power": outputs.W_dot_field_pump,
        "field_collector_tracking_power": outputs.W_dot_sca_track,
        "pc_startup_thermal_power": outputs.q_dot_pc_startup,
        "field_piping_thermal_loss": outputs.q_dot_piping_loss,
        "receiver_thermal_loss": outputs.q_dot_rec_thermal_loss,
        "annual_energy": outputs.annual_energy,
        "gross_annual_energy": outputs.annual_W_cycle_gross,
        "land_area": outputs.total_land_area,
        "land_cost": outputs.csp_dtr_cost_plm_total,
        "total_installed_cost": outputs.total_installed_cost,
    }


# --------------------------------------------------
# Public simulation entry point
# --------------------------------------------------


def run_simulation(overrides: dict):
    """
    Run one simulation.

    Returns:
        (result, penalty_flag)

        result:
            Dictionary of simulation outputs on success,
            otherwise None.

        penalty_flag:
            False on success, True on failure.
    """

    display_params = overrides or {"Mode": "Baseline"}

    start_time = time.perf_counter()

    try:
        result = _run_simulation_core(overrides or {})
        duration = time.perf_counter() - start_time

        # Avoid large multi-column tables in high-volume GA logs.
        if CONFIG.get("log_each_simulation", False):
            logger.info(
                "Simulation completed in %.2f s | Parameters: %s",
                duration,
                display_params,
            )

        return result, False

    except Exception:
        duration = time.perf_counter() - start_time

        logger.error(
            "Simulation failed after %.2f s | Parameters: %s",
            duration,
            display_params,
            exc_info=True,
        )

        logger.warning(
            "Applying penalty: %.0e",
            CONFIG["penalty"],
        )

        return None, True
