import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

from demand_data import get_dynamic_price


# ---------------------------------------------------------
# Dynamic Price Cache
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_cached_dynamic_price() -> np.ndarray:
    file_path = Path("electricity_data/dynamic_price_data_haryana.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)
        return np.asarray(df["dynamic_price"].values, dtype=float)

    else:
        return np.asarray(get_dynamic_price()["dynamic_price"].values, dtype=float)


# ---------------------------------------------------------
# Daily Objective Function
# ---------------------------------------------------------


def objective_function(
    hourly_energy: list[float],
    field_htf_pump_power: list[float],
    pc_htf_pump_power: list[float],
    field_collector_tracking_power: list[float],
    pc_startup_thermal_power: list[float],
    field_piping_thermal_loss: list[float],
    receiver_thermal_loss: list[float],
    day_index: int,
) -> float:
    """
    Calculates DAILY objective function using exact 24-hour hourly aggregation.
    """

    # -----------------------------
    # map day -> 24 hour slice
    # -----------------------------
    start = day_index * 24
    end = start + 24
    day_slice = slice(start, end)

    # -----------------------------
    # price profile (scale to kWh)
    # -----------------------------
    dynamic_price = get_cached_dynamic_price()[day_slice] * 1_000.0

    # -----------------------------
    # electrical revenue & power cost terms
    # -----------------------------
    hourly_energy_cost_term = float(
        np.dot(hourly_energy[day_slice], dynamic_price)
    )

    field_htf_pump_power_cost_term = float(
        np.dot(field_htf_pump_power[day_slice], dynamic_price)
    )

    pc_htf_pump_power_cost_term = float(
        np.dot(pc_htf_pump_power[day_slice], dynamic_price)
    )

    field_collector_tracking_power_cost_term = float(
        np.dot(field_collector_tracking_power[day_slice], dynamic_price)
    )

    # -----------------------------
    # thermal losses (40% cycle efficiency equivalent)
    # -----------------------------
    pc_startup_thermal_power_cost_term = float(
        np.dot(pc_startup_thermal_power[day_slice], dynamic_price) * 0.4
    )

    field_piping_thermal_loss_cost_term = float(
        np.dot(field_piping_thermal_loss[day_slice], dynamic_price) * 0.4
    )

    receiver_thermal_loss_cost_term = float(
        np.dot(receiver_thermal_loss[day_slice], dynamic_price) * 0.4
    )

    # -----------------------------
    # objective
    # -----------------------------
    obj = (
        hourly_energy_cost_term
        - field_htf_pump_power_cost_term
        - pc_htf_pump_power_cost_term
        - field_collector_tracking_power_cost_term
        - pc_startup_thermal_power_cost_term
        - field_piping_thermal_loss_cost_term
        - receiver_thermal_loss_cost_term
    )

    return float(obj)
