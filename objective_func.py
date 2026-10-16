from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from demand_data import get_dynamic_price

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Dynamic Price Cache
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_cached_dynamic_price() -> np.ndarray:
    min_p = float(CONFIG.get("min_price", 3.0))
    max_p = float(CONFIG.get("max_price", 8.0))
    file_path = PROJECT_ROOT / "electricity_data" / "dynamic_price_data_haryana.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)
        if "dynamic_price" in df.columns:
            prices = np.asarray(df["dynamic_price"].values, dtype=float)
            if (
                len(prices) == 8760
                and np.isclose(prices.min(), min_p, atol=1e-2)
                and np.isclose(prices.max(), max_p, atol=1e-2)
            ):
                return prices

    dataset = get_dynamic_price(min_price=min_p, max_price=max_p)
    dataset.to_csv(file_path, index=False)
    return np.asarray(dataset["dynamic_price"].values, dtype=float)


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
    # Helper to handle both 8760-hour and 24-hour inputs
    # -----------------------------
    def _extract_24h(series) -> np.ndarray:
        if len(series) == 24:
            return np.asarray(series, dtype=float)
        return np.asarray(series[day_slice], dtype=float)

    # -----------------------------
    # price profile (scale to kWh)
    # -----------------------------
    dynamic_price = get_cached_dynamic_price()[day_slice] * 1_000.0

    # -----------------------------
    # electrical revenue & power cost terms
    # -----------------------------
    hourly_energy_cost_term = float(
        np.dot(_extract_24h(hourly_energy), dynamic_price)
    )

    field_htf_pump_power_cost_term = float(
        np.dot(_extract_24h(field_htf_pump_power), dynamic_price)
    )

    pc_htf_pump_power_cost_term = float(
        np.dot(_extract_24h(pc_htf_pump_power), dynamic_price)
    )

    field_collector_tracking_power_cost_term = float(
        np.dot(_extract_24h(field_collector_tracking_power), dynamic_price)
    )

    # -----------------------------
    # thermal losses (40% cycle efficiency equivalent)
    # -----------------------------
    pc_startup_thermal_power_cost_term = float(
        np.dot(_extract_24h(pc_startup_thermal_power), dynamic_price) * 0.4
    )

    field_piping_thermal_loss_cost_term = float(
        np.dot(_extract_24h(field_piping_thermal_loss), dynamic_price) * 0.4
    )

    receiver_thermal_loss_cost_term = float(
        np.dot(_extract_24h(receiver_thermal_loss), dynamic_price) * 0.4
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


if __name__ == "__main__":
    print("Testing objective_function with PySAM Day 140 simulation...")
    from simulation.simulation import run_simulation

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
        obj_val = objective_function(
            hourly_energy=result["hourly_energy"],
            field_htf_pump_power=result["field_htf_pump_power"],
            pc_htf_pump_power=result["pc_htf_pump_power"],
            field_collector_tracking_power=result["field_collector_tracking_power"],
            pc_startup_thermal_power=result["pc_startup_thermal_power"],
            field_piping_thermal_loss=result["field_piping_thermal_loss"],
            receiver_thermal_loss=result["receiver_thermal_loss"],
            day_index=140,
        )
        print("Objective function evaluated successfully!")
        print(f"Daily Objective Value (Day 140): ${obj_val:,.2f}")
