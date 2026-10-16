"""
Economic objective function for Concentrated Solar Power (CSP) evaluation.
"""

from typing import Sequence
import numpy as np

from .pricing import get_cached_dynamic_prices


def calculate_daily_objective(
    hourly_energy: Sequence[float],
    field_htf_pump_power: Sequence[float],
    pc_htf_pump_power: Sequence[float],
    field_collector_tracking_power: Sequence[float],
    pc_startup_thermal_power: Sequence[float],
    field_piping_thermal_loss: Sequence[float],
    receiver_thermal_loss: Sequence[float],
    day_index: int,
    min_price: float = 3.0,
    max_price: float = 8.0,
) -> float:
    """
    Calculate the net daily economic objective value (INR) for a 24-hour simulation slice.

    Economic balance:
      Objective = Electrical Revenue
                - Field HTF Pump Cost
                - Power Cycle HTF Pump Cost
                - Collector Tracking Cost
                - 0.4 * PC Startup Thermal Cost
                - 0.4 * Field Piping Thermal Loss Cost
                - 0.4 * Receiver Thermal Loss Cost

    Thermal terms are weighted by 40% (0.4) to represent thermal-to-electric cycle efficiency.
    """
    start_hour = day_index * 24
    end_hour = start_hour + 24
    day_slice = slice(start_hour, end_hour)

    def _extract_24h(series: Sequence[float]) -> np.ndarray:
        if len(series) == 24:
            return np.asarray(series, dtype=float)
        return np.asarray(series[day_slice], dtype=float)

    # Convert dynamic price tariff from INR/kWh to INR/MWh (* 1,000.0)
    # matching PySAM electrical outputs in MW
    dynamic_prices = get_cached_dynamic_prices(min_price=min_price, max_price=max_price)[day_slice] * 1000.0

    # Electrical generation revenue
    revenue = float(np.dot(_extract_24h(hourly_energy), dynamic_prices))

    # Parasitic electrical pumping and tracking costs
    field_pump_cost = float(np.dot(_extract_24h(field_htf_pump_power), dynamic_prices))
    pc_pump_cost = float(np.dot(_extract_24h(pc_htf_pump_power), dynamic_prices))
    tracking_cost = float(np.dot(_extract_24h(field_collector_tracking_power), dynamic_prices))

    # Thermal energy losses (40% equivalent electric penalty)
    startup_loss_cost = float(np.dot(_extract_24h(pc_startup_thermal_power), dynamic_prices) * 0.4)
    piping_loss_cost = float(np.dot(_extract_24h(field_piping_thermal_loss), dynamic_prices) * 0.4)
    receiver_loss_cost = float(np.dot(_extract_24h(receiver_thermal_loss), dynamic_prices) * 0.4)

    net_objective = (
        revenue
        - field_pump_cost
        - pc_pump_cost
        - tracking_cost
        - startup_loss_cost
        - piping_loss_cost
        - receiver_loss_cost
    )

    return float(net_objective)
