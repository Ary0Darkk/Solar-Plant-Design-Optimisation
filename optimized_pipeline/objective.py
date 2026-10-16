"""
Vectorized economic objective function operating on 24-hour float32 slices.
"""

from typing import Optional
import numpy as np

from .simulation import DailySimulationSlice
from .tariff import get_shared_tariff


def compute_daily_objective_value(
    sim_slice: DailySimulationSlice,
    day_index: int,
    shared_tariff_8760: Optional[np.ndarray] = None,
) -> float:
    """
    Compute daily net objective value in INR using vectorized 24-element NumPy operations.

    Formula:
        Net Objective = Revenue - Parasitic Cost - 0.4 * Thermal Loss Cost
    """
    start = day_index * 24
    end = start + 24

    tariff = shared_tariff_8760 if shared_tariff_8760 is not None else get_shared_tariff()

    # Tariff in INR/MWh (* 1,000) matching PySAM electrical outputs in MW
    prices_24h = tariff[start:end] * np.float32(1000.0)

    # 1. Electrical revenue
    revenue = float(np.dot(sim_slice.hourly_energy, prices_24h))

    # 2. Combined parasitic power cost (HTF field pump + cycle pump + SCA tracking)
    parasitic_load = (
        sim_slice.field_htf_pump_power
        + sim_slice.pc_htf_pump_power
        + sim_slice.collector_tracking_power
    )
    parasitic_cost = float(np.dot(parasitic_load, prices_24h))

    # 3. Combined thermal loss cost (startup thermal + piping thermal loss + receiver loss)
    # Scaled by 40% (0.4) equivalent cycle efficiency
    thermal_losses = (
        sim_slice.pc_startup_thermal
        + sim_slice.piping_loss
        + sim_slice.receiver_loss
    )
    thermal_loss_cost = float(np.dot(thermal_losses, prices_24h) * np.float32(0.4))

    return revenue - parasitic_cost - thermal_loss_cost
