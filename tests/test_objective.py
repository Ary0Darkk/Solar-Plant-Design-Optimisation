"""
Tests for economic objective calculation and smart standby logic.
"""

import numpy as np
import pytest

from solar_opt.objective import ObjectiveResult, compute_daily_objective_value
from solar_opt.simulation import DailySimulationSlice


def test_compute_daily_objective_value_standard():
    energy = np.ones(24, dtype=np.float32)  # 1 MWh/hr
    par = np.full(24, 0.1, dtype=np.float32)  # 0.1 MWh/hr
    loss = np.full(24, 0.2, dtype=np.float32)  # 0.2 MWt/hr
    tariff = np.full(24, 4000.0, dtype=np.float32)  # 4000 INR/MWh

    slice_data = DailySimulationSlice(
        day_index=0,
        hourly_energy=energy,
        hourly_parasitic=par,
        hourly_heat_loss=loss,
    )

    res = compute_daily_objective_value(slice_data, tariff)
    assert isinstance(res, ObjectiveResult)
    assert len(res) == 4

    net_profit, gross_rev, par_cost, loss_cost = res
    assert gross_rev == pytest.approx(24 * 1.0 * 4000.0)  # 96,000 INR
    assert par_cost == pytest.approx(24 * 0.1 * 4000.0)  # 9,600 INR
    assert loss_cost == pytest.approx(24 * (0.4 * 0.2) * 4000.0)  # 7,680 INR
    assert net_profit == pytest.approx(gross_rev - par_cost - loss_cost)


def test_smart_no_dispatch_logic():
    # Zero energy generation with parasitics and heat loss (cloudy/night)
    energy = np.zeros(24, dtype=np.float32)
    par = np.full(24, 0.1, dtype=np.float32)
    loss = np.full(24, 0.2, dtype=np.float32)
    tariff = np.full(24, 4000.0, dtype=np.float32)

    slice_data = DailySimulationSlice(
        day_index=0,
        hourly_energy=energy,
        hourly_parasitic=par,
        hourly_heat_loss=loss,
    )

    net_profit, gross_rev, _, _ = compute_daily_objective_value(slice_data, tariff)
    assert gross_rev == 0.0
    # Standby logic clamps negative profit to 0 when generation is 0
    assert net_profit == 0.0
