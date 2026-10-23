"""
Tests for vectorized economic objective function.
"""

import numpy as np

from solar_opt.objective import compute_daily_objective_value
from solar_opt.simulation import DailySimulationSlice


def test_compute_daily_objective_value_standard():
    # 24-hour dummy slice with known outputs
    hourly_energy = np.ones(24, dtype=np.float32) * 5.0  # 5 MW
    parasitic = np.ones(24, dtype=np.float32) * 0.5  # 0.5 MW
    losses = np.ones(24, dtype=np.float32) * 1.0  # 1.0 MW thermal

    sim_slice = DailySimulationSlice(
        hourly_energy=hourly_energy,
        field_htf_pump_power=parasitic / 3,
        pc_htf_pump_power=parasitic / 3,
        collector_tracking_power=parasitic / 3,
        pc_startup_thermal=losses / 3,
        piping_loss=losses / 3,
        receiver_loss=losses / 3,
        annual_energy=100.0,
        land_area=50.0,
        total_installed_cost=1000.0,
    )

    # Constant tariff of 4.0 INR/kWh -> 4000 INR/MWh
    tariff_8760 = np.full(8760, 4.0, dtype=np.float32)
    # Revenue = 24 * 5 * 4000 = 480,000 INR
    # Parasitic = 24 * 0.5 * 4000 = 48,000 INR
    # Thermal loss = 24 * 1.0 * 4000 * 0.4 = 38,400 INR
    # Net = 480000 - 48000 - 38400 = 393,600 INR
    val = compute_daily_objective_value(sim_slice, day_index=0, shared_tariff_8760=tariff_8760)
    assert np.isclose(val, 393600.0, rtol=1e-3)


def test_smart_no_dispatch_logic():
    # Overcast day with zero generation and parasitic losses
    zero_energy = np.zeros(24, dtype=np.float32)
    parasitic = np.ones(24, dtype=np.float32) * 0.2

    sim_slice = DailySimulationSlice(
        hourly_energy=zero_energy,
        field_htf_pump_power=parasitic,
        pc_htf_pump_power=np.zeros(24, dtype=np.float32),
        collector_tracking_power=np.zeros(24, dtype=np.float32),
        pc_startup_thermal=np.zeros(24, dtype=np.float32),
        piping_loss=np.zeros(24, dtype=np.float32),
        receiver_loss=np.zeros(24, dtype=np.float32),
        annual_energy=0.0,
        land_area=50.0,
        total_installed_cost=1000.0,
    )

    tariff_8760 = np.full(8760, 4.0, dtype=np.float32)
    val = compute_daily_objective_value(sim_slice, day_index=0, shared_tariff_8760=tariff_8760)
    # In standby mode, net losses are capped at 0.0
    assert val == 0.0
