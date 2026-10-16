"""
Unit tests for daily objective function calculation.
"""

import numpy as np

from optimized_pipeline.objective import (
    DailySimulationSlice,
    compute_daily_objective_value,
)


def test_zero_energy_zero_objective():
    """Verify that zero power and zero losses result in zero objective."""
    zeros = np.zeros(24, dtype=np.float32)
    slice_data = DailySimulationSlice(
        hourly_energy=zeros,
        field_htf_pump_power=zeros,
        pc_htf_pump_power=zeros,
        collector_tracking_power=zeros,
        pc_startup_thermal=zeros,
        piping_loss=zeros,
        receiver_loss=zeros,
        annual_energy=0.0,
        land_area=100.0,
        total_installed_cost=1000.0,
    )
    obj = compute_daily_objective_value(slice_data, day_index=10)
    assert np.isclose(obj, 0.0)


def test_positive_generation_positive_objective():
    """Verify that net generation results in positive revenue."""
    zeros = np.zeros(24, dtype=np.float32)
    gen = np.ones(24, dtype=np.float32)  # 1 MW each hour
    slice_data = DailySimulationSlice(
        hourly_energy=gen,
        field_htf_pump_power=zeros,
        pc_htf_pump_power=zeros,
        collector_tracking_power=zeros,
        pc_startup_thermal=zeros,
        piping_loss=zeros,
        receiver_loss=zeros,
        annual_energy=24.0,
        land_area=100.0,
        total_installed_cost=1000.0,
    )
    obj = compute_daily_objective_value(slice_data, day_index=10)
    assert obj > 0.0


def test_losses_penalize_objective():
    """Verify that parasitic loads and thermal losses decrease the objective value."""
    zeros = np.zeros(24, dtype=np.float32)
    gen = np.full(24, 2.0, dtype=np.float32)

    slice_base = DailySimulationSlice(
        hourly_energy=gen,
        field_htf_pump_power=zeros,
        pc_htf_pump_power=zeros,
        collector_tracking_power=zeros,
        pc_startup_thermal=zeros,
        piping_loss=zeros,
        receiver_loss=zeros,
        annual_energy=48.0,
        land_area=100.0,
        total_installed_cost=1000.0,
    )
    obj_base = compute_daily_objective_value(slice_base, day_index=10)

    losses = np.full(24, 0.5, dtype=np.float32)
    slice_with_losses = DailySimulationSlice(
        hourly_energy=gen,
        field_htf_pump_power=losses,
        pc_htf_pump_power=losses,
        collector_tracking_power=losses,
        pc_startup_thermal=losses,
        piping_loss=losses,
        receiver_loss=losses,
        annual_energy=48.0,
        land_area=100.0,
        total_installed_cost=1000.0,
    )
    obj_with_losses = compute_daily_objective_value(slice_with_losses, day_index=10)

    assert obj_with_losses < obj_base
