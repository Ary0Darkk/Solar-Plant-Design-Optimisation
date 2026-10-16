"""
Unit tests for daily objective function calculation.
"""

from solar_opt.economic.objective import calculate_daily_objective


def test_zero_energy_zero_objective():
    """Verify that zero power and zero losses result in zero objective."""
    zeros = [0.0] * 24
    obj = calculate_daily_objective(
        hourly_energy=zeros,
        field_htf_pump_power=zeros,
        pc_htf_pump_power=zeros,
        field_collector_tracking_power=zeros,
        pc_startup_thermal_power=zeros,
        field_piping_thermal_loss=zeros,
        receiver_thermal_loss=zeros,
        day_index=10,
    )
    assert obj == 0.0


def test_positive_generation_positive_objective():
    """Verify that net generation results in positive revenue."""
    zeros = [0.0] * 24
    generation = [1.0] * 24  # 1 MW each hour
    obj = calculate_daily_objective(
        hourly_energy=generation,
        field_htf_pump_power=zeros,
        pc_htf_pump_power=zeros,
        field_collector_tracking_power=zeros,
        pc_startup_thermal_power=zeros,
        field_piping_thermal_loss=zeros,
        receiver_thermal_loss=zeros,
        day_index=10,
    )
    assert obj > 0.0


def test_losses_penalize_objective():
    """Verify that parasitic loads and thermal losses decrease the objective value."""
    zeros = [0.0] * 24
    generation = [2.0] * 24

    obj_base = calculate_daily_objective(
        hourly_energy=generation,
        field_htf_pump_power=zeros,
        pc_htf_pump_power=zeros,
        field_collector_tracking_power=zeros,
        pc_startup_thermal_power=zeros,
        field_piping_thermal_loss=zeros,
        receiver_thermal_loss=zeros,
        day_index=10,
    )

    losses = [0.5] * 24
    obj_with_losses = calculate_daily_objective(
        hourly_energy=generation,
        field_htf_pump_power=losses,
        pc_htf_pump_power=losses,
        field_collector_tracking_power=losses,
        pc_startup_thermal_power=losses,
        field_piping_thermal_loss=losses,
        receiver_thermal_loss=losses,
        day_index=10,
    )

    assert obj_with_losses < obj_base
