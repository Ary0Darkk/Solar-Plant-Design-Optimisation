"""
Economic objective function module.
Backward compatibility wrapper delegating to solar_opt.economic.
"""

from solar_opt.economic.objective import calculate_daily_objective as objective_function
from solar_opt.economic.pricing import get_cached_dynamic_prices as get_cached_dynamic_price

__all__ = ["objective_function", "get_cached_dynamic_price"]

if __name__ == "__main__":
    from solar_opt.simulation.pysam_runner import run_simulation

    print("Testing objective_function with PySAM Day 140 simulation...")
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
