"""
PySAM Simulation module.
Backward compatibility wrapper delegating to solar_opt.simulation.
"""

from solar_opt.simulation.models import SimulationOutputs
from solar_opt.simulation.pysam_runner import (
    PySAMSimulationRunner,
    get_default_runner,
    replace_1st_order,
    run_simulation,
)

__all__ = [
    "SimulationOutputs",
    "PySAMSimulationRunner",
    "get_default_runner",
    "replace_1st_order",
    "run_simulation",
]

if __name__ == "__main__":
    print("Testing PySAM single-day simulation (Day 140)...")
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
        day_slice = result["hourly_energy"][140 * 24 : 141 * 24]
        peak_mw = max(day_slice)
        print("Simulation completed successfully!")
        print(f"Peak electric output on Day 140: {peak_mw:.4f} MW")
        print(f"Total land area: {result['land_area']:.2f} acres")
        print(f"Installed capital cost: ${result['total_installed_cost']:,.2f}")
