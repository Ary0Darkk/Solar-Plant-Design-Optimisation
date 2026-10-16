"""
Unit tests for PySAM physical constraints and simulation runner.
"""

from solar_opt.simulation.pysam_runner import (
    PySAMSimulationRunner,
    get_default_runner,
    run_simulation,
)


def test_constraint_validation():
    """Verify that Row_Distance <= W_aperture is rejected as collision infeasible."""
    runner = get_default_runner()

    # Valid: Row_Distance 16m > W_aperture 6m
    assert runner.validate_physical_constraints({"Row_Distance": 16.0, "W_aperture": 6.0}) is True

    # Infeasible: Row_Distance 5m <= W_aperture 6m
    assert runner.validate_physical_constraints({"Row_Distance": 5.0, "W_aperture": 6.0}) is False

    # Infeasible: Equal spacing
    assert runner.validate_physical_constraints({"Row_Distance": 6.0, "W_aperture": 6.0}) is False


def test_single_day_simulation_day_140():
    """Verify that a single-day simulation runs successfully and returns positive energy."""
    test_overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
        "ColperSCA": 4,
        "L_SCA": 100,
    }
    result, penalty = run_simulation(test_overrides, day_index=140)

    assert penalty is False
    assert result is not None
    assert len(result["hourly_energy"]) == 8760
    assert result["land_area"] > 0
    assert result["total_installed_cost"] > 0
