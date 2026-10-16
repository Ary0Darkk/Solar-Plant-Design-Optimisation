"""
Unit tests for PySAM physical constraints and simulation runner.
"""

from optimized_pipeline.simulation import (
    PySAMSimulator,
    get_simulator,
    validate_constraints,
)


def test_constraint_validation():
    """Verify that Row_Distance <= W_aperture is rejected as collision infeasible."""
    # Valid: Row_Distance 16m > W_aperture 6m
    assert validate_constraints({"Row_Distance": 16.0, "W_aperture": 6.0}) is True

    # Infeasible: Row_Distance 5m <= W_aperture 6m
    assert validate_constraints({"Row_Distance": 5.0, "W_aperture": 6.0}) is False

    # Infeasible: Equal spacing
    assert validate_constraints({"Row_Distance": 6.0, "W_aperture": 6.0}) is False

    # Infeasible: T_shutdown > T_startup
    assert validate_constraints({"T_startup": 300, "T_shutdown": 350}) is False


def test_single_day_simulation_day_140():
    """Verify that a single-day simulation runs successfully and returns 24-hr float32 slice."""
    sim = get_simulator()
    test_overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
        "ColperSCA": 4,
        "L_SCA": 100,
        "m_dot": 5,
        "T_startup": 350,
        "T_shutdown": 280,
    }
    result_slice, is_penalized = sim.run_daily_simulation(test_overrides, day_index=140)

    assert is_penalized is False
    assert result_slice is not None
    assert len(result_slice.hourly_energy) == 24
    assert result_slice.land_area > 0
    assert result_slice.total_installed_cost > 0
