"""
Unit tests for PySAM physical constraints and simulation runner.
"""

from optimized_pipeline.simulation import (
    get_simulator,
    validate_constraints,
)


def test_constraint_validation():
    """Verify that Row_Distance - W_aperture >= 3.5m and T_startup - T_shutdown >= 15C are enforced."""
    # Valid: Row_Distance 16m - W_aperture 6m = 10m >= 3.5m
    assert validate_constraints({"Row_Distance": 16.0, "W_aperture": 6.0}) is True

    # Infeasible: Row_Distance 8m - W_aperture 6m = 2m < 3.5m (maintenance truck corridor violation)
    assert validate_constraints({"Row_Distance": 8.0, "W_aperture": 6.0}) is False

    # Infeasible: Row_Distance 5m <= W_aperture 6m
    assert validate_constraints({"Row_Distance": 5.0, "W_aperture": 6.0}) is False

    # Infeasible: Equal spacing
    assert validate_constraints({"Row_Distance": 6.0, "W_aperture": 6.0}) is False

    # Valid thermal deadband: 350°C - 300°C = 50°C >= 15°C
    assert validate_constraints({"T_startup": 350, "T_shutdown": 300}) is True

    # Infeasible thermal deadband: 300°C - 290°C = 10°C < 15°C
    assert validate_constraints({"T_startup": 300, "T_shutdown": 290}) is False

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
