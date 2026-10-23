"""
Tests for physical constraint validation and PySAM simulation.
"""

from solar_opt.simulation import get_simulator, validate_constraints


def test_validate_constraints_feasible():
    valid_params = {
        "Row_Distance": 15,
        "W_aperture": 6,  # 15 - 6 = 9 >= 3.5 -> OK
        "T_startup": 350,
        "T_shutdown": 280,  # 350 - 280 = 70 >= 15 -> OK
    }
    assert validate_constraints(valid_params) is True


def test_validate_constraints_row_distance_too_close():
    invalid_params = {
        "Row_Distance": 8,
        "W_aperture": 6,  # 8 - 6 = 2 < 3.5 -> Infeasible
        "T_startup": 350,
        "T_shutdown": 280,
    }
    assert validate_constraints(invalid_params) is False


def test_validate_constraints_thermal_hysteresis_too_narrow():
    invalid_params = {
        "Row_Distance": 15,
        "W_aperture": 6,
        "T_startup": 300,
        "T_shutdown": 290,  # 300 - 290 = 10 < 15 -> Infeasible
    }
    assert validate_constraints(invalid_params) is False


def test_compact_simulator_execution():
    sim = get_simulator()
    overrides = {
        "specified_total_aperture": 10000,
        "Row_Distance": 15,
        "ColperSCA": 4,
        "W_aperture": 5,
        "L_SCA": 100,
        "m_dot": 5,
        "T_startup": 350,
        "T_shutdown": 280,
    }
    slice_res, penalized = sim.run_daily_simulation(overrides, day_index=67)
    assert not penalized
    assert slice_res is not None
    assert len(slice_res.hourly_energy) == 24
    assert slice_res.hourly_energy.dtype == "float32"
    assert isinstance(slice_res.annual_energy, float)
    assert slice_res.land_area > 0
    assert slice_res.total_installed_cost > 0
