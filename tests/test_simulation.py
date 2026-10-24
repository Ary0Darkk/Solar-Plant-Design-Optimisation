"""
Tests for PySAM simulator, physical feasibility constraints, and 24-hour slicing.
"""

from solar_opt.config import PipelineConfig
from solar_opt.simulation import (
    CompactCSPPlantSimulator,
    validate_physical_constraints,
)


def test_validate_constraints_feasible():
    params = {
        "Row_Distance": 15.0,
        "W_aperture": 6.0,
        "T_startup": 350.0,
        "T_shutdown": 280.0,
    }
    valid, msg = validate_physical_constraints(params)
    assert valid is True
    assert msg == "Valid"


def test_validate_constraints_row_distance_too_close():
    # Clearance < 3.5m (8.0 - 6.0 = 2.0m)
    params = {
        "Row_Distance": 8.0,
        "W_aperture": 6.0,
        "T_startup": 350.0,
        "T_shutdown": 280.0,
    }
    valid, msg = validate_physical_constraints(params)
    assert valid is False
    assert "Corridor clearance violated" in msg


def test_validate_constraints_thermal_hysteresis_too_narrow():
    # Deadband < 15°C (300.0 - 295.0 = 5.0°C)
    params = {
        "Row_Distance": 15.0,
        "W_aperture": 6.0,
        "T_startup": 300.0,
        "T_shutdown": 295.0,
    }
    valid, msg = validate_physical_constraints(params)
    assert valid is False
    assert "Anti-chattering deadband violated" in msg


def test_compact_simulator_execution():
    cfg = PipelineConfig()
    sim = CompactCSPPlantSimulator(cfg.simulation)

    overrides = {
        "specified_total_aperture": 8745.0,
        "Row_Distance": 23.0,
        "ColperSCA": 2,
        "W_aperture": 6.0,
        "L_SCA": 152.0,
        "m_dot": 4.0,
        "T_startup": 313.0,
        "T_shutdown": 292.0,
    }

    slice_out, penalized = sim.run_daily_simulation(overrides, day_index=129)
    assert penalized is False
    assert slice_out is not None
    assert len(slice_out.hourly_energy) == 24
    assert len(slice_out.hourly_parasitic) == 24
    assert len(slice_out.hourly_heat_loss) == 24
