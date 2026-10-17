"""
Unit tests for PySAM physical constraints and simulation runner.
"""

import numpy as np
import pytest

from optimized_pipeline.config import SimulationConfig
from optimized_pipeline.simulation import (
    CompactSimulator,
    DailySimulationSlice,
    get_simulator,
    validate_constraints,
)


def test_constraint_validation():
    """Verify geometric spacing corridor and thermal deadband constraint enforcement."""
    # Empty or partial parameters pass smoothly without crashing
    assert validate_constraints({}) is True
    assert validate_constraints({"Row_Distance": 16.0}) is True
    assert validate_constraints({"T_startup": 350}) is True

    # Valid: Row_Distance 16m - W_aperture 6m = 10m >= 3.5m
    assert validate_constraints({"Row_Distance": 16.0, "W_aperture": 6.0}) is True

    # Exact boundary condition: Row_Distance 9.5m - W_aperture 6m = 3.5m >= 3.5m
    assert validate_constraints({"Row_Distance": 9.5, "W_aperture": 6.0}) is True

    # Just below boundary: Row_Distance 9.49m - W_aperture 6m = 3.49m < 3.5m
    assert validate_constraints({"Row_Distance": 9.49, "W_aperture": 6.0}) is False

    # Infeasible: Row_Distance 8m - W_aperture 6m = 2m < 3.5m (truck corridor violation)
    assert validate_constraints({"Row_Distance": 8.0, "W_aperture": 6.0}) is False

    # Infeasible: Equal spacing
    assert validate_constraints({"Row_Distance": 6.0, "W_aperture": 6.0}) is False

    # Infeasible: Row_Distance 5m <= W_aperture 6m (collision)
    assert validate_constraints({"Row_Distance": 5.0, "W_aperture": 6.0}) is False

    # Valid thermal deadband: 350°C - 300°C = 50°C >= 15°C
    assert validate_constraints({"T_startup": 350, "T_shutdown": 300}) is True

    # Exact boundary deadband: 350°C - 335°C = 15°C >= 15°C
    assert validate_constraints({"T_startup": 350, "T_shutdown": 335}) is True

    # Infeasible thermal deadband: 350°C - 336°C = 14°C < 15°C
    assert validate_constraints({"T_startup": 350, "T_shutdown": 336}) is False

    # Infeasible: T_shutdown > T_startup
    assert validate_constraints({"T_startup": 300, "T_shutdown": 350}) is False


def test_simulator_singleton_and_probing():
    """Verify that get_simulator returns a shared singleton with probed defaults."""
    sim1 = get_simulator()
    sim2 = get_simulator()
    assert sim1 is sim2

    assert len(sim1.valid_defaults) > 0
    assert "Row_Distance" in sim1.param_kinds
    assert sim1.param_kinds["Row_Distance"] == "scalar"


def test_simulator_missing_config_raises():
    """Verify that CompactSimulator raises FileNotFoundError if JSON config is missing."""
    bogus_cfg = SimulationConfig(json_file="nonexistent_plant_file_9999.json")
    with pytest.raises(FileNotFoundError):
        CompactSimulator(config=bogus_cfg)


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
    assert isinstance(result_slice, DailySimulationSlice)

    # Check 24-hr buffer types, shapes, and contiguity
    for attr in [
        "hourly_energy",
        "field_htf_pump_power",
        "pc_htf_pump_power",
        "collector_tracking_power",
        "pc_startup_thermal",
        "piping_loss",
        "receiver_loss",
    ]:
        arr = getattr(result_slice, attr)
        assert isinstance(arr, np.ndarray), f"{attr} should be a NumPy ndarray"
        assert arr.dtype == np.float32, f"{attr} should be float32"
        assert arr.shape == (24,), f"{attr} shape should be (24,)"
        assert arr.flags.c_contiguous, f"{attr} should be C-contiguous"

    # Check plant scalar outputs
    assert isinstance(result_slice.annual_energy, float)
    assert isinstance(result_slice.land_area, float)
    assert isinstance(result_slice.total_installed_cost, float)
    assert result_slice.land_area > 0
    assert result_slice.total_installed_cost > 0


def test_simulation_constraint_violation_returns_penalty():
    """Verify that run_daily_simulation short-circuits on constraint violations."""
    sim = get_simulator()

    # Geometry violation
    bad_geom = {"Row_Distance": 6.0, "W_aperture": 6.0}
    res, is_pen = sim.run_daily_simulation(bad_geom, day_index=67)
    assert res is None
    assert is_pen is True

    # Thermal deadband violation
    bad_thermal = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
        "T_startup": 300,
        "T_shutdown": 295,  # Only 5°C deadband < 15°C
    }
    res, is_pen = sim.run_daily_simulation(bad_thermal, day_index=67)
    assert res is None
    assert is_pen is True


def test_simulation_multi_season_execution():
    """Verify daily simulations run across different seasonal days without error."""
    sim = get_simulator()
    overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
        "ColperSCA": 4,
        "L_SCA": 100,
        "m_dot": 5,
        "T_startup": 350,
        "T_shutdown": 280,
    }

    # Winter Day 1 (Jan 8, day_index=7) and Summer Day 1 (March 9, day_index=67)
    for day_idx in [7, 67]:
        slice_res, is_pen = sim.run_daily_simulation(overrides, day_index=day_idx)
        assert is_pen is False
        assert slice_res is not None
        assert len(slice_res.hourly_energy) == 24


def test_simulation_runtime_failure_handling(monkeypatch):
    """Verify that runtime PySAM execution exceptions are caught and penalized."""
    sim = get_simulator()

    # Simulate a PySAM runtime solver exception during execute()
    def failing_apply_overrides(model, params):
        raise RuntimeError("PySAM C++ simulation convergence failed")

    monkeypatch.setattr(sim, "_apply_overrides", failing_apply_overrides)

    overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
    }
    res, is_pen = sim.run_daily_simulation(overrides, day_index=67)
    assert res is None
    assert is_pen is True
