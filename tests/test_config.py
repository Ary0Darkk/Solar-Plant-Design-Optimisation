"""
Unit tests for configuration models and defaults.
"""

import pytest

from solar_opt.config import (
    DEFAULT_DESIGN_BOUNDS,
    DEFAULT_OPERATIONAL_BOUNDS,
    DEFAULT_SEASONS,
    VariableBounds,
    get_default_config,
)


def test_default_seasons():
    """Verify all 4 seasons exist and have exactly 7 representative days."""
    assert set(DEFAULT_SEASONS.keys()) == {"winter", "summer", "monsoon", "post"}
    for season, days in DEFAULT_SEASONS.items():
        assert len(days) == 7, f"Season {season} should have 7 representative days"
        for m, d in days:
            assert 1 <= m <= 12
            assert 1 <= d <= 31


def test_variable_bounds():
    """Verify default design and operational bounds are positive and non-empty."""
    assert len(DEFAULT_DESIGN_BOUNDS.names) == 5
    assert len(DEFAULT_OPERATIONAL_BOUNDS.names) == 3

    for low, high in zip(DEFAULT_DESIGN_BOUNDS.lb, DEFAULT_DESIGN_BOUNDS.ub):
        assert low <= high

    for low, high in zip(DEFAULT_OPERATIONAL_BOUNDS.lb, DEFAULT_OPERATIONAL_BOUNDS.ub):
        assert low <= high


def test_invalid_bounds_raise():
    """Verify that inverted bounds raise ValueError."""
    with pytest.raises(ValueError):
        VariableBounds(
            names=("x",),
            lb=(100,),
            ub=(10,),
            types=(int,),
        )


def test_to_legacy_dict():
    """Verify that to_legacy_dict exports all expected keys."""
    cfg = get_default_config()
    d = cfg.to_legacy_dict()

    required_keys = [
        "demand_file_path",
        "min_price",
        "max_price",
        "json_file",
        "model",
        "penalty",
        "SEASONS",
        "design",
        "operational",
        "pop_size",
        "num_generations",
        "cxpb",
        "mutpb",
        "indpb",
    ]
    for key in required_keys:
        assert key in d, f"Missing expected legacy key: {key}"
