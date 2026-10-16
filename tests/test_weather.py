"""
Unit tests for weather and representative day indexing.
"""

import pytest

from optimized_pipeline.weather import get_season_days


def test_get_season_days_count():
    """Verify that each season returns 7 representative day objects."""
    for s in ["winter", "summer", "monsoon", "post"]:
        days = get_season_days(s)
        assert len(days) == 7
        for idx, d in enumerate(days):
            assert d["local_index"] == idx
            assert 0 <= d["day_index"] <= 364
            assert 1 <= d["month"] <= 12
            assert 1 <= d["day"] <= 31


def test_invalid_season():
    """Verify that requesting an unknown season raises ValueError."""
    with pytest.raises(ValueError):
        get_season_days("spring")
