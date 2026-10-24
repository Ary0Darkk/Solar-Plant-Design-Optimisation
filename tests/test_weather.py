"""
Tests for seasonal calendar mapping and representative day generation.
"""

import pytest

from solar_opt.config import PipelineConfig
from solar_opt.weather import get_season_days, month_day_to_day_index


def test_month_day_to_day_index():
    assert month_day_to_day_index(1, 1) == 0
    assert month_day_to_day_index(1, 8) == 7
    assert month_day_to_day_index(5, 9) == 120 + 8  # 128 (0-indexed: 129th day)
    assert month_day_to_day_index(12, 31) == 364


def test_get_season_days_valid():
    cfg = PipelineConfig()
    winter_days = get_season_days("winter", cfg.seasons)
    assert len(winter_days) == 7
    assert winter_days[0]["month"] == 1
    assert winter_days[0]["day"] == 8
    assert winter_days[0]["day_index"] == 7
    assert winter_days[6]["day"] == 14
    assert winter_days[6]["day_index"] == 13


def test_get_season_days_all_seasons():
    cfg = PipelineConfig()
    for s in ["winter", "summer", "monsoon", "post"]:
        days = get_season_days(s, cfg.seasons)
        assert len(days) == 7


def test_get_season_days_unknown_raises():
    cfg = PipelineConfig()
    with pytest.raises(ValueError):
        get_season_days("autumn", cfg.seasons)
