"""
Tests for seasonal calendar and weather day mapping.
"""

import pytest

from solar_opt.weather import get_season_days


def test_get_season_days_valid():
    winter_days = get_season_days("winter")
    assert len(winter_days) == 7
    first = winter_days[0]
    assert first["month"] == 1
    assert first["day"] == 8
    assert first["local_index"] == 0
    assert first["day_index"] == 7  # Jan 8 is 8th day of year (0-indexed: 7)


def test_get_season_days_all_seasons():
    for season in ["winter", "summer", "monsoon", "post"]:
        days = get_season_days(season)
        assert len(days) == 7
        for d in days:
            assert 0 <= d["day_index"] < 365
            assert 0 <= d["local_index"] < 7


def test_get_season_days_unknown_raises():
    with pytest.raises(ValueError, match="Unknown season"):
        get_season_days("autumn")
