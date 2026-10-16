"""
Weather index and seasonal representative days mapping.
"""

from datetime import date
from typing import Dict, List, Optional, Tuple, TypedDict

from ..config.defaults import DEFAULT_SEASONS

REFERENCE_YEAR = 2021


class DayInfo(TypedDict):
    """Metadata describing a single representative optimization day."""

    month: int
    day: int
    local_index: int
    day_index: int


def get_season_days(
    season: str,
    seasons_dict: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    reference_year: int = REFERENCE_YEAR,
) -> List[DayInfo]:
    """
    Convert configured (month, day) pairs into exact calendar day indices.

    Args:
        season: Name of the season ('winter', 'summer', 'monsoon', 'post').
        seasons_dict: Optional custom dictionary of season days.
        reference_year: Non-leap reference year (default 2021) matching 8,760-hour weather data.

    Returns:
        List of DayInfo containing local_index, month, day, and 0-based day_index (0-364).
    """
    lookup = seasons_dict if seasons_dict is not None else DEFAULT_SEASONS
    if season not in lookup:
        raise ValueError(
            f"Unknown season '{season}'. Available seasons: {list(lookup.keys())}"
        )

    season_dates = lookup[season]

    return [
        {
            "month": month,
            "day": day,
            "local_index": local_index,
            "day_index": (
                date(reference_year, month, day).timetuple().tm_yday - 1
            ),
        }
        for local_index, (month, day) in enumerate(season_dates)
    ]
