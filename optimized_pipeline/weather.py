"""
Weather indexing and representative days calendar mapping.
"""

from datetime import date
from typing import Dict, List, Optional, Tuple, TypedDict

from .config import SEASONS

REFERENCE_YEAR = 2021  # Non-leap year matching 8,760-hour weather datasets


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
    Convert (month, day) pairs into calendar day indices (0 to 364).

    Args:
        season: Name of the season ('winter', 'summer', 'monsoon', 'post').
        seasons_dict: Optional custom season calendar dictionary.
        reference_year: Reference year without Feb 29 (default: 2021).

    Returns:
        List of DayInfo dicts.
    """
    lookup = seasons_dict if seasons_dict is not None else SEASONS
    if season not in lookup:
        raise ValueError(
            f"Unknown season '{season}'. Available seasons: {list(lookup.keys())}"
        )

    dates = lookup[season]
    return [
        {
            "month": m,
            "day": d,
            "local_index": idx,
            "day_index": date(reference_year, m, d).timetuple().tm_yday - 1,
        }
        for idx, (m, d) in enumerate(dates)
    ]
