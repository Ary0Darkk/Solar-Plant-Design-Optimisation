"""
Seasonal Calendar Mapping & Representative Day Calculations.
"""

from typing import Any, Dict, List

DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def month_day_to_day_index(month: int, day: int) -> int:
    """Convert 1-based month (1-12) and 1-based day (1-31) to 0-based day of year (0-364)."""
    return DAYS_BEFORE_MONTH[month - 1] + (day - 1)


def get_season_days(season: str, seasons_cfg: Dict[str, Any]) -> List[Dict[str, int]]:
    """
    Return list of 7 metadata dicts for the specified season:
    [{"month": int, "day": int, "day_index": int, "local_index": int}, ...]
    """
    season_key = season.lower()
    if season_key not in seasons_cfg:
        raise ValueError(
            f"Unknown season '{season}'. Available seasons: {list(seasons_cfg.keys())}"
        )

    meta = seasons_cfg[season_key]
    start_month = meta["start_month"]
    start_day = meta["start_day"]
    num_days = meta.get("num_days", 7)

    days = []
    for i in range(num_days):
        m = start_month
        d = start_day + i
        day_idx = month_day_to_day_index(m, d)
        days.append(
            {
                "month": m,
                "day": d,
                "day_index": day_idx,
                "local_index": i,
            }
        )

    return days
