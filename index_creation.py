"""
Representative days indexing module.
Backward compatibility wrapper delegating to solar_opt.simulation.weather.
"""

from solar_opt.simulation.weather import REFERENCE_YEAR, get_season_days

__all__ = ["REFERENCE_YEAR", "get_season_days"]

if __name__ == "__main__":
    for season_name in ["winter", "summer", "monsoon", "post"]:
        print(f"\n--- {season_name.upper()} SEASON DAYS ---")
        for d in get_season_days(season_name):
            print(f"Day {d['local_index']}: Month {d['month']}, Day {d['day']} -> DayIndex {d['day_index']}")
