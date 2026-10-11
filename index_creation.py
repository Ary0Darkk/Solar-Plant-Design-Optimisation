from datetime import date
from config import CONFIG

REFERENCE_YEAR = 2020


def get_season_days(season):
    """
    Convert configured (month, day) dates into information
    required by the seasonal GA.

    Returns:
        A list containing:

        local_index:
            Position of the representative day within the season.
            Used to select the 3 operational genes.

        month:
            Calendar month.

        day:
            Calendar day.

        day_index:
            Zero-based day-of-year.
            Used by objective_function() to select the
            corresponding 24-hour electricity-price block.
    """

    season_dates = CONFIG["SEASONS"][season]

    return [
        {
            "month": month,
            "day": day,
            "local_index": local_index,
            "day_index": (
                date(
                    REFERENCE_YEAR,
                    month,
                    day,
                )
                .timetuple()
                .tm_yday
                - 1
            ),
        }
        for local_index, (month, day) in enumerate(season_dates)
    ]


if __name__ == "__main__":
    for season_name in CONFIG["SEASONS"]:
        print(f"\n--- {season_name.upper()} SEASON DAYS ---")
        for d in get_season_days(season_name):
            print(
                f"  Day {d['local_index'] + 1}: Month {d['month']:02d}, "
                f"Day {d['day']:02d} -> Year Day Index {d['day_index']}"
            )
