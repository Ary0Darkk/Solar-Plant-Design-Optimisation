import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CONFIG

# Load the file - skip the first two rows of metadata as seen in your image


def plot_dni_consecutive(selections):
    """
    Plots DNI data consecutively for a list of (month, day) tuples.
    Example selections: [(1, 1), (1, 2), (2, 15)]
    """
    print(selections)
    all_selected_data = []
    df = pd.read_csv(
        "simulation/28.428_77.159_28.428_77.159_himawari_60_2020.csv", skiprows=2
    )

    for month, day in selections:
        day_data = df[(df["Month"] == month) & (df["Day"] == day)].copy()
        if not day_data.empty:
            # Create a nice string label for the X-axis: "Jan 01 12:00"
            day_data["Timestamp_Str"] = pd.to_datetime(
                day_data[["Year", "Month", "Day", "Hour"]]
            ).dt.strftime("%b %d %H:00")
            all_selected_data.append(day_data)

    if not all_selected_data:
        print("No data found for the selected days.")
        return

    # Combine the data
    plot_df = pd.concat(all_selected_data)

    plt.figure(figsize=(15, 6))

    # Plotting using the string labels on the X-axis removes the time gaps
    plt.plot(
        plot_df["Timestamp_Str"],
        plot_df["Clearsky DNI"],
        color="orange",
        marker="o",
        markersize=2,
    )

    # Clean up the X-axis so it doesn't show every single hour label
    # This keeps every 6th label (e.g., 00:00, 06:00, 12:00, 18:00)
    ax = plt.gca()
    n = 24

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)

    plt.title("Weather data: specific month & day", fontsize=14)
    plt.ylabel("Clearsky DNI (W/m²)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# To plot Jan 1st, Jan 2nd, and Feb 1st in a sequence:


if __name__ == "__main__":
    # Example Usage:
    # my_dates = CONFIG["SEASONS"]["summer"]
    my_dates = [
        day for season_days in CONFIG["SEASONS"].values() for day in season_days
    ]
    # print(my_dates)
    plot_dni_consecutive(my_dates)
