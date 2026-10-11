import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import CONFIG


def format_data(file_path: str, show_demand_plot: bool):
    df = pd.read_excel(Path(file_path))

    df_2020 = df[df["State"] == "Haryana - 2020"].copy()

    df_2020["DateTime"] = pd.to_datetime(
        "2020-" + df_2020["Date"], format="%Y-%d-%b %I%p"
    )

    # Create a boolean mask where Month is 2 (February) AND Day is 29.
    is_leap_day = (df_2020["DateTime"].dt.month == 2) & (
        df_2020["DateTime"].dt.day == 29
    )

    # Use the NOT operator (~) to keep all rows EXCEPT those that are the leap day.
    df_2020_no_leap = df_2020[~is_leap_day].copy()

    formatted_demand_df = df_2020_no_leap[
        ["DateTime", "Hourly Demand Met (in MW)"]
    ].reset_index(drop=True)

    if show_demand_plot is True:
        plt.figure(figsize=(12, 6))
        plt.plot(
            formatted_demand_df["DateTime"],
            formatted_demand_df["Hourly Demand Met (in MW)"],
        )
        plt.xlabel("DateTime")
        plt.ylabel("Hourly Demand Met (MW)")
        plt.title("Hourly Demand Met Over Time")
        plt.grid(True)
        plt.show()

    # print(max(formatted_demand_df['Hourly Demand Met (in MW)']))
    # print(min(formatted_demand_df['Hourly Demand Met (in MW)']))

    return formatted_demand_df


def calc_dynamic_price(
    demand_data: pd.DataFrame,
    min_price: float = 7,
    max_price: float = 15,
):
    # define min and max of demand data
    min_demand = demand_data.min()
    max_demand = demand_data.max()

    # calc price and demand range
    price_range = max_price - min_price
    demand_range = max_demand - min_demand

    # maps price range to demand range (using min-max scaling)
    dynamic_price = (
        min_price + ((demand_data - min_demand) / demand_range) * price_range
    )

    dynamic_price_df = dynamic_price.clip(lower=min_price, upper=max_price)

    return dynamic_price_df


def get_dynamic_price() -> pd.DataFrame:
    formatted_data = format_data(
        file_path=CONFIG["demand_file_path"],
        show_demand_plot=CONFIG["show_demand_plot"],
    )

    demand_data = formatted_data["Hourly Demand Met (in MW)"]
    dynamic_price_data = calc_dynamic_price(demand_data=demand_data)

    dynamic_price_data_df = dynamic_price_data.to_frame(name="dynamic_price")

    final_dataset = pd.concat([formatted_data, dynamic_price_data_df], axis=1)

    if CONFIG["show_price_plot"] is True:
        plt.figure(figsize=(12, 6))
        plt.plot(
            final_dataset["DateTime"],
            final_dataset["dynamic_price"],
        )
        plt.xlabel("DateTime")
        plt.ylabel("Dynamic Price")
        plt.title("Dynamic Price Over Time")
        plt.grid(True)
        plt.show()

    print(dynamic_price_data.head())
    print(type(dynamic_price_data))
    print(len(dynamic_price_data))

    print(len(formatted_data))
    print(type(formatted_data))
    print(type(dynamic_price_data_df))
    print(len(dynamic_price_data_df))

    print(final_dataset.head(10))
    print(final_dataset.tail(10))
    print(type(final_dataset))

    return final_dataset


def save_data():
    final_dataset = get_dynamic_price()

    final_dataset.to_csv(Path("electricity_data/dynamic_price_data_haryana.csv"))


if __name__ == "__main__":
    save_data()
