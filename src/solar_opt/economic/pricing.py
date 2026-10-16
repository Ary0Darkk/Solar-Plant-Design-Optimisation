"""
Dynamic electricity pricing calculations and demand curve mapping.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.defaults import PROJECT_ROOT


def load_formatted_demand_data(
    file_path: Union[str, Path] = "electricity_data/Yearly Demand Profile Haryana-Himanchal.xlsx",
    show_plot: bool = False,
) -> pd.DataFrame:
    """
    Load yearly demand profile, filter for Haryana - 2020, and omit leap day (Feb 29).

    Returns:
        DataFrame with columns ['DateTime', 'Hourly Demand Met (in MW)'] containing 8,760 rows.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Electricity demand file not found at: {path}")

    df = pd.read_excel(path)
    df_state = df[df["State"] == "Haryana - 2020"].copy()
    if df_state.empty:
        raise ValueError(f"No records found for 'Haryana - 2020' in {path}")

    df_state["DateTime"] = pd.to_datetime(
        "2020-" + df_state["Date"], format="%Y-%d-%b %I%p"
    )

    # Exclude Feb 29 leap day to ensure clean 8,760 hours (matching 365-day weather data)
    is_leap_day = (df_state["DateTime"].dt.month == 2) & (df_state["DateTime"].dt.day == 29)
    df_clean = df_state[~is_leap_day].copy()

    formatted_df = df_clean[["DateTime", "Hourly Demand Met (in MW)"]].reset_index(drop=True)

    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(
            formatted_df["DateTime"],
            formatted_df["Hourly Demand Met (in MW)"],
        )
        plt.xlabel("DateTime")
        plt.ylabel("Hourly Demand Met (MW)")
        plt.title("Haryana Hourly Electricity Demand Profile")
        plt.grid(True)
        plt.show()

    return formatted_df


def compute_dynamic_tariff(
    demand_series: pd.Series,
    min_price: float = 3.0,
    max_price: float = 8.0,
) -> pd.Series:
    """
    Linearly scale electricity price (INR/kWh) against grid demand using min-max normalization.
    """
    min_demand = float(demand_series.min())
    max_demand = float(demand_series.max())
    demand_range = max_demand - min_demand
    price_range = max_price - min_price

    if demand_range <= 0:
        return pd.Series(np.full(len(demand_series), min_price), index=demand_series.index)

    scaled_price = min_price + ((demand_series - min_demand) / demand_range) * price_range
    return scaled_price.clip(lower=min_price, upper=max_price)


@lru_cache(maxsize=4)
def get_cached_dynamic_prices(
    min_price: float = 3.0,
    max_price: float = 8.0,
    cache_file: Union[str, Path] = "electricity_data/dynamic_price_data_haryana.csv",
    demand_file: Union[str, Path] = "electricity_data/Yearly Demand Profile Haryana-Himanchal.xlsx",
) -> np.ndarray:
    """
    Retrieve 8,760-hour dynamic price array from cache file or compute from Excel profile.

    Returns:
        1D numpy array of 8,760 hourly electricity tariffs in INR/kWh.
    """
    c_path = Path(cache_file)
    if not c_path.is_absolute():
        c_path = PROJECT_ROOT / c_path

    if c_path.exists():
        try:
            df = pd.read_csv(c_path)
            if "dynamic_price" in df.columns:
                prices = np.asarray(df["dynamic_price"].values, dtype=float)
                if (
                    len(prices) == 8760
                    and np.isclose(prices.min(), min_price, atol=1e-2)
                    and np.isclose(prices.max(), max_price, atol=1e-2)
                ):
                    return prices
        except Exception:
            pass

    # Compute freshly from demand spreadsheet
    formatted = load_formatted_demand_data(file_path=demand_file)
    tariff = compute_dynamic_tariff(
        demand_series=formatted["Hourly Demand Met (in MW)"],
        min_price=min_price,
        max_price=max_price,
    )
    formatted["dynamic_price"] = tariff

    c_path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(c_path, index=False)

    return np.asarray(formatted["dynamic_price"].values, dtype=float)
