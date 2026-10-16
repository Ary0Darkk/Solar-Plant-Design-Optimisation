"""
Pre-computed, memory-efficient float32 electricity tariff array and demand processor.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .config import TariffConfig, PROJECT_ROOT

_CACHED_TARIFF_ARRAY: Optional[np.ndarray] = None


def load_demand_data(excel_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load annual hourly demand profile, filter for Haryana 2020, and exclude leap day.

    Returns:
        DataFrame containing exactly 8,760 hourly rows.
    """
    path = Path(excel_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Electricity demand file not found: {path}")

    df = pd.read_excel(path)
    df_state = df[df["State"] == "Haryana - 2020"].copy()
    if df_state.empty:
        raise ValueError(f"State 'Haryana - 2020' not found in {path}")

    df_state["DateTime"] = pd.to_datetime("2020-" + df_state["Date"], format="%Y-%d-%b %I%p")
    # Omit Feb 29 to align with 8,760-hour weather datasets
    is_leap = (df_state["DateTime"].dt.month == 2) & (df_state["DateTime"].dt.day == 29)
    clean_df = df_state[~is_leap][["DateTime", "Hourly Demand Met (in MW)"]].reset_index(drop=True)

    if len(clean_df) != 8760:
        raise ValueError(f"Expected 8,760 rows, got {len(clean_df)}")

    return clean_df


def compute_tariff_from_demand(
    demand_series: pd.Series,
    min_price: float = 3.0,
    max_price: float = 8.0,
) -> np.ndarray:
    """Linearly map electricity demand to tariff range [min_price, max_price] as float32."""
    d_min = float(demand_series.min())
    d_max = float(demand_series.max())
    d_range = d_max - d_min
    p_range = max_price - min_price

    if d_range <= 0:
        return np.full(len(demand_series), min_price, dtype=np.float32)

    scaled = min_price + ((demand_series.values - d_min) / d_range) * p_range
    clipped = np.clip(scaled, min_price, max_price)
    return clipped.astype(np.float32)


# Alias for compatibility
compute_tariff = compute_tariff_from_demand



def get_shared_tariff(
    config: Optional[TariffConfig] = None,
    force_reload: bool = False,
) -> np.ndarray:
    """
    Retrieve read-only 8,760-hour tariff array as float32.
    Pre-cached in memory to prevent IPC copying or re-reading across processes.
    """
    global _CACHED_TARIFF_ARRAY
    if _CACHED_TARIFF_ARRAY is not None and not force_reload:
        return _CACHED_TARIFF_ARRAY

    cfg = config or TariffConfig()
    cache_path = Path(cfg.cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if cache_path.exists() and not force_reload:
        try:
            df = pd.read_csv(cache_path)
            if "dynamic_price" in df.columns:
                arr = df["dynamic_price"].values.astype(np.float32)
                if (
                    len(arr) == 8760
                    and np.isclose(arr.min(), cfg.min_price, atol=1e-2)
                    and np.isclose(arr.max(), cfg.max_price, atol=1e-2)
                ):
                    arr.setflags(write=False)  # Read-only memory safety
                    _CACHED_TARIFF_ARRAY = arr
                    return _CACHED_TARIFF_ARRAY
        except Exception:
            pass

    # Compute from Excel
    df_demand = load_demand_data(cfg.demand_file)
    tariff_arr = compute_tariff_from_demand(
        demand_series=df_demand["Hourly Demand Met (in MW)"],
        min_price=cfg.min_price,
        max_price=cfg.max_price,
    )

    df_demand["dynamic_price"] = tariff_arr
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_demand.to_csv(cache_path, index=False)

    tariff_arr.setflags(write=False)
    _CACHED_TARIFF_ARRAY = tariff_arr
    return _CACHED_TARIFF_ARRAY
