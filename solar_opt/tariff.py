"""
Read-Only Cached Electricity Tariff Calculation.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_GLOBAL_TARIFF_CACHE: Optional[np.ndarray] = None


def compute_tariff_from_demand(
    demand_series: np.ndarray,
    min_price: float = 3.0,
    max_price: float = 8.0,
) -> np.ndarray:
    """
    Scale electrical demand to tariff in INR/MWh.
    Tariff (INR/kWh) is scaled to INR/MWh (x 1000).
    """
    d_min = float(np.min(demand_series))
    d_max = float(np.max(demand_series))

    if abs(d_max - d_min) < 1e-6:
        scaled = np.full_like(demand_series, min_price)
    else:
        scaled = min_price + (demand_series - d_min) / (d_max - d_min) * (max_price - min_price)

    # Convert INR/kWh -> INR/MWh
    return (scaled * 1000.0).astype(np.float32)


def load_yearly_tariff(
    source_file: Optional[str] = None,
    demand_file: Optional[str] = None,
    min_price: float = 3.0,
    max_price: float = 8.0,
) -> np.ndarray:
    """Load or generate 8,760-hour tariff schedule in INR/MWh."""
    # 1. Try precomputed dynamic price CSV
    if source_file:
        p = Path(source_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            try:
                df = pd.read_csv(p)
                for col in ["price", "tariff", "Price", "Tariff", "rate"]:
                    if col in df.columns:
                        vals = df[col].to_numpy(dtype=np.float32)
                        if len(vals) >= 8760:
                            # If values are in INR/kWh (~3-8), scale to INR/MWh
                            if np.nanmean(vals[:8760]) < 100.0:
                                vals = vals * 1000.0
                            return vals[:8760].astype(np.float32)
            except Exception:
                pass

    # 2. Try Yearly Demand Profile Excel
    if demand_file:
        p = Path(demand_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            try:
                df = pd.read_excel(p)
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        vals = df[col].dropna().to_numpy(dtype=np.float32)
                        if len(vals) >= 8760:
                            return compute_tariff_from_demand(vals[:8760], min_price, max_price)
            except Exception:
                pass

    # 3. Fallback: synthetic sinusoidal 8,760 profile
    hours = np.arange(8760, dtype=np.float32)
    daily_cycle = np.sin((hours % 24 - 6) * np.pi / 12.0)
    synthetic = min_price + (daily_cycle + 1.0) / 2.0 * (max_price - min_price)
    return (synthetic * 1000.0).astype(np.float32)


def get_shared_tariff(tariff_config: Optional[Any] = None) -> np.ndarray:
    """Thread-safe global singleton getter for tariff array."""
    global _GLOBAL_TARIFF_CACHE
    if _GLOBAL_TARIFF_CACHE is None:
        source_file = getattr(tariff_config, "source_file", None)
        demand_file = getattr(tariff_config, "demand_file", None)
        min_p = getattr(tariff_config, "min_price", 3.0)
        max_p = getattr(tariff_config, "max_price", 8.0)
        _GLOBAL_TARIFF_CACHE = load_yearly_tariff(source_file, demand_file, min_p, max_p)
    return _GLOBAL_TARIFF_CACHE
