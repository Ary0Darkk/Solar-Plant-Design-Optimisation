"""
Unit tests for dynamic electricity pricing and tariff calculation.
"""

import numpy as np
import pandas as pd
import pytest

from solar_opt.economic.pricing import (
    compute_dynamic_tariff,
    get_cached_dynamic_prices,
)


def test_dynamic_tariff_bounds():
    """Verify that computed tariffs strictly lie within [min_price, max_price]."""
    demands = pd.Series([1000.0, 2500.0, 5000.0, 7500.0, 10000.0])
    tariffs = compute_dynamic_tariff(demands, min_price=3.0, max_price=8.0)

    assert np.isclose(tariffs.min(), 3.0)
    assert np.isclose(tariffs.max(), 8.0)
    assert tariffs.iloc[0] < tariffs.iloc[1] < tariffs.iloc[2] < tariffs.iloc[3] < tariffs.iloc[4]


def test_cached_dynamic_prices_length():
    """Verify that cached price profile returns exactly 8,760 hourly points."""
    prices = get_cached_dynamic_prices(min_price=3.0, max_price=8.0)
    assert len(prices) == 8760
    assert prices.min() >= 2.99
    assert prices.max() <= 8.01
