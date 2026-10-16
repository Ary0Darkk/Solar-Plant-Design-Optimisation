"""
Unit tests for dynamic electricity pricing and tariff calculation.
"""

import numpy as np
import pandas as pd
import pytest

from optimized_pipeline.tariff import (
    compute_tariff,
    get_shared_tariff,
)


def test_dynamic_tariff_bounds():
    """Verify that computed tariffs strictly lie within [min_price, max_price]."""
    demands = pd.Series([1000.0, 2500.0, 5000.0, 7500.0, 10000.0])
    tariffs = compute_tariff(demands, min_price=3.0, max_price=8.0)

    assert np.isclose(tariffs.min(), 3.0)
    assert np.isclose(tariffs.max(), 8.0)
    assert tariffs[0] < tariffs[1] < tariffs[2] < tariffs[3] < tariffs[4]


def test_shared_tariff_length_and_type():
    """Verify that shared tariff returns an 8,760-element float32 array."""
    prices = get_shared_tariff()
    assert len(prices) == 8760
    assert prices.dtype == np.float32
    assert prices.min() >= 2.99
    assert prices.max() <= 8.01
