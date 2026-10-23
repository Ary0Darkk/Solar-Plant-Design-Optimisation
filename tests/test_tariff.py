"""
Tests for electricity tariff computation and caching.
"""

import numpy as np
import pandas as pd

from solar_opt.tariff import compute_tariff_from_demand, get_shared_tariff


def test_compute_tariff_from_demand():
    s = pd.Series([100.0, 200.0, 300.0])
    tar = compute_tariff_from_demand(s, min_price=3.0, max_price=7.0)
    assert len(tar) == 3
    assert np.isclose(tar[0], 3.0)
    assert np.isclose(tar[1], 5.0)
    assert np.isclose(tar[2], 7.0)
    assert tar.dtype == np.float32


def test_compute_tariff_constant_demand():
    s = pd.Series([100.0, 100.0, 100.0])
    tar = compute_tariff_from_demand(s, min_price=3.0, max_price=7.0)
    assert np.all(tar == 3.0)


def test_get_shared_tariff_dimensions():
    tar = get_shared_tariff()
    assert len(tar) == 8760
    assert tar.dtype == np.float32
    assert tar.min() >= 3.0
    assert tar.max() <= 8.0
    assert not tar.flags.writeable  # Read-only memory protection
