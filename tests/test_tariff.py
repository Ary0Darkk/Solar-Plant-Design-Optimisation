"""
Tests for dynamic electricity tariff loading, demand scaling, and caching.
"""

import numpy as np

from solar_opt.config import PipelineConfig
from solar_opt.tariff import (
    compute_tariff_from_demand,
    get_shared_tariff,
)


def test_compute_tariff_from_demand():
    demand = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    # min_price 3.0, max_price 8.0 (in INR/kWh) -> in INR/MWh: 3000 to 8000
    tariff = compute_tariff_from_demand(demand, min_price=3.0, max_price=8.0)
    assert len(tariff) == 3
    assert tariff[0] == 3000.0
    assert tariff[2] == 8000.0
    assert tariff[1] == 5500.0


def test_compute_tariff_constant_demand():
    demand = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    tariff = compute_tariff_from_demand(demand, min_price=3.0, max_price=8.0)
    assert np.all(tariff == 3000.0)


def test_get_shared_tariff_dimensions():
    cfg = PipelineConfig()
    tariff = get_shared_tariff(cfg.tariff)
    assert isinstance(tariff, np.ndarray)
    assert len(tariff) == 8760
    assert tariff.dtype == np.float32
