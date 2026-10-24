"""
Tests for Genetic Algorithm engine, evaluation loops, and elitism.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from solar_opt.config import PipelineConfig
from solar_opt.engine import (
    evaluate_individual_season,
    run_optimization,
    worker_event_loop,
)
from solar_opt.simulation import DailySimulationSlice


def test_evaluate_individual_season_feasible():
    cfg = PipelineConfig()
    sim = MagicMock()
    # Mock successful daily slice
    slice_mock = DailySimulationSlice(
        day_index=10,
        hourly_energy=np.full(24, 0.5, dtype=np.float32),
        hourly_parasitic=np.full(24, 0.05, dtype=np.float32),
        hourly_heat_loss=np.full(24, 0.1, dtype=np.float32),
    )
    sim.run_daily_simulation.return_value = (slice_mock, False)

    ind = [10000, 15, 4, 5, 100] + [5, 350, 280] * 7
    tariff = np.full(8760, 5000.0, dtype=np.float32)
    season_days = [{"month": 1, "day": 8, "day_index": 7, "local_index": 0}]

    profit, succ = evaluate_individual_season(ind, "winter", cfg, sim, tariff, season_days)
    assert succ is True
    assert profit > 0


def test_evaluate_individual_season_infeasible():
    cfg = PipelineConfig()
    sim = MagicMock()
    sim.run_daily_simulation.return_value = (None, True)

    ind = [10000, 15, 4, 5, 100] + [5, 350, 280] * 7
    tariff = np.full(8760, 5000.0, dtype=np.float32)
    season_days = [{"month": 1, "day": 8, "day_index": 7, "local_index": 0}]

    profit, succ = evaluate_individual_season(ind, "winter", cfg, sim, tariff, season_days)
    assert succ is False
    assert profit == cfg.simulation.penalty


def test_micro_ga_optimization_run(tmp_path):
    cfg = PipelineConfig()
    cfg.ga.num_generations = 1
    cfg.ga.pop_size = 4
    cfg.tracking.enabled = False

    res = run_optimization(
        season="summer",
        config=cfg,
        workers=1,
        checkpoint_dir=tmp_path,
    )
    assert "best_fitness" in res
    assert "season" in res
    assert res["season"] == "summer"


def test_run_optimization_requires_rank_zero():
    comm = MagicMock()
    comm.Get_rank.return_value = 1
    with pytest.raises(RuntimeError):
        run_optimization(season="summer", comm=comm)


def test_worker_event_loop_rejects_rank_zero():
    comm = MagicMock()
    comm.Get_rank.return_value = 0
    with pytest.raises(RuntimeError):
        worker_event_loop(comm)
