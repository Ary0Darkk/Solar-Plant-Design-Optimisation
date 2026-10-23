"""
Tests for GA optimization engine, individual evaluation, and elitism loop.
"""

from unittest.mock import MagicMock

import pytest

from solar_opt.config import PipelineConfig
from solar_opt.engine import evaluate_individual_season, run_optimization, worker_event_loop
from solar_opt.parallel import StandaloneComm
from solar_opt.weather import get_season_days


def test_evaluate_individual_season_feasible():
    cfg = PipelineConfig()
    season_days = get_season_days("summer", cfg.seasons)

    # Feasible chromosome: design + 7 * operational
    # Row_Distance=15, W_aperture=5 -> Row_Distance - W_aperture = 10 >= 3.5
    # T_startup=350, T_shutdown=280 -> 350 - 280 = 70 >= 15
    feasible_chrom = [10000, 15, 4, 5, 100] + [5, 350, 280] * 7
    (fit_val,), penalized = evaluate_individual_season(feasible_chrom, season_days, cfg)
    assert not penalized
    assert fit_val >= 0.0


def test_evaluate_individual_season_infeasible():
    cfg = PipelineConfig()
    season_days = get_season_days("summer", cfg.seasons)

    # Infeasible: Row_Distance=7, W_aperture=5 -> 7 - 5 = 2 < 3.5
    infeasible_chrom = [10000, 7, 4, 5, 100] + [5, 350, 280] * 7
    (fit_val,), penalized = evaluate_individual_season(infeasible_chrom, season_days, cfg)
    assert penalized
    assert fit_val == cfg.simulation.penalty


def test_micro_ga_optimization_run():
    cfg = PipelineConfig()
    cfg.ga.pop_size = 4
    cfg.ga.num_generations = 2
    cfg.ga.resume_from_checkpoint = False

    comm = StandaloneComm()
    sol_dict, best_fit, ga_state = run_optimization(
        season="summer",
        config=cfg,
        pool=None,  # sequential execution for fast deterministic test
        comm=comm,
        rank=0,
        mpi_size=1,
        job_id="test_micro_run",
    )

    assert "season" in sol_dict
    assert sol_dict["season"] == "summer"
    assert len(ga_state["pop"]) == 4
    assert len(ga_state["logbook"]) == 2  # Gen 0 and Gen 1


def test_run_optimization_requires_rank_zero():
    cfg = PipelineConfig()
    mock_comm = MagicMock()
    mock_pool = MagicMock()

    with pytest.raises(ValueError, match="0th rank must be the master rank"):
        run_optimization(
            season="summer",
            config=cfg,
            pool=mock_pool,
            comm=mock_comm,
            rank=1,
            mpi_size=2,
        )


def test_worker_event_loop_rejects_rank_zero():
    cfg = PipelineConfig()
    mock_comm = MagicMock()

    with pytest.raises(ValueError, match="Rank 0 is designated as Master Rank"):
        worker_event_loop(
            season="summer",
            comm=mock_comm,
            rank=0,
            config=cfg,
        )
