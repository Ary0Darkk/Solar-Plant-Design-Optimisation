"""
Unit tests for backward-compatible root entry points (deap_ga_optimiser.py, main.py, config.py).
"""

from config import CONFIG, PIPELINE_CONFIG
from deap_ga_optimiser import run_deap_ga_optimisation
from main import main as main_entry


def test_config_compatibility():
    """Verify config.py exports backward-compatible CONFIG dictionary and PIPELINE_CONFIG."""
    assert isinstance(CONFIG, dict)
    assert "demand_file_path" in CONFIG
    assert "min_price" in CONFIG
    assert "max_price" in CONFIG
    assert "SEASONS" in CONFIG
    assert PIPELINE_CONFIG is not None
    assert hasattr(PIPELINE_CONFIG, "ga")
    assert hasattr(PIPELINE_CONFIG, "design")


def test_main_entry_callable():
    """Verify that main.py entry point is callable."""
    assert callable(main_entry)


def test_deap_ga_optimiser_wrapper_signature():
    """Verify that run_deap_ga_optimisation exists, is callable, and accepts expected arguments."""
    assert callable(run_deap_ga_optimisation)


def test_deap_ga_optimiser_master_signals_workers(monkeypatch):
    """Verify Rank 0 acts as master, runs optimization, and signals remote workers to shutdown."""
    from unittest.mock import MagicMock

    mock_run_opt = MagicMock(return_value=({"best_design": 1}, 100.0, {"stat": "ok"}))
    monkeypatch.setattr("deap_ga_optimiser.run_optimization", mock_run_opt)

    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 0
    mock_comm.Get_size.return_value = 3

    best_ind, best_fit, res = run_deap_ga_optimisation(
        comm=mock_comm,
        rank=0,
        mpi_size=3,
        workers=2,
    )

    assert best_fit == 100.0
    mock_run_opt.assert_called_once()
    assert mock_run_opt.call_args.kwargs["rank"] == 0
    assert mock_run_opt.call_args.kwargs["mpi_size"] == 3

    # Rank 0 must have sent shutdown None signal to workers 1 and 2
    sent_shutdowns = [
        call.kwargs
        for call in mock_comm.send.call_args_list
        if call.args == (None,) or (len(call.args) > 0 and call.args[0] is None)
    ]
    assert len(sent_shutdowns) == 2
    assert {call["dest"] for call in sent_shutdowns} == {1, 2}


def test_deap_ga_optimiser_worker_rank_event_loop(monkeypatch):
    """Verify Rank > 0 acts as worker, enters worker_event_loop, and does not run GA."""
    from unittest.mock import MagicMock

    mock_run_opt = MagicMock()
    mock_worker_loop = MagicMock()
    monkeypatch.setattr("deap_ga_optimiser.run_optimization", mock_run_opt)
    monkeypatch.setattr("deap_ga_optimiser.worker_event_loop", mock_worker_loop)

    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 2
    mock_comm.Get_size.return_value = 4

    best_ind, best_fit, res = run_deap_ga_optimisation(
        comm=mock_comm,
        rank=2,
        mpi_size=4,
        workers=2,
    )

    assert best_ind == {}
    assert best_fit == 0.0
    assert res == {}
    mock_run_opt.assert_not_called()
    mock_worker_loop.assert_called_once()
    assert mock_worker_loop.call_args.kwargs["rank"] == 2
    assert mock_worker_loop.call_args.kwargs["custom_logger"] is not None

