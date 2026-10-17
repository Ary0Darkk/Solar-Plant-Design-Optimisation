"""
Unit tests for CLI argument parsing, flags, and configuration overrides.
"""

import pytest

from optimized_pipeline.cli import parse_arguments


def test_parse_arguments_defaults():
    """Verify default CLI arguments when no options are provided."""
    args = parse_arguments([])
    assert args.season == "summer"
    assert args.generations is None
    assert args.pop_size is None
    assert args.workers is None
    assert args.max_tasks_per_child is None
    assert args.seed is None
    assert args.min_price is None
    assert args.max_price is None
    assert args.no_tracking is False


def test_parse_arguments_custom_flags():
    """Verify that custom CLI flags correctly override defaults."""
    argv = [
        "--season",
        "winter",
        "--generations",
        "15",
        "--pop-size",
        "28",
        "--workers",
        "6",
        "--max-tasks-per-child",
        "30",
        "--seed",
        "123",
        "--min-price",
        "2.5",
        "--max-price",
        "9.0",
        "--no-tracking",
    ]
    args = parse_arguments(argv)
    assert args.season == "winter"
    assert args.generations == 15
    assert args.pop_size == 28
    assert args.workers == 6
    assert args.max_tasks_per_child == 30
    assert args.seed == 123
    assert args.min_price == 2.5
    assert args.max_price == 9.0
    assert args.no_tracking is True


def test_parse_arguments_invalid_season():
    """Verify that an invalid season choice raises SystemExit (argparse error)."""
    with pytest.raises(SystemExit):
        parse_arguments(["--season", "autumn"])


def test_collect_run_tags_with_job_id(monkeypatch):
    """Verify that collect_run_tags correctly captures job_id from CLI and environment."""
    from optimized_pipeline.cli import collect_run_tags

    # 1. From CLI flag
    args = parse_arguments(["--season", "monsoon", "--job-id", "PBS_998877"])
    tags = collect_run_tags(args, rank=0, size=7, workers=16)
    assert tags["job_id"] == "PBS_998877"
    assert tags["season"] == "monsoon"
    assert tags["mpi_size"] == "7"
    assert tags["workers_per_rank"] == "16"

    # 2. From environment variable fallback
    monkeypatch.setenv("PBS_JOBID", "12345.headnode")
    monkeypatch.setenv("PBS_QUEUE", "standard")
    args_env = parse_arguments(["--season", "winter"])
    tags_env = collect_run_tags(args_env, rank=0, size=7, workers=16)
    assert tags_env["job_id"] == "12345.headnode"
    assert tags_env["queue"] == "standard"
    assert tags_env["scheduler"] == "PBS"


def test_resolve_worker_count_logic(monkeypatch):
    """Verify that worker resolution targets 112 on HPC and falls back to 4 on local workstations."""
    from optimized_pipeline.cli import resolve_worker_count

    # 1. Explicit CLI argument takes precedence
    assert resolve_worker_count(8) == 8
    assert resolve_worker_count(16) == 16

    # 2. On local machine without HPC scheduler, falls back to 4 even if LOCAL_WORKERS=112
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("JOB_ID", raising=False)
    monkeypatch.setenv("LOCAL_WORKERS", "112")
    assert resolve_worker_count(None) == 4

    # 3. On HPC cluster (PBS Pro / Slurm), targets 112 (or LOCAL_WORKERS)
    monkeypatch.setenv("PBS_JOBID", "9999.headnode")
    assert resolve_worker_count(None) == 112

    # 4. Custom LOCAL_WORKERS on HPC cluster
    monkeypatch.setenv("LOCAL_WORKERS", "128")
    assert resolve_worker_count(None) == 128


def test_cli_worker_rank_passes_logger(monkeypatch):
    """Verify that cli.main initializes logger and passes it to worker_event_loop on worker rank."""
    from unittest.mock import MagicMock

    from optimized_pipeline.cli import main

    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 1
    mock_comm.Get_size.return_value = 2
    mock_comm.bcast.return_value = "2026-10-18_12-00-00"

    mock_worker_loop = MagicMock()
    monkeypatch.setattr("optimized_pipeline.cli.get_communicator", lambda: mock_comm)
    monkeypatch.setattr("optimized_pipeline.cli.worker_event_loop", mock_worker_loop)

    main(["--season", "winter"])

    mock_worker_loop.assert_called_once()
    assert mock_worker_loop.call_args.kwargs["rank"] == 1
    assert mock_worker_loop.call_args.kwargs["custom_logger"] is not None

