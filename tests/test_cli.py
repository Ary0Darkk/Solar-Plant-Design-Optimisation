"""
Tests for CLI argument parsing and worker scaling detection.
"""

from solar_opt.cli import parse_arguments, resolve_worker_count


def test_parse_arguments_defaults():
    args = parse_arguments([])
    assert args.season == "summer"
    assert args.experiment == "CSP_Seasonal_GA"
    assert args.no_tracking is False
    assert args.simulate is False
    assert args.day == 1


def test_parse_arguments_custom():
    args = parse_arguments(
        [
            "--season",
            "winter",
            "--generations",
            "50",
            "--pop-size",
            "60",
            "--workers",
            "8",
            "--no-tracking",
        ]
    )
    assert args.season == "winter"
    assert args.generations == 50
    assert args.pop_size == 60
    assert args.workers == 8
    assert args.no_tracking is True


def test_resolve_worker_count_explicit():
    assert resolve_worker_count(2) == 2


def test_resolve_worker_count_local_fallback(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    workers = resolve_worker_count(None)
    assert 1 <= workers <= 4
