"""
Tests for CLI argument parsing and worker scaling.
"""

from solar_opt.cli import parse_arguments, resolve_worker_count


def test_parse_arguments_defaults():
    args = parse_arguments([])
    assert args.season == "summer"
    assert args.experiment == "CSP_Seasonal_GA"
    assert args.generations is None
    assert args.pop_size is None
    assert args.workers is None
    assert not args.no_tracking


def test_parse_arguments_custom():
    args = parse_arguments(
        [
            "--season",
            "winter",
            "--generations",
            "20",
            "--pop-size",
            "14",
            "--workers",
            "4",
            "--no-tracking",
        ]
    )
    assert args.season == "winter"
    assert args.generations == 20
    assert args.pop_size == 14
    assert args.workers == 4
    assert args.no_tracking is True


def test_resolve_worker_count_explicit():
    assert resolve_worker_count(8) == 8


def test_resolve_worker_count_local_fallback(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("JOB_ID", raising=False)
    monkeypatch.setenv("LOCAL_WORKERS", "112")
    # On local machine without scheduler, must safely fall back to <= 4
    workers = resolve_worker_count(None)
    assert workers <= 4
