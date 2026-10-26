"""
Tests for main entrypoint argument parsing and worker scaling detection.
"""

from main import parse_arguments, resolve_worker_count


def test_parse_arguments_default():
    args = parse_arguments([])
    assert args.season == "summer"


def test_parse_arguments_custom():
    args = parse_arguments(["--season", "winter"])
    assert args.season == "winter"


def test_parse_arguments_all():
    args = parse_arguments(["--season", "all"])
    assert args.season == "all"


def test_resolve_worker_count_pbs_env(monkeypatch):
    monkeypatch.setenv("PBS_NCPUS", "64")
    assert resolve_worker_count(112) == 64


def test_resolve_worker_count_local_fallback(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_NCPUS", raising=False)
    monkeypatch.delenv("LOCAL_WORKERS", raising=False)
    monkeypatch.delenv("MPI_LOCALRANKID", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    workers = resolve_worker_count(112)
    assert 1 <= workers <= 4
