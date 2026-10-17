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
        "--season", "winter",
        "--generations", "15",
        "--pop-size", "28",
        "--workers", "6",
        "--max-tasks-per-child", "30",
        "--seed", "123",
        "--min-price", "2.5",
        "--max-price", "9.0",
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
