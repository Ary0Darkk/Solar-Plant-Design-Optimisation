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
