"""
Unit tests for configuration models and defaults.
"""

import pytest

from optimized_pipeline.config import (
    DESIGN_BOUNDS,
    OPERATIONAL_BOUNDS,
    SEASONS,
    VariableBounds,
    get_pipeline_config,
)


def test_default_seasons():
    """Verify all 4 seasons exist and have exactly 7 representative days."""
    assert set(SEASONS.keys()) == {"winter", "summer", "monsoon", "post"}
    for season, days in SEASONS.items():
        assert len(days) == 7, f"Season {season} should have 7 representative days"
        for m, d in days:
            assert 1 <= m <= 12
            assert 1 <= d <= 31


def test_variable_bounds():
    """Verify default design and operational bounds are positive and non-empty."""
    assert len(DESIGN_BOUNDS.names) == 5
    assert len(OPERATIONAL_BOUNDS.names) == 3

    for low, high in zip(DESIGN_BOUNDS.lb, DESIGN_BOUNDS.ub):
        assert low <= high

    for low, high in zip(OPERATIONAL_BOUNDS.lb, OPERATIONAL_BOUNDS.ub):
        assert low <= high


def test_invalid_bounds_raise():
    """Verify that inverted bounds raise ValueError."""
    with pytest.raises(ValueError):
        VariableBounds(
            names=("x",),
            lb=(100,),
            ub=(10,),
            types=(int,),
        )


def test_to_legacy_dict():
    """Verify that to_legacy_dict exports all expected keys."""
    cfg = get_pipeline_config()
    d = cfg.to_legacy_dict()

    required_keys = [
        "demand_file_path",
        "min_price",
        "max_price",
        "json_file",
        "model",
        "penalty",
        "SEASONS",
        "design",
        "operational",
        "pop_size",
        "num_generations",
        "cxpb",
        "mutpb",
        "indpb",
    ]
    for key in required_keys:
        assert key in d, f"Missing expected legacy key: {key}"


def test_load_config_yaml_defaults():
    """Verify that get_pipeline_config loads from config.yaml with expected values."""
    cfg = get_pipeline_config()
    assert cfg.ga.pop_size == 112
    assert cfg.ga.num_generations == 100
    assert cfg.tariff.min_price == 3.0
    assert cfg.tariff.max_price == 8.0
    assert len(cfg.design.names) == 5
    assert len(cfg.operational.names) == 3


def test_load_custom_yaml(tmp_path):
    """Verify that load_pipeline_config correctly loads and overrides from a custom YAML file."""
    from optimized_pipeline.config import load_pipeline_config

    custom_yaml = tmp_path / "custom_config.yaml"
    custom_yaml.write_text(
        """
ga:
  pop_size: 28
  num_generations: 10
  mutpb: 0.35
tariff:
  min_price: 2.5
  max_price: 9.5
design_bounds:
  specified_total_aperture: [7000, 15000]
  Row_Distance: [12, 30]
""",
        encoding="utf-8",
    )

    cfg = load_pipeline_config(custom_yaml)
    assert cfg.ga.pop_size == 28
    assert cfg.ga.num_generations == 10
    assert cfg.ga.mutpb == 0.35
    assert cfg.tariff.min_price == 2.5
    assert cfg.tariff.max_price == 9.5
    assert cfg.design.names == ("specified_total_aperture", "Row_Distance")
    assert cfg.design.lb == (7000, 12)
    assert cfg.design.ub == (15000, 30)


def test_load_nonexistent_yaml_raises():
    """Verify that specifying a non-existent YAML file raises FileNotFoundError."""
    from optimized_pipeline.config import load_pipeline_config

    with pytest.raises(FileNotFoundError):
        load_pipeline_config("definitely_non_existent_config_file_12345.yaml")


def test_save_and_reload_yaml(tmp_path):
    """Verify that save_pipeline_config_to_yaml produces a valid, reloadable YAML file."""
    from optimized_pipeline.config import load_pipeline_config, save_pipeline_config_to_yaml

    cfg = get_pipeline_config()
    cfg.ga.pop_size = 56
    cfg.ga.num_generations = 25
    cfg.tariff.min_price = 2.0

    save_path = tmp_path / "exported_config.yaml"
    save_pipeline_config_to_yaml(cfg, save_path)
    assert save_path.exists()

    reloaded = load_pipeline_config(save_path)
    assert reloaded.ga.pop_size == 56
    assert reloaded.ga.num_generations == 25
    assert reloaded.tariff.min_price == 2.0

