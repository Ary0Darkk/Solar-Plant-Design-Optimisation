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


def test_load_all_sections_from_yaml(tmp_path):
    """Verify that every field across all 6 YAML configuration sections parses correctly."""
    from optimized_pipeline.config import PROJECT_ROOT, load_pipeline_config

    full_yaml = tmp_path / "comprehensive_config.yaml"
    full_yaml.write_text(
        """
ga:
  pop_size: 42
  num_generations: 15
  tournament_size: 4
  hall_of_fame_size: 2
  cxpb: 0.75
  mutpb: 0.25
  indpb: 0.05
  random_seed: 99
  resume_from_checkpoint: false
  checkpoint_interval: 3
  max_tasks_per_child: 10
  enable_gc_collection: false
  early_stopping_patience: 8

tariff:
  min_price: 1.5
  max_price: 10.5
  demand_file: "custom_demand.xlsx"
  cache_file: "custom_cache.csv"

simulation:
  json_file: "custom_plant.json"
  model_name: "CustomTrough"
  penalty: -9.99e+12
  log_each_simulation: true

design_bounds:
  param_a: [100, 200]
  param_b: [10, 50]

operational_bounds:
  op_x: [5, 15]

seasons:
  winter:
    - [1, 10]
    - [1, 20]
  summer:
    - [6, 1]
    - [6, 15]
""",
        encoding="utf-8",
    )

    cfg = load_pipeline_config(full_yaml)

    # GA assertions
    assert cfg.ga.pop_size == 42
    assert cfg.ga.num_generations == 15
    assert cfg.ga.tournament_size == 4
    assert cfg.ga.hall_of_fame_size == 2
    assert cfg.ga.cxpb == 0.75
    assert cfg.ga.mutpb == 0.25
    assert cfg.ga.indpb == 0.05
    assert cfg.ga.random_seed == 99
    assert cfg.ga.resume_from_checkpoint is False
    assert cfg.ga.checkpoint_interval == 3
    assert cfg.ga.max_tasks_per_child == 10
    assert cfg.ga.enable_gc_collection is False
    assert cfg.ga.early_stopping_patience == 8

    # Tariff assertions
    assert cfg.tariff.min_price == 1.5
    assert cfg.tariff.max_price == 10.5
    assert cfg.tariff.demand_file == PROJECT_ROOT / "custom_demand.xlsx"
    assert cfg.tariff.cache_file == PROJECT_ROOT / "custom_cache.csv"

    # Simulation assertions
    assert cfg.simulation.json_file == PROJECT_ROOT / "custom_plant.json"
    assert cfg.simulation.model_name == "CustomTrough"
    assert cfg.simulation.penalty == -9.99e12
    assert cfg.simulation.log_each_simulation is True

    # Bounds assertions
    assert cfg.design.names == ("param_a", "param_b")
    assert cfg.design.lb == (100, 10)
    assert cfg.design.ub == (200, 50)
    assert cfg.operational.names == ("op_x",)
    assert cfg.operational.lb == (5,)
    assert cfg.operational.ub == (15,)

    # Seasons assertions
    assert set(cfg.seasons.keys()) == {"winter", "summer"}
    assert cfg.seasons["winter"] == [(1, 10), (1, 20)]
    assert cfg.seasons["summer"] == [(6, 1), (6, 15)]


def test_load_empty_or_comment_yaml(tmp_path):
    """Verify that an empty or comment-only YAML file falls back to defaults without error."""
    from optimized_pipeline.config import load_pipeline_config

    empty_yaml = tmp_path / "empty_config.yaml"
    empty_yaml.write_text("# Only comments\n", encoding="utf-8")

    cfg = load_pipeline_config(empty_yaml)
    assert cfg.ga.pop_size == 112
    assert cfg.ga.num_generations == 100
    assert cfg.tariff.min_price == 3.0
    assert cfg.tariff.max_price == 8.0


def test_invalid_yaml_bounds_raise(tmp_path):
    """Verify that a YAML file specifying lower bound > upper bound raises ValueError."""
    from optimized_pipeline.config import load_pipeline_config

    invalid_yaml = tmp_path / "inverted_bounds.yaml"
    invalid_yaml.write_text(
        """
design_bounds:
  Row_Distance: [50, 10]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Lower bound .* > upper bound"):
        load_pipeline_config(invalid_yaml)


def test_load_missing_default_yaml_fallback(tmp_path, monkeypatch):
    """Verify that missing default config.yaml falls back gracefully to built-in defaults."""
    import optimized_pipeline.config as conf_mod

    # Redirect PROJECT_ROOT to empty directory where config.yaml does not exist
    monkeypatch.setattr(conf_mod, "PROJECT_ROOT", tmp_path)

    cfg = conf_mod.get_pipeline_config()
    assert cfg.ga.pop_size == 112
    assert cfg.ga.num_generations == 100
    assert cfg.tariff.min_price == 3.0


def test_root_config_py_exports():
    """Verify root config.py properly exposes YAML helpers and synchronized CONFIG dict."""
    import config

    assert hasattr(config, "load_pipeline_config")
    assert hasattr(config, "save_pipeline_config_to_yaml")
    assert hasattr(config, "get_pipeline_config")
    assert hasattr(config, "PIPELINE_CONFIG")
    assert hasattr(config, "CONFIG")

    assert isinstance(config.CONFIG, dict)
    assert config.CONFIG["pop_size"] == config.PIPELINE_CONFIG.ga.pop_size
    assert config.CONFIG["num_generations"] == config.PIPELINE_CONFIG.ga.num_generations
    assert config.CONFIG["min_price"] == config.PIPELINE_CONFIG.tariff.min_price


def test_cli_loads_custom_config_and_overrides(tmp_path, monkeypatch):
    """Verify that CLI loads --config YAML and applies command-line flag overrides."""
    from optimized_pipeline.cli import main

    custom_yaml = tmp_path / "cli_override.yaml"
    custom_yaml.write_text(
        """
ga:
  pop_size: 32
  num_generations: 12
tariff:
  min_price: 2.2
""",
        encoding="utf-8",
    )

    captured_configs = []

    def fake_run_optimization(**kwargs):
        captured_configs.append(kwargs.get("config"))
        return {}, 0.0, {}

    monkeypatch.setattr("optimized_pipeline.cli.run_optimization", fake_run_optimization)

    # Pass --config along with --pop-size override
    main([
        "--config",
        str(custom_yaml),
        "--pop-size",
        "64",
        "--season",
        "winter",
        "--no-tracking",
    ])

    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    # Loaded from YAML
    assert cfg.ga.num_generations == 12
    assert cfg.tariff.min_price == 2.2
    # Overridden by CLI argument
    assert cfg.ga.pop_size == 64


