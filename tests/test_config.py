"""
Tests for configuration loading, bounds, and YAML serialization.
"""

from pathlib import Path

import pytest

from solar_opt.config import (
    PipelineConfig,
    VariableBounds,
    load_pipeline_config,
    save_pipeline_config_to_yaml,
)


def test_default_config_initialization():
    cfg = PipelineConfig()
    assert cfg.ga.pop_size == 112
    assert cfg.ga.num_generations == 100
    assert cfg.ga.cxpb == 0.8
    assert cfg.ga.mutpb == 0.2
    assert cfg.tariff.min_price == 3.0
    assert cfg.tariff.max_price == 8.0
    assert len(cfg.design.names) == 5
    assert len(cfg.operational.names) == 3
    assert set(cfg.seasons.keys()) == {"winter", "summer", "monsoon", "post"}


def test_variable_bounds_validation():
    with pytest.raises(ValueError):
        VariableBounds(
            names=("a", "b"),
            lb=(1,),
            ub=(10, 20),
            types=(int, int),
        )


def test_yaml_config_save_and_load(tmp_path: Path):
    cfg = PipelineConfig()
    cfg.ga.pop_size = 50
    cfg.ga.num_generations = 10
    cfg.tariff.min_price = 4.0

    yaml_file = tmp_path / "test_config.yaml"
    save_pipeline_config_to_yaml(cfg, yaml_file)

    assert yaml_file.exists()
    loaded_cfg = load_pipeline_config(str(yaml_file))
    assert loaded_cfg.ga.pop_size == 50
    assert loaded_cfg.ga.num_generations == 10
    assert loaded_cfg.tariff.min_price == 4.0
    assert loaded_cfg.bounds.design["specified_total_aperture"] == (8000.0, 12000.0)


def test_legacy_dict_conversion():
    cfg = PipelineConfig()
    d = cfg.to_dict()
    assert "min_price" in d
    assert "pop_size" in d


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pipeline_config("non_existent_file_path.yaml")


def test_config_validation_and_bounds_helpers():
    cfg = PipelineConfig()
    cfg.validate()

    assert cfg.design.dimension == 5
    assert len(cfg.design) == 5
    assert "Row_Distance" in cfg.design.to_dict()

    cfg.ga.pop_size = -1
    with pytest.raises(ValueError, match="pop_size"):
        cfg.validate()

    cfg.ga.pop_size = 112
    cfg.tariff.min_price = 10.0
    cfg.tariff.max_price = 5.0
    with pytest.raises(ValueError, match="min_price"):
        cfg.validate()

