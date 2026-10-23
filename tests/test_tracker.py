"""
Tests for MLflow tracking, artifact plotting, and export formats.
"""

from pathlib import Path
from unittest.mock import MagicMock

import mlflow

from clean_mlflow_runs import clean_runs
from solar_opt.config import PipelineConfig
from solar_opt.tracker import (
    init_tracking,
    log_run_params,
    log_step_metrics,
    plot_and_export,
    sanitize_key,
    setup_logger,
)


def test_sanitize_key():
    assert sanitize_key("foo.bar/baz-1") == "foo.bar/baz-1"
    assert sanitize_key("param (with spaces)") == "param__with_spaces_"


def test_setup_logger(tmp_path: Path):
    log = setup_logger(rank=0, timestamp="2026-10-23_test", log_dir=tmp_path)
    log.info("Test log line")
    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8")
    assert "Test log line" in content


def test_init_tracking_disabled():
    assert init_tracking(enabled=False) is False


def test_init_tracking_local_sqlite(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("solar_opt.tracker.PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    sqlite_db = tmp_path / "results" / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{sqlite_db.as_posix()}")

    success = init_tracking(enabled=True, experiment_name="Test_Local_Exp")
    assert success is True
    uri = mlflow.get_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert sqlite_db.exists()


def test_log_run_params_and_step_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("solar_opt.tracker.PROJECT_ROOT", tmp_path)
    init_tracking(enabled=True, experiment_name="Test_Metrics_Exp")

    cfg = PipelineConfig()
    with mlflow.start_run():
        log_run_params(config=cfg, season="winter", workers=4, job_id="test_job_123")

        # Fake population and Hall of Fame
        pop = [MagicMock()]
        pop[0].fitness.values = (1200.0,)
        pop[0].__getitem__ = lambda self, idx: 10 + idx
        hof = [pop[0]]

        record = {"avg": 1000.0, "std": 50.0, "min": 800.0, "max": 1200.0}
        season_days = [{"month": 1, "day": 8, "local_index": 0, "day_index": 7}]

        log_step_metrics(
            pop=pop,
            hof=hof,
            record=record,
            gen=0,
            design_names=cfg.design.names,
            operational_names=cfg.operational.names,
            season_days=season_days,
            sim_successful=1,
            sim_penalized=0,
            total_successful=1,
            total_penalized=0,
        )

        active = mlflow.active_run()
        assert active is not None


def test_plot_and_export_generation_with_mlflow(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("solar_opt.tracker.PROJECT_ROOT", tmp_path)
    init_tracking(enabled=True, experiment_name="Test_Artifacts_Exp")

    best_ind = [10000, 15, 4, 5, 100] + [5, 350, 280] * 7
    design_names = ["specified_total_aperture", "Row_Distance", "ColperSCA", "W_aperture", "L_SCA"]
    op_names = ["m_dot", "T_startup", "T_shutdown"]
    season_days = [{"month": 3, "day": 9, "local_index": i, "day_index": 67 + i} for i in range(7)]
    logbook = [{"gen": 0, "nevals": 10, "avg": 100.0, "std": 5.0, "min": 50.0, "max": 150.0}]

    with mlflow.start_run():
        sol = plot_and_export(
            season="summer",
            timestamp="test_export",
            best_ind=best_ind,
            best_fitness=150.0,
            design_names=design_names,
            operational_names=op_names,
            season_days=season_days,
            logbook=logbook,
            gens=[0],
            max_fitness=[150.0],
            avg_fitness=[100.0],
            job_id="test_job",
        )
        assert sol["season"] == "summer"
        assert sol["best_fitness"] == 150.0
        assert sol["job_id"] == "test_job"


def test_clean_mlflow_runs_utility(tmp_path: Path):
    db_file = tmp_path / "test_clean_mlflow.db"
    uri = f"sqlite:///{db_file.as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Test_Clean_Exp")

    # Create a dummy run
    with mlflow.start_run():
        mlflow.log_param("test_param", 123)

    # Clean runs using utility
    res = clean_runs(tracking_uri=uri, experiment_name="Test_Clean_Exp")
    assert res == 0
