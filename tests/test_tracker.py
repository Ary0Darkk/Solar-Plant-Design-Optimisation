"""
Unit tests for tracker, logging, artifact exports (Excel, PDF, PNG, Markdown), and MLflow integration.
"""

from pathlib import Path

from optimized_pipeline.config import DESIGN_BOUNDS, OPERATIONAL_BOUNDS, SEASONS
from optimized_pipeline.tracker import plot_and_export, setup_logger
from optimized_pipeline.weather import get_season_days


def test_setup_logger(tmp_path: Path):
    """Verify setup_logger creates a valid logger and initializes the directory."""
    log_dir = tmp_path / "test_logs"
    logger = setup_logger(rank=0, timestamp="2026-10-17_00-00-00", log_dir=log_dir)
    assert logger is not None
    assert log_dir.exists()
    assert (log_dir / "opt_2026-10-17_00-00-00_rank_0.log").exists()


def test_plot_and_export_artifacts(tmp_path: Path, monkeypatch):
    """Verify that plot_and_export creates PNG, PDF, Excel, JSON, CSV, and Markdown report."""
    # Temporarily redirect PROJECT_ROOT inside tracker module to tmp_path
    monkeypatch.setattr("optimized_pipeline.tracker.PROJECT_ROOT", tmp_path)

    season = "summer"
    season_days = get_season_days(season, SEASONS)
    num_days = len(season_days)

    # Synthetic individual: 5 design genes + 7 * 3 operational genes = 26 genes
    best_ind = [9000, 16, 4, 6, 120] + [5, 330, 270] * num_days
    best_fitness = 150000.0
    timestamp = "20261017_120000"

    logbook = [
        {"gen": 0, "nevals": 10, "avg": 50000.0, "max": 100000.0},
        {"gen": 1, "nevals": 10, "avg": 80000.0, "max": 150000.0},
    ]
    gens = [0, 1]
    max_fitness = [100000.0, 150000.0]
    avg_fitness = [50000.0, 80000.0]

    sol = plot_and_export(
        season=season,
        best_ind=best_ind,
        best_fitness=best_fitness,
        timestamp=timestamp,
        design_names=DESIGN_BOUNDS.names,
        operational_names=OPERATIONAL_BOUNDS.names,
        season_days=season_days,
        logbook=logbook,
        gens=gens,
        max_fitness=max_fitness,
        avg_fitness=avg_fitness,
        job_id="TEST_JOB_999",
    )

    results_dir = tmp_path / "results" / "GA_results"
    plots_dir = tmp_path / "plots" / "GA_plots"

    # 1. Check generated plots (PNG & vector PDF)
    assert (plots_dir / f"{season}_convergence_{timestamp}.png").exists()
    assert (plots_dir / f"{season}_convergence_{timestamp}.pdf").exists()

    # 2. Check CSV & JSON
    assert (results_dir / f"{season}_{timestamp}.csv").exists()
    assert (results_dir / f"best_solution_{season}.json").exists()
    assert (results_dir / f"{season}_logbook.csv").exists()

    # 3. Check Multi-sheet Excel workbook
    excel_file = results_dir / f"{season}_summary_{timestamp}.xlsx"
    assert excel_file.exists()

    # 4. Check Markdown executive report
    report_file = results_dir / f"{season}_executive_report_{timestamp}.md"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Optimal Plant Design Parameters" in content
    assert "Daily Operational Schedule" in content
    assert "150,000.00 INR" in content
    assert "TEST_JOB_999" in content

    # 5. Check returned dictionary
    assert sol["season"] == "summer"
    assert sol["job_id"] == "TEST_JOB_999"
    assert sol["best_fitness"] == 150000.0
    assert sol["design_specified_total_aperture"] == 9000
    assert sol["design_Row_Distance"] == 16


def test_init_tracking_local_fallback(tmp_path: Path, monkeypatch):
    """Verify that init_tracking falls back to local SQLite database when remote token is absent."""
    import mlflow

    from optimized_pipeline.tracker import init_tracking

    monkeypatch.setattr("optimized_pipeline.tracker.PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    # 1. Disabled tracking flag
    assert init_tracking(enabled=False) is False

    # 2. Enabled tracking without credentials triggers local SQLite fallback
    # Redirect Path.home() so .dagshub_token is not discovered
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")

    success = init_tracking(enabled=True, experiment_name="Test_Fallback_Exp")
    assert success is True

    # Confirm active tracking URI points to local SQLite database
    uri = mlflow.get_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert "mlflow.db" in uri
    assert (tmp_path / "results" / "mlflow.db").exists()


def test_setup_custom_logger_and_flush(tmp_path: Path):
    """Verify setup_custom_logger emits logs immediately and flushes to disk with 0 delay."""
    from setup_custom_logger import setup_custom_logger

    log_dir = tmp_path / "custom_logs"
    custom_log = setup_custom_logger(
        name="Test_Custom_Logger",
        log_folder=str(log_dir),
        timestamp="2026-10-18_00-00-00",
    )
    assert custom_log is not None
    assert log_dir.exists()

    test_msg = "Immediate unbuffered logging verification check 12345"
    custom_log.info(test_msg)

    # Immediately check file content on disk without waiting or closing logger
    log_file = log_dir / "opt_2026-10-18_00-00-00_rank_0.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content
