"""
Tests for artifact plotting, export formats, and tracking.
"""

from pathlib import Path

from solar_opt.tracker import plot_and_export, sanitize_key, setup_logger


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


def test_plot_and_export_generation():
    best_ind = [10000, 15, 4, 5, 100] + [5, 350, 280] * 7
    design_names = ["specified_total_aperture", "Row_Distance", "ColperSCA", "W_aperture", "L_SCA"]
    op_names = ["m_dot", "T_startup", "T_shutdown"]
    season_days = [{"month": 3, "day": 9, "local_index": i, "day_index": 67 + i} for i in range(7)]
    logbook = [{"gen": 0, "nevals": 10, "avg": 100.0, "std": 5.0, "min": 50.0, "max": 150.0}]

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
