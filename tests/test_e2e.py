"""
Comprehensive End-to-End integration tests for the Solar Optimization Pipeline.
"""

import json
from pathlib import Path

import pandas as pd

from solar_opt.cli import main
from solar_opt.config import PROJECT_ROOT, PipelineConfig
from solar_opt.engine import run_optimization
from solar_opt.parallel import StandaloneComm


def test_e2e_engine_with_multiprocessing_pool():
    """Test full end-to-end optimization with multiprocessing pool and artifact creation."""
    import multiprocessing as mp

    cfg = PipelineConfig()
    cfg.ga.pop_size = 6
    cfg.ga.num_generations = 2
    cfg.ga.resume_from_checkpoint = False

    ctx = mp.get_context()
    comm = StandaloneComm()

    with ctx.Pool(processes=2, maxtasksperchild=cfg.ga.max_tasks_per_child) as pool:
        sol_dict, best_fit, ga_state = run_optimization(
            season="winter",
            config=cfg,
            pool=pool,
            comm=comm,
            rank=0,
            mpi_size=1,
            job_id="test_e2e_pool_run",
        )

    # 1. Validate solution dictionary
    assert sol_dict["season"] == "winter"
    assert "best_fitness" in sol_dict
    assert sol_dict["job_id"] == "test_e2e_pool_run"

    # Verify all 5 design variables exist
    for d_name in cfg.design.names:
        assert f"design_{d_name}" in sol_dict

    # Verify all 7 days * 3 operational variables exist
    for day_idx in range(1, 8):
        assert f"day_{day_idx:02d}_" in "".join(sol_dict.keys())

    # 2. Validate GA evolution history
    logbook = ga_state["logbook"]
    assert len(logbook) == 2
    assert logbook[0]["gen"] == 0
    assert logbook[1]["gen"] == 1
    assert logbook[1]["nevals"] == 6

    # 3. Validate Hall of Fame
    hof = ga_state["hof"]
    assert len(hof) > 0
    assert len(hof[0]) == 26  # 5 design + 21 operational genes


def test_e2e_cli_execution_and_artifact_exports(tmp_path: Path):
    """Test running CLI from arguments and verifying all exported artifact files."""
    # Run a fast 2-generation optimization via CLI
    cli_args = [
        "--season",
        "winter",
        "--pop-size",
        "4",
        "--generations",
        "2",
        "--workers",
        "2",
        "--no-tracking",
        "--job-id",
        "e2e_cli_verification_job",
    ]
    main(cli_args)

    results_dir = PROJECT_ROOT / "results" / "GA_results"
    plots_dir = PROJECT_ROOT / "plots" / "GA_plots"

    assert results_dir.exists(), "Results directory was not created"
    assert plots_dir.exists(), "Plots directory was not created"

    # Verify JSON solution
    json_path = results_dir / "best_solution_winter.json"
    assert json_path.exists(), "best_solution_winter.json not found"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["season"] == "winter"
        assert "fitness" in data
        assert "solution" in data

    # Verify logbook CSV
    logbook_csv = results_dir / "winter_logbook.csv"
    assert logbook_csv.exists(), "winter_logbook.csv not found"
    df_log = pd.read_csv(logbook_csv)
    assert len(df_log) >= 2
    assert "gen" in df_log.columns
    assert "max" in df_log.columns

    # Verify convergence plots exist
    png_plots = list(plots_dir.glob("winter_convergence_*.png"))
    pdf_plots = list(plots_dir.glob("winter_convergence_*.pdf"))
    assert len(png_plots) > 0, "PNG convergence plot was not generated"
    assert len(pdf_plots) > 0, "PDF convergence plot was not generated"
    assert png_plots[-1].stat().st_size > 1000, "PNG plot is empty"
    assert pdf_plots[-1].stat().st_size > 1000, "PDF plot is empty"

    # Verify Excel workbook exists and has the expected sheets
    excel_files = list(results_dir.glob("winter_summary_*.xlsx"))
    assert len(excel_files) > 0, "Excel summary was not generated"
    xls = pd.ExcelFile(excel_files[-1])
    assert "Optimal Solution" in xls.sheet_names
    assert "GA History" in xls.sheet_names
    assert "Operational Schedule" in xls.sheet_names

    # Verify Executive Report exists
    reports = list(results_dir.glob("winter_executive_report_*.md"))
    assert len(reports) > 0, "Executive markdown report was not generated"
    report_content = reports[-1].read_text(encoding="utf-8")
    assert "CSP Genetic Algorithm Optimization Report" in report_content
    assert "Optimal Plant Design Parameters" in report_content
    assert "Daily Operational Schedule" in report_content
