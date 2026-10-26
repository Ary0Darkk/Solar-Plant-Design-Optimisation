"""
Experiment Tracking, Artifact Exporting, and Remote DagsHub / MLflow Integration.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_key(key: str) -> str:
    """Sanitize MLflow metric / parameter key to conform to naming regex."""
    return re.sub(r"[^a-zA-Z0-9_\-\./]", "_", str(key))


def setup_logger(
    rank: int = 0,
    timestamp: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure formatted dual console and file logging."""
    logger = logging.getLogger(f"solar_opt_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][Rank %(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if rank == 0:
        target_dir = log_dir or (PROJECT_ROOT / "logs")
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            target_dir / f"run_{ts_str}_rank_0.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def init_tracking(
    enabled: bool = True,
    experiment_name: str = "CSP_Seasonal_GA",
    disabled: bool = False,
    tracking_uri: Optional[str] = None,
) -> bool:
    """
    Initialize MLflow remote experiment tracking (DagsHub or remote MLflow server).
    Local storage (SQLite/mlruns) is completely disabled.
    Returns True if remote MLflow tracking is active.
    """
    if not enabled or disabled or os.environ.get("DISABLE_TRACKING", "").lower() in ["1", "true"]:
        return False

    import mlflow

    # 1. If explicit tracking_uri argument was passed, use it directly (e.g. for testing)
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            return True
        except Exception as exc:
            logging.getLogger("solar_opt").warning(
                f"MLflow connection to {tracking_uri} failed: {exc}"
            )
            return False

    # 2. Try DagsHub Remote Integration
    repo = os.environ.get("DAGSHUB_REPO_NAME", "NYS-Design-Optimisation-using-PySAM")
    user = os.environ.get("DAGSHUB_USER_NAME", "aryanvj787")
    token = os.environ.get("DAGSHUB_USER_TOKEN") or os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if token:
        try:
            import dagshub

            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            os.environ["DAGSHUB_USER_TOKEN"] = token
            dagshub.auth.add_app_token(token)

            dagshub.init(repo_owner=user, repo_name=repo.split("/")[-1], mlflow=True)
            mlflow.set_experiment(experiment_name)
            return True
        except Exception as exc:
            logging.getLogger("solar_opt").warning(
                f"DagsHub remote tracking initialization failed: {exc}"
            )

    # 3. If MLFLOW_TRACKING_URI environment variable is configured
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        try:
            mlflow.set_tracking_uri(env_uri)
            mlflow.set_experiment(experiment_name)
            return True
        except Exception as exc:
            logging.getLogger("solar_opt").warning(
                f"MLflow remote connection to {env_uri} failed: {exc}"
            )
            return False

    # Local fallback is explicitly removed - return False if no remote tracking is configured
    return False


def log_run_params(
    params: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    season: Optional[str] = None,
    workers: Optional[int] = None,
    job_id: Optional[str] = None,
    **kwargs,
) -> None:
    """Log parameters to active MLflow run."""
    try:
        import mlflow

        if mlflow.active_run():
            p_dict = {}
            if params:
                p_dict.update(params)
            if config:
                if hasattr(config, "ga"):
                    p_dict["pop_size"] = config.ga.pop_size
                    p_dict["generations"] = config.ga.num_generations
                if hasattr(config, "tariff"):
                    p_dict["min_price"] = config.tariff.min_price
                    p_dict["max_price"] = config.tariff.max_price
            if season:
                p_dict["season"] = season
            if workers:
                p_dict["workers"] = workers
            if job_id:
                p_dict["job_id"] = job_id
            p_dict.update(kwargs)

            clean_params = {sanitize_key(k): str(v) for k, v in p_dict.items()}
            mlflow.log_params(clean_params)
    except Exception:
        pass


def log_step_metrics(
    pop: List[Any],
    hof: List[Any],
    record: Dict[str, Any],
    gen: int,
    design_names: Tuple[str, ...],
    operational_names: Tuple[str, ...],
    season_days: List[Dict[str, int]],
    sim_successful: int,
    sim_penalized: int,
    total_successful: int,
    total_penalized: int,
) -> None:
    """Log generational metrics and elite values to active MLflow run."""
    try:
        import mlflow

        if not mlflow.active_run():
            return

        metrics = {}
        for k in ["avg", "std", "min", "max"]:
            if k in record:
                metrics[f"fitness_{k}"] = float(record[k])

        metrics["gen_sim_success"] = float(sim_successful)
        metrics["gen_sim_penalized"] = float(sim_penalized)
        metrics["cum_sim_success"] = float(total_successful)
        metrics["cum_sim_penalized"] = float(total_penalized)

        if hof and len(hof) > 0:
            elite = hof[0]
            metrics["elite_fitness"] = (
                float(elite.fitness.values[0]) if elite.fitness.valid else 0.0
            )

        clean_metrics = {sanitize_key(k): v for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics, step=gen)
    except Exception:
        pass


def plot_and_export(
    season: str,
    timestamp: str,
    best_ind: List[float],
    best_fitness: float,
    design_names: Tuple[str, ...],
    operational_names: Tuple[str, ...],
    season_days: List[Dict[str, int]],
    logbook: Any,
    gens: List[int],
    max_fitness: List[float],
    avg_fitness: List[float],
    job_id: str = "local",
    out_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate publication-ready artifacts:
    1. 300 DPI PNG & vector PDF convergence plots
    2. Multi-sheet Excel summary workbook
    3. Formatted Executive Markdown Report
    4. Best solution JSON
    """
    out_dir = out_dir or (PROJECT_ROOT / "results" / "GA_results")
    plots_dir = plots_dir or (PROJECT_ROOT / "plots" / "GA_plots")
    out_dir = Path(out_dir)
    plots_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp
    result_data = {
        "season": season,
        "best_fitness": best_fitness,
        "job_id": job_id,
        "timestamp": ts,
    }

    # 1. Dual Convergence Plots (PNG & Vector PDF)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(gens, max_fitness, label="Max Fitness (Best)", color="#2563EB", linewidth=2.5)
    ax.plot(
        gens,
        avg_fitness,
        label="Mean Fitness",
        color="#059669",
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_title(
        f"HelioOptima CSP Optimization Convergence - {season.upper()}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Net Daily Operational Profit (INR)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=11)
    fig.tight_layout()

    png_path = plots_dir / f"{season}_convergence_{ts}.png"
    pdf_path = plots_dir / f"{season}_convergence_{ts}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    result_data["png"] = png_path
    result_data["pdf"] = pdf_path

    # 2. Multi-Sheet Excel Summary
    excel_path = out_dir / f"{season}_summary_{ts}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        ind = list(best_ind)
        design_rows = [
            {"Parameter": "Season", "Value": season.upper(), "Unit": "-"},
            {"Parameter": "Job ID", "Value": job_id, "Unit": "-"},
            {"Parameter": "Max Net Profit", "Value": round(best_fitness, 2), "Unit": "INR"},
            {
                "Parameter": "Total Aperture Area",
                "Value": round(ind[0], 2),
                "Unit": "m²",
            },
            {"Parameter": "Row Pitch", "Value": round(ind[1], 2), "Unit": "m"},
            {"Parameter": "Collectors per SCA", "Value": int(round(ind[2])), "Unit": "count"},
            {"Parameter": "Aperture Width", "Value": round(ind[3], 2), "Unit": "m"},
            {"Parameter": "SCA Length", "Value": round(ind[4], 2), "Unit": "m"},
        ]
        pd.DataFrame(design_rows).to_excel(writer, sheet_name="Optimal_Design", index=False)

        if logbook:
            try:
                pd.DataFrame(logbook).to_excel(
                    writer, sheet_name="Generational_History", index=False
                )
            except Exception:
                pass

        op_rows = []
        for d in range(7):
            op_rows.append(
                {
                    "Day": f"Day {d + 1}",
                    "m_dot (kg/s)": round(ind[5 + d * 3 + 0], 2),
                    "T_startup (°C)": round(ind[5 + d * 3 + 1], 2),
                    "T_shutdown (°C)": round(ind[5 + d * 3 + 2], 2),
                }
            )
        pd.DataFrame(op_rows).to_excel(writer, sheet_name="Operational_Schedule", index=False)

    result_data["excel"] = excel_path

    # 3. Executive Markdown Report
    report_path = out_dir / f"{season}_executive_report_{ts}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# HelioOptima CSP Optimization Executive Report\n\n")
        f.write(f"- **Season**: {season.upper()}\n")
        f.write(f"- **Optimal Net Daily Profit**: ₹{result_data['best_fitness']:,.2f}\n")
        f.write(f"- **Job ID**: {result_data['job_id']}\n")
        f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Optimal Plant Design Parameters\n\n")
        f.write("| Parameter | Optimal Value | Unit |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Total Solar Aperture | {ind[0]:,.2f} | m² |\n")
        f.write(f"| Row Pitch Distance | {ind[1]:.2f} | m |\n")
        f.write(f"| Collectors per Assembly | {int(round(ind[2]))} | count |\n")
        f.write(f"| Aperture Width | {ind[3]:.2f} | m |\n")
        f.write(f"| SCA Length | {ind[4]:.2f} | m |\n")
    result_data["report"] = report_path

    # 4. JSON Solution
    json_path = out_dir / f"best_solution_{season}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "season": season,
                "best_fitness": result_data["best_fitness"],
                "job_id": result_data["job_id"],
                "genes": ind,
            },
            f,
            indent=2,
        )
    result_data["json"] = json_path

    # 5. Upload to MLflow if run is active
    try:
        import mlflow

        if mlflow.active_run():
            for k in ["png", "pdf", "excel", "report", "json"]:
                if k in result_data and Path(result_data[k]).exists():
                    mlflow.log_artifact(str(result_data[k]))
    except Exception:
        pass

    return result_data
