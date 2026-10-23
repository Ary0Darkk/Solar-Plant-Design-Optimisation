"""
Experiment tracking, visualization, and multi-format artifact exporters.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tabulate as tb

from .config import PROJECT_ROOT

logger = logging.getLogger("Solar_Optimisation")


def sanitize_key(key: str) -> str:
    """Sanitize metric keys for MLflow compatibility."""
    return re.sub(r"[^a-zA-Z0-9_\-\.\/]", "_", str(key))


def setup_logger(
    rank: int = 0,
    timestamp: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure rank-aware logger streaming to console and rank-specific file."""
    log = logging.getLogger("Solar_Optimisation")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = log_dir or (PROJECT_ROOT / "logs")
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / f"run_{ts}_rank_{rank}.log"

    fmt = logging.Formatter(
        f"[%(asctime)s] [Rank {rank}] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        log.addHandler(ch)

    return log


def init_tracking(
    enabled: bool = True,
    experiment_name: str = "CSP_Plant_Optimization",
) -> bool:
    """Initialize MLflow / DagsHub tracking with graceful offline SQLite fallback."""
    if not enabled:
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    dagshub_user = (
        os.environ.get("DAGSHUB_USER_NAME")
        or os.environ.get("MLFLOW_TRACKING_USERNAME")
        or "aryanvj787"
    )
    dagshub_token = (
        os.environ.get("DAGSHUB_USER_TOKEN")
        or os.environ.get("DAGSHUB_TOKEN")
        or os.environ.get("MLFLOW_TRACKING_PASSWORD")
    )
    dagshub_repo = os.environ.get("DAGSHUB_REPO_NAME", "NYS-Design-Optimisation-using-PySAM")

    custom_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # 1. Explicit SQLite URI takes priority if requested
    if custom_uri and "sqlite" in custom_uri:
        try:
            db_path = PROJECT_ROOT / "results" / "mlflow.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(custom_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("Local MLflow SQLite tracking active at: %s", custom_uri)
            return True
        except Exception as exc:
            logger.warning("Local MLflow initialization failed (%s). Continuing without tracking.", exc)
            return False

    if not dagshub_token:
        try:
            import dagshub.auth

            dagshub_token = dagshub.auth.get_token()
        except Exception:
            pass

    # 2. Custom remote or DagsHub URI
    if custom_uri and custom_uri.startswith(("http://", "https://")):
        try:
            if dagshub_token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_tracking_uri(custom_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("Remote MLflow tracking connected via URI: %s", custom_uri)
            return True
        except Exception as exc:
            logger.warning("Failed to connect to remote MLflow URI (%s): %s", custom_uri, exc)

    # 3. DagsHub client integration if token available
    if dagshub_token and dagshub_user:
        try:
            import dagshub

            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            dagshub.init(
                repo_owner=dagshub_user,
                repo_name=dagshub_repo,
                mlflow=True,
            )
            mlflow.set_experiment(experiment_name)
            logger.info(
                "Remote MLflow tracking connected via DagsHub (%s/%s)", dagshub_user, dagshub_repo
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to connect to DagsHub MLflow (%s). Falling back to local tracking.", exc
            )

    # 3. Local SQLite fallback
    try:
        db_path = PROJECT_ROOT / "results" / "mlflow.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_uri = f"sqlite:///{db_path.as_posix()}"
        os.environ["MLFLOW_TRACKING_URI"] = sqlite_uri
        mlflow.set_tracking_uri(sqlite_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("Local MLflow SQLite tracking active at: %s", sqlite_uri)
        return True
    except Exception as exc:
        logger.warning("Local MLflow initialization failed (%s). Continuing without tracking.", exc)
        return False


def log_run_params(
    config: Any,
    season: str,
    workers: int,
    job_id: Optional[str] = None,
) -> None:
    """Log pipeline configuration hyperparameters and physical bounds to active MLflow run."""
    if not mlflow.active_run():
        return

    try:
        params = {
            "season": str(season),
            "pop_size": int(config.ga.pop_size),
            "generations": int(config.ga.num_generations),
            "tournament_size": int(config.ga.tournament_size),
            "hall_of_fame_size": int(config.ga.hall_of_fame_size),
            "cxpb": float(config.ga.cxpb),
            "mutpb": float(config.ga.mutpb),
            "indpb": float(config.ga.indpb),
            "random_seed": int(config.ga.random_seed),
            "workers": int(workers),
            "early_stopping_patience": int(config.ga.early_stopping_patience),
            "tariff/min_price": float(config.tariff.min_price),
            "tariff/max_price": float(config.tariff.max_price),
            "simulation/model": str(config.simulation.model_name),
            "simulation/penalty": float(config.simulation.penalty),
        }
        if job_id:
            params["job_id"] = str(job_id)

        # Log physical design and operational bounds
        for name, low, high in zip(config.design.names, config.design.lb, config.design.ub):
            params[f"bounds/design/{sanitize_key(name)}/min"] = low
            params[f"bounds/design/{sanitize_key(name)}/max"] = high

        for name, low, high in zip(
            config.operational.names, config.operational.lb, config.operational.ub
        ):
            params[f"bounds/operational/{sanitize_key(name)}/min"] = low
            params[f"bounds/operational/{sanitize_key(name)}/max"] = high

        mlflow.log_params(params)
    except Exception as exc:
        logger.debug("Failed logging parameters to MLflow: %s", exc)


def log_step_metrics(
    pop: Sequence[Any],
    hof: Sequence[Any],
    record: Dict[str, Any],
    gen: int,
    design_names: Sequence[str],
    operational_names: Sequence[str],
    season_days: Sequence[Dict[str, Any]],
    sim_successful: int,
    sim_penalized: int,
    total_successful: int,
    total_penalized: int,
) -> None:
    """Record generational statistics and global best design parameters."""
    if not hof:
        return

    glob_best = hof[0]
    gen_best = max(pop, key=lambda ind: ind.fitness.values[0]) if pop else glob_best
    sim_total = sim_successful + sim_penalized
    success_rate = (sim_successful / sim_total * 100.0) if sim_total > 0 else 0.0

    metrics: Dict[str, float] = {
        "fitness/avg": float(record["avg"]),
        "fitness/std": float(record["std"]),
        "fitness/min": float(record["min"]),
        "fitness/max": float(record["max"]),
        "fitness/generation_best": float(gen_best.fitness.values[0]),
        "fitness/global_best": float(glob_best.fitness.values[0]),
        "best_fitness": float(glob_best.fitness.values[0]),
        "simulations/successful": float(sim_successful),
        "simulations/penalized": float(sim_penalized),
        "simulations/total": float(sim_total),
        "simulations/success_rate_pct": float(success_rate),
        "simulations/total_successful": float(total_successful),
        "simulations/total_penalized": float(total_penalized),
    }

    # Design genes
    for i, name in enumerate(design_names):
        k = sanitize_key(name)
        metrics[f"global_best/design/{k}"] = float(glob_best[i])

    # Operational genes
    n_ops = len(operational_names)
    for day_info in season_days:
        l_idx = day_info["local_index"]
        d_lbl = f"day_{l_idx + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        for op_idx, name in enumerate(operational_names):
            gene_idx = len(design_names) + l_idx * n_ops + op_idx
            k = sanitize_key(name)
            metrics[f"global_best/operational/{d_lbl}/{k}"] = float(glob_best[gene_idx])

    clean_metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
    try:
        if mlflow.active_run():
            mlflow.log_metrics(clean_metrics, step=gen)
    except Exception:
        pass


def plot_and_export(
    season: str,
    timestamp: str,
    best_ind: Sequence[Any],
    best_fitness: float,
    design_names: Sequence[str],
    operational_names: Sequence[str],
    season_days: Sequence[Dict[str, Any]],
    logbook: Sequence[Dict[str, Any]],
    gens: Sequence[int],
    max_fitness: Sequence[float],
    avg_fitness: Sequence[float],
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate final plots and export CSV, JSON, Excel, and Markdown artifacts."""
    if job_id is None:
        job_id = (
            os.environ.get("PBS_JOBID")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("JOB_ID")
        )

    results_dir = PROJECT_ROOT / "results" / "GA_results"
    plots_dir = PROJECT_ROOT / "plots" / "GA_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dual format plot export (300 DPI PNG + vector PDF)
    plot_path_png = plots_dir / f"{season}_convergence_{timestamp}.png"
    plot_path_pdf = plots_dir / f"{season}_convergence_{timestamp}.pdf"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, max_fitness, label="Max Fitness", color="crimson", lw=2)
    ax.plot(gens, avg_fitness, label="Avg Fitness", color="navy", lw=1.5, ls="--")
    ax.set_title(f"GA Convergence ({season.capitalize()})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (INR)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path_png, dpi=300)
    fig.savefig(plot_path_pdf)
    plt.close(fig)

    # 2. Results dictionary
    sol: Dict[str, Any] = {"season": season, "best_fitness": best_fitness}
    if job_id:
        sol["job_id"] = str(job_id)
    design_dict: Dict[str, Any] = {}
    for i, name in enumerate(design_names):
        sol[f"design_{name}"] = best_ind[i]
        design_dict[name] = best_ind[i]

    n_ops = len(operational_names)
    ops_rows = []
    for day_info in season_days:
        l_idx = day_info["local_index"]
        d_lbl = f"day_{l_idx + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        row = {
            "Day": l_idx + 1,
            "Date": f"{day_info['month']:02d}-{day_info['day']:02d}",
            "Day_Index": day_info["day_index"],
        }
        for op_idx, name in enumerate(operational_names):
            gene_idx = len(design_names) + l_idx * n_ops + op_idx
            val = best_ind[gene_idx]
            sol[f"{d_lbl}_{name}"] = val
            row[name] = val
        ops_rows.append(row)

    # 3. Save CSV & JSON
    csv_path = results_dir / f"{season}_{timestamp}.csv"
    pd.DataFrame([sol]).to_csv(csv_path, index=False)

    json_path = results_dir / f"best_solution_{season}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"season": season, "fitness": best_fitness, "solution": sol}, f, indent=4)

    # 4. Save logbook CSV
    logbook_path = results_dir / f"{season}_logbook.csv"
    df_log = pd.DataFrame(logbook)
    df_log.to_csv(logbook_path, index=False)

    # 5. Multi-sheet Excel workbook export
    excel_path = results_dir / f"{season}_summary_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame([{"Property": k, "Value": v} for k, v in sol.items()]).to_excel(
                writer, sheet_name="Optimal Solution", index=False
            )
            df_log.to_excel(writer, sheet_name="GA History", index=False)
            pd.DataFrame(ops_rows).to_excel(writer, sheet_name="Operational Schedule", index=False)
    except Exception as exc:
        logger.warning("Could not export multi-sheet Excel summary: %s", exc)

    # 6. Executive Markdown summary report
    report_path = results_dir / f"{season}_executive_report_{timestamp}.md"
    initial_fit = max_fitness[0] if max_fitness else best_fitness
    fit_gain = best_fitness - initial_fit
    report_lines = [
        f"# CSP Genetic Algorithm Optimization Report — {season.capitalize()}",
        "",
        f"- **Job ID**: `{job_id}`" if job_id else "- **Job ID**: `local`",
        f"- **Execution Timestamp**: `{timestamp}`",
        f"- **Season**: `{season}`",
        f"- **Final Best Fitness**: `{best_fitness:,.2f} INR`",
        f"- **Initial Gen 0 Best**: `{initial_fit:,.2f} INR`",
        f"- **Optimization Gain**: `+{fit_gain:,.2f} INR`",
        f"- **Total Generations**: `{len(gens)}`",
        "",
        "## 1. Optimal Plant Design Parameters",
        "| Parameter | Value | Unit / Description |",
        "| :--- | :--- | :--- |",
    ]
    units_map = {
        "specified_total_aperture": "m² (Solar field reflective area)",
        "Row_Distance": "m (Collector row spacing pitch)",
        "ColperSCA": "modules (Collector modules per SCA)",
        "W_aperture": "m (Reflective aperture width)",
        "L_SCA": "m (Collector assembly length)",
    }
    for k, v in design_dict.items():
        desc = units_map.get(k, "")
        report_lines.append(f"| `{k}` | **{v}** | {desc} |")

    report_lines.extend(
        [
            "",
            "## 2. Daily Operational Schedule (7 Representative Days)",
            "| Day | Date | Mass Flow Limit (kg/s) | Startup Temp (°C) | Shutdown Temp (°C) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for r in ops_rows:
        report_lines.append(
            f"| Day {r['Day']} | `{r['Date']}` | **{r.get('m_dot', '-')}** | "
            f"**{r.get('T_startup', '-')}** | **{r.get('T_shutdown', '-')}** |"
        )
    report_lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 7. Log all summary metrics & artifacts to MLflow if active
    if mlflow.active_run():
        try:
            mlflow.log_metric("final/best_fitness", float(best_fitness))
            mlflow.log_metric("final/initial_fitness", float(initial_fit))
            mlflow.log_metric("final/optimization_gain", float(fit_gain))
            mlflow.log_metric("final/total_generations", float(len(gens)))
            for k, v in design_dict.items():
                mlflow.log_metric(f"final/design/{sanitize_key(k)}", float(v))

            mlflow.log_artifact(str(csv_path), artifact_path="results")
            mlflow.log_artifact(str(json_path), artifact_path="results")
            mlflow.log_artifact(str(report_path), artifact_path="results")
            if excel_path.exists():
                mlflow.log_artifact(str(excel_path), artifact_path="results")
            mlflow.log_artifact(str(logbook_path), artifact_path="ga_history")
            if plot_path_png.exists():
                mlflow.log_artifact(str(plot_path_png), artifact_path="plots")
            if plot_path_pdf.exists():
                mlflow.log_artifact(str(plot_path_pdf), artifact_path="plots")
        except Exception:
            pass

    table_str = tb.tabulate(sol.items(), tablefmt="grid")
    logger.info("\n%s\nOptimal Solution (%s):\n%s\n%s", "=" * 40, season, table_str, "=" * 40)

    return sol
