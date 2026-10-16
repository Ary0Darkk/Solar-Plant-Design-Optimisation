"""
Synchronized rank logging, optional MLflow/DagsHub tracking, and results exporting.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tabulate as tb

from .config import PROJECT_ROOT

try:
    import dagshub
except ImportError:
    dagshub = None

logger = logging.getLogger("NYS_Optimisation")


class ColoredFormatter(logging.Formatter):
    """ANSI color-coded formatter for clean console logs."""

    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m\033[1m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{super().format(record)}{self.RESET}"


def setup_logger(
    rank: int = 0,
    timestamp: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Initialize file and rank-0 console logging."""
    folder = log_dir or (PROJECT_ROOT / "logs")
    folder.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("NYS_Optimisation")
    log.setLevel(logging.INFO)
    log.propagate = False

    if log.handlers:
        log.handlers.clear()

    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = folder / f"opt_{ts}_rank_{rank}.log"

    f_format = logging.Formatter(
        "%(asctime)s | PID:%(process)-5d | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
    )
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(f_format)
    log.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(ColoredFormatter("%(asctime)s | PID:%(process)-5d | %(levelname)-8s | %(message)s"))
        log.addHandler(ch)

    return log


def init_tracking(
    enabled: bool = True,
    repo_owner: str = "aryanvj787",
    repo_name: str = "NYS-Design-Optimisation-using-PySAM",
    experiment_name: str = "CSP_Seasonal_GA",
) -> bool:
    """Connect to DagsHub / MLflow if enabled and available."""
    if not enabled:
        logger.info("Remote experiment tracking disabled by user flag.")
        return False

    if dagshub is not None:
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_experiment(experiment_name)
            return True
        except Exception as exc:
            logger.warning("Could not initialize DagsHub (%s). Continuing with local MLflow fallback.", exc)

    try:
        mlflow.set_experiment(experiment_name)
        return True
    except Exception:
        return False


def sanitize_key(name: Any) -> str:
    """Sanitize parameter and metric keys for MLflow."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def log_step_metrics(
    population: Sequence[Any],
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
    """Log generation metrics and gene values to MLflow."""
    from deap import tools

    gen_best = tools.selBest(population, k=1)[0]
    glob_best = hof[0]

    sim_total = sim_successful + sim_penalized
    success_rate = (100.0 * sim_successful / sim_total) if sim_total > 0 else 0.0

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
) -> Dict[str, Any]:
    """Generate final plot and export CSV, JSON, and logbook artifacts."""
    results_dir = PROJECT_ROOT / "results" / "GA_results"
    plots_dir = PROJECT_ROOT / "plots" / "GA_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot convergence
    plot_path = plots_dir / f"{season}_convergence_{timestamp}.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, max_fitness, label="Max Fitness", color="crimson", lw=2)
    ax.plot(gens, avg_fitness, label="Avg Fitness", color="navy", lw=1.5, ls="--")
    ax.set_title(f"GA Convergence ({season.capitalize()})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (INR)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    # 2. Results dictionary
    sol: Dict[str, Any] = {"season": season, "best_fitness": best_fitness}
    for i, name in enumerate(design_names):
        sol[f"design_{name}"] = best_ind[i]

    n_ops = len(operational_names)
    for day_info in season_days:
        l_idx = day_info["local_index"]
        d_lbl = f"day_{l_idx + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        for op_idx, name in enumerate(operational_names):
            gene_idx = len(design_names) + l_idx * n_ops + op_idx
            sol[f"{d_lbl}_{name}"] = best_ind[gene_idx]

    # Save CSV & JSON
    csv_path = results_dir / f"{season}_{timestamp}.csv"
    pd.DataFrame([sol]).to_csv(csv_path, index=False)

    json_path = results_dir / f"best_solution_{season}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"season": season, "fitness": best_fitness, "solution": sol}, f, indent=4)

    # Save logbook
    logbook_path = results_dir / f"{season}_logbook.csv"
    pd.DataFrame(logbook).to_csv(logbook_path, index=False)

    # Log to MLflow if active
    if mlflow.active_run():
        try:
            mlflow.log_artifact(str(csv_path), artifact_path="results")
            mlflow.log_artifact(str(json_path), artifact_path="results")
            mlflow.log_artifact(str(logbook_path), artifact_path="ga_history")
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
        except Exception:
            pass

    table_str = tb.tabulate(sol.items(), tablefmt="grid")
    logger.info("\n%s\nOptimal Solution (%s):\n%s\n%s", "=" * 40, season, table_str, "=" * 40)

    return sol
