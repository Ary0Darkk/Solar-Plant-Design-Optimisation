"""
Results formatting, export, and convergence plotting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mlflow
import pandas as pd
import tabulate as tb

from ..config.defaults import PROJECT_ROOT


def plot_convergence(
    gens: Sequence[int],
    max_fitness: Sequence[float],
    avg_fitness: Sequence[float],
    season: str,
    output_path: Path,
) -> None:
    """Plot and save GA fitness convergence trajectory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, max_fitness, label="Max Fitness", color="crimson", lw=2)
    ax.plot(gens, avg_fitness, label="Average Fitness", color="navy", lw=1.5, ls="--")
    ax.set_title(f"GA Convergence Curve ({season.capitalize()})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (INR)", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    if len(gens) > 1:
        step = max(1, len(gens) // 10)
        ax.xaxis.set_major_locator(MultipleLocator(step))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_optimization_results(
    season: str,
    timestamp: str,
    best_ind: Sequence[Any],
    best_fitness: float,
    design_vars: Sequence[str],
    operational_vars: Sequence[str],
    season_days: Sequence[Dict[str, Any]],
    logbook: Sequence[Dict[str, Any]],
    plot_path: Path,
) -> Tuple[Dict[str, Any], Path, Path, Path]:
    """
    Format and export optimization outputs: summary table, results CSV, best solution JSON, and logbook.

    Returns:
        (solution_dict, results_csv_path, solution_json_path, logbook_csv_path)
    """
    results_dir = PROJECT_ROOT / "results" / "GA_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sol_dict: Dict[str, Any] = {
        "season": season,
        "best_fitness": best_fitness,
    }

    for i, name in enumerate(design_vars):
        sol_dict[f"design_{name}"] = best_ind[i]

    num_ops = len(operational_vars)
    for day_idx, day_info in enumerate(season_days):
        day_label = f"day_{day_idx + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        for op_idx, name in enumerate(operational_vars):
            gene_idx = len(design_vars) + day_idx * num_ops + op_idx
            sol_dict[f"{day_label}_{name}"] = best_ind[gene_idx]

    # Save Results CSV
    csv_path = results_dir / f"{season}_{timestamp}.csv"
    df_results = pd.DataFrame([sol_dict])
    df_results.index = df_results.index + 1
    df_results.index.name = "serial"
    df_results.to_csv(csv_path, index=True)

    # Save Best Solution JSON
    json_path = results_dir / f"best_solution_{season}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "season": season,
                "fitness": best_fitness,
                "individual": list(best_ind),
                "details": sol_dict,
            },
            f,
            indent=4,
        )

    # Save GA Logbook CSV
    logbook_path = results_dir / f"{season}_logbook.csv"
    pd.DataFrame(logbook).to_csv(logbook_path, index=False)

    # Log to MLflow if active run exists
    if mlflow.active_run():
        try:
            mlflow.log_artifact(str(csv_path), artifact_path="results")
            mlflow.log_artifact(str(json_path), artifact_path="results")
            mlflow.log_artifact(str(logbook_path), artifact_path="ga_history")
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
        except Exception:
            pass

    return sol_dict, csv_path, json_path, logbook_path


def format_solution_table(sol_dict: Dict[str, Any]) -> str:
    """Format solution dictionary as an aligned grid table."""
    return tb.tabulate(sol_dict.items(), tablefmt="grid")
