"""
MLflow and DagsHub experiment tracking integration.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import mlflow

try:
    import dagshub
except ImportError:
    dagshub = None

logger = logging.getLogger("NYS_Optimisation")


def init_tracking(
    repo_owner: str = "aryanvj787",
    repo_name: str = "NYS-Design-Optimisation-using-PySAM",
    experiment_name: str = "CSP_Seasonal_GA",
) -> None:
    """Initialize DagsHub remote tracking if available, falling back to local MLflow."""
    if dagshub is not None:
        try:
            dagshub.init(
                repo_owner=repo_owner,
                repo_name=repo_name,
                mlflow=True,
            )
        except Exception as exc:
            logger.warning("Could not initialize DagsHub (%s). Falling back to local MLflow.", exc)
    mlflow.set_experiment(experiment_name)


def safe_name(name: Any) -> str:
    """Sanitize parameter and metric keys for MLflow compliance."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def log_generation_metrics(
    population: Sequence[Any],
    hof: Sequence[Any],
    record: Dict[str, Any],
    nevals: int,
    gen: int,
    design_var: Sequence[str],
    operational_var: Sequence[str],
    season_days: Sequence[Dict[str, Any]],
    simulation_successful: int,
    simulation_penalized: int,
    run_successful_total: int,
    run_penalized_total: int,
    step: Optional[int] = None,
) -> None:
    """Log generation statistics and gene values to MLflow."""
    from deap import tools

    generation_best = tools.selBest(population, k=1)[0]
    global_best = hof[0]

    metrics: Dict[str, float] = {
        "fitness/avg": float(record["avg"]),
        "fitness/std": float(record["std"]),
        "fitness/min": float(record["min"]),
        "fitness/max": float(record["max"]),
        "fitness/nevals": float(nevals),
        "fitness/generation_best": float(generation_best.fitness.values[0]),
        "fitness/global_best": float(global_best.fitness.values[0]),
        "best_fitness": float(global_best.fitness.values[0]),
    }

    sim_total = simulation_successful + simulation_penalized
    success_rate = (100.0 * simulation_successful / sim_total) if sim_total > 0 else 0.0

    metrics.update(
        {
            "simulations/successful": float(simulation_successful),
            "simulations/penalized": float(simulation_penalized),
            "simulations/total": float(sim_total),
            "simulations/success_rate_pct": float(success_rate),
            "simulations/run_successful_total": float(run_successful_total),
            "simulations/run_penalized_total": float(run_penalized_total),
        }
    )

    # Design genes
    for i, name in enumerate(design_var):
        k = safe_name(name)
        metrics[f"generation_best/design/{k}"] = float(generation_best[i])
        metrics[f"global_best/design/{k}"] = float(global_best[i])

    # Operational genes per representative day
    num_op_vars = len(operational_var)
    for day_info in season_days:
        local_idx = day_info["local_index"]
        day_label = f"day_{local_idx + 1:02d}_{day_info['month']:02d}_{day_info['day']:02d}"
        for op_idx, name in enumerate(operational_var):
            gene_idx = len(design_var) + local_idx * num_op_vars + op_idx
            k = safe_name(name)
            metrics[f"generation_best/operational/{day_label}/{k}"] = float(generation_best[gene_idx])
            metrics[f"global_best/operational/{day_label}/{k}"] = float(global_best[gene_idx])

    current_step = step if step is not None else gen
    mlflow.log_metrics(metrics, step=current_step)
