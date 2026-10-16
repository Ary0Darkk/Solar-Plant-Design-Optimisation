"""
Season-wise DEAP Genetic Algorithm Optimization module.
Backward compatibility wrapper delegating to solar_opt.
"""

from solar_opt.cli import main
from solar_opt.genetic.engine import (
    distribute_and_evaluate as distribute_batches,
    run_seasonal_optimization as run_deap_ga_optimisation,
    run_worker_loop as worker_loop,
)
from solar_opt.genetic.evaluator import (
    evaluate_batch,
    run_one_simulation_task as run_one_simulation,
)
from solar_opt.tracking.mlflow_tracker import log_generation_metrics

__all__ = [
    "run_deap_ga_optimisation",
    "worker_loop",
    "distribute_batches",
    "evaluate_batch",
    "run_one_simulation",
    "log_generation_metrics",
    "main",
]

if __name__ == "__main__":
    main()
