"""
Genetic algorithm optimization subpackage for solar_opt.
"""

from .checkpoint import (
    compute_checkpoint_key,
    load_checkpoint,
    safe_pickle_save,
)
from .chromosome import (
    create_individual_generator,
    deserialize_population,
    serialize_population,
    setup_deap_primitives,
)
from .engine import (
    distribute_and_evaluate,
    run_seasonal_optimization,
    run_worker_loop,
)
from .evaluator import evaluate_batch, run_one_simulation_task
from .operators import create_bounded_integer_mutation

__all__ = [
    "compute_checkpoint_key",
    "load_checkpoint",
    "safe_pickle_save",
    "create_individual_generator",
    "deserialize_population",
    "serialize_population",
    "setup_deap_primitives",
    "distribute_and_evaluate",
    "run_seasonal_optimization",
    "run_worker_loop",
    "evaluate_batch",
    "run_one_simulation_task",
    "create_bounded_integer_mutation",
]
