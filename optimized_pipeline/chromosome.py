"""
Memory-efficient integer chromosome representation and DEAP primitives.
"""

import random
from typing import Any, Callable, List, Sequence, Tuple

from deap import base, creator, tools
import numpy as np

from .config import DESIGN_BOUNDS, OPERATIONAL_BOUNDS


def init_deap_types() -> None:
    """Ensure FitnessMax and Individual types are registered in DEAP creator."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def build_gene_generator(
    design_lb: Sequence[int],
    design_ub: Sequence[int],
    operational_lb: Sequence[int],
    operational_ub: Sequence[int],
    num_days: int,
) -> Callable[[], List[int]]:
    """Factory creating an individual chromosome as compact integers."""
    expected_length = len(design_lb) + num_days * len(operational_lb)

    def _generate() -> List[int]:
        genes = [random.randint(low, high) for low, high in zip(design_lb, design_ub)]
        for _ in range(num_days):
            for low, high in zip(operational_lb, operational_ub):
                genes.append(random.randint(low, high))
        assert len(genes) == expected_length
        return genes

    return _generate


def serialize_pop(pop: Sequence[Any]) -> List[Tuple[List[int], Tuple[float, ...]]]:
    """Serialize population to compact tuples for disk storage."""
    return [
        (list(ind), tuple(ind.fitness.values) if ind.fitness.valid else ())
        for ind in pop
    ]


def deserialize_pop(
    serialized: Sequence[Tuple[List[int], Tuple[float, ...]]],
    toolbox: base.Toolbox,
) -> List[Any]:
    """Reconstruct DEAP Individual instances from serialized tuples."""
    pop = []
    for genes, fit in serialized:
        ind = toolbox.individual()
        ind[:] = genes
        if fit:
            ind.fitness.values = fit
        pop.append(ind)
    return pop
