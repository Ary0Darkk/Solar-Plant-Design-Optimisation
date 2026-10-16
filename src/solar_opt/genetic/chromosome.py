"""
Chromosome structure, initialization, and DEAP creator setup.
"""

import random
from typing import Any, Callable, List, Sequence, Tuple

from deap import base, creator, tools


def setup_deap_primitives() -> None:
    """Ensure FitnessMax and Individual classes exist in DEAP creator."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def create_individual_generator(
    design_lb: Sequence[int],
    design_ub: Sequence[int],
    operational_lb: Sequence[int],
    operational_ub: Sequence[int],
    num_days: int,
) -> Callable[[], List[int]]:
    """
    Factory returning a zero-argument generator function that generates a randomized individual:
      - 5 design parameters
      - num_days * 3 operational parameters
    """
    expected_len = len(design_lb) + num_days * len(operational_lb)

    def _generate() -> List[int]:
        ind: List[int] = []
        for lb, ub in zip(design_lb, design_ub):
            ind.append(random.randint(lb, ub))
        for _ in range(num_days):
            for lb, ub in zip(operational_lb, operational_ub):
                ind.append(random.randint(lb, ub))
        assert len(ind) == expected_len, f"Generated individual length {len(ind)} != expected {expected_len}"
        return ind

    return _generate


def serialize_population(population: Sequence[Any]) -> List[Tuple[List[Any], Tuple[float, ...]]]:
    """Serialize individuals and fitness values for pickling."""
    return [
        (list(ind), ind.fitness.values if ind.fitness.valid else ())
        for ind in population
    ]


def deserialize_population(
    serialized: Sequence[Tuple[List[Any], Tuple[float, ...]]],
    toolbox: base.Toolbox,
) -> List[Any]:
    """Reconstruct DEAP Individual objects from serialized tuples."""
    pop = []
    for genes, fit in serialized:
        ind = toolbox.individual()
        ind[:] = genes
        if fit:
            ind.fitness.values = fit
        pop.append(ind)
    return pop
