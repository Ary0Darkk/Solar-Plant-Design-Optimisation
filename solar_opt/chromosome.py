"""
DEAP 26-Gene Chromosome Encoder and Type Initializer.
"""

import random
from typing import Any, Dict, List, Tuple

from deap import base, creator, tools

from .config import VariableBounds


def init_deap_types():
    """Register FitnessMax and Individual classes with DEAP creator."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def generate_gene_vector(
    design_bounds: VariableBounds,
    operational_bounds: VariableBounds,
    num_days: int = 7,
) -> List[float]:
    """
    Generate a 26-element random float vector satisfying boundary constraints.
    5 Design genes + (3 Operational genes * 7 days) = 26 genes.
    """
    genes = []

    # 1. Design genes (5)
    for lb, ub, tp in zip(design_bounds.lb, design_bounds.ub, design_bounds.types):
        val = random.uniform(lb, ub)
        if tp is int:
            val = float(round(val))
        genes.append(val)

    # 2. Operational genes (3 per day * 7 days = 21)
    for _ in range(num_days):
        for lb, ub in zip(operational_bounds.lb, operational_bounds.ub):
            genes.append(random.uniform(lb, ub))

    return genes


def build_toolbox(
    design_bounds: VariableBounds,
    operational_bounds: VariableBounds,
    num_days: int = 7,
) -> base.Toolbox:
    """Build DEAP toolbox configured with individual and population generators."""
    init_deap_types()
    toolbox = base.Toolbox()

    toolbox.register(
        "individual_guess",
        generate_gene_vector,
        design_bounds=design_bounds,
        operational_bounds=operational_bounds,
        num_days=num_days,
    )
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.individual_guess,
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
    )

    return toolbox


def serialize_pop(population: List[Any]) -> List[Tuple[List[float], float]]:
    """Serialize DEAP population into lightweight picklable primitives."""
    serialized = []
    for ind in population:
        fit = ind.fitness.values[0] if ind.fitness.valid else float("-inf")
        serialized.append((list(ind), float(fit)))
    return serialized


def deserialize_pop(serialized: List[Tuple[List[float], float]]) -> List[Any]:
    """Reconstruct DEAP population from serialized tuples."""
    init_deap_types()
    population = []
    for genes, fit in serialized:
        ind = creator.Individual(genes)
        if fit != float("-inf"):
            ind.fitness.values = (fit,)
        population.append(ind)
    return population


def decode_individual(
    ind: List[float],
    design_names: Tuple[str, ...],
    operational_names: Tuple[str, ...],
    day_offset: int = 0,
) -> Dict[str, Any]:
    """Extract design parameters and day-specific operational overrides."""
    d_params = {}
    for i, name in enumerate(design_names):
        val = ind[i]
        if name == "ColperSCA":
            val = int(round(val))
        d_params[name] = val

    op_start = len(design_names) + day_offset * len(operational_names)
    for j, name in enumerate(operational_names):
        d_params[name] = ind[op_start + j]

    return d_params
