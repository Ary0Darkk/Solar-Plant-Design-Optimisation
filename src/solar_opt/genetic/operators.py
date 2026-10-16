"""
Custom genetic operators tailored for integer-bounded CSP design & operation chromosomes.
"""

import random
from typing import Any, Callable, List, Sequence, Tuple


def create_bounded_integer_mutation(
    design_lb: Sequence[int],
    design_ub: Sequence[int],
    operational_lb: Sequence[int],
    operational_ub: Sequence[int],
    num_days: int,
    indpb: float = 0.1,
) -> Callable[[List[int]], Tuple[List[int]]]:
    """
    Factory returning a DEAP-compatible mutation operator that perturbs integer genes
    using Gaussian displacement (sigma = 10% of variable range) and clamps within bounds.
    """
    num_design = len(design_lb)
    num_op = len(operational_lb)

    def _mutate(individual: List[int]) -> Tuple[List[int]]:
        # Mutate design parameters
        for i in range(num_design):
            if random.random() < indpb:
                sigma = 0.1 * (design_ub[i] - design_lb[i])
                mutated = round(individual[i] + random.gauss(0, sigma))
                individual[i] = int(max(design_lb[i], min(design_ub[i], mutated)))

        # Mutate daily operational parameters
        for day in range(num_days):
            start = num_design + day * num_op
            for j in range(num_op):
                if random.random() < indpb:
                    sigma = 0.1 * (operational_ub[j] - operational_lb[j])
                    mutated = round(individual[start + j] + random.gauss(0, sigma))
                    individual[start + j] = int(
                        max(operational_lb[j], min(operational_ub[j], mutated))
                    )

        return (individual,)

    return _mutate
