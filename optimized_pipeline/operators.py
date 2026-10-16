"""
In-place bounded genetic operators optimized for integer chromosomes.
"""

import random
from typing import Callable, List, Sequence, Tuple


def build_bounded_mutation(
    design_lb: Sequence[int],
    design_ub: Sequence[int],
    operational_lb: Sequence[int],
    operational_ub: Sequence[int],
    num_days: int,
    indpb: float = 0.1,
) -> Callable[[List[int]], Tuple[List[int]]]:
    """
    In-place mutation operator displacing integer genes with Gaussian noise
    (sigma = 10% of parameter span) and clamping strictly within [lb, ub].
    """
    n_design = len(design_lb)
    n_op = len(operational_lb)

    def _mutate(ind: List[int]) -> Tuple[List[int]]:
        # 1. Mutate design parameters
        for i in range(n_design):
            if random.random() < indpb:
                sigma = 0.1 * (design_ub[i] - design_lb[i])
                val = round(ind[i] + random.gauss(0, sigma))
                ind[i] = int(max(design_lb[i], min(design_ub[i], val)))

        # 2. Mutate operational parameters across each representative day
        for d in range(num_days):
            offset = n_design + d * n_op
            for j in range(n_op):
                idx = offset + j
                if random.random() < indpb:
                    sigma = 0.1 * (operational_ub[j] - operational_lb[j])
                    val = round(ind[idx] + random.gauss(0, sigma))
                    ind[idx] = int(max(operational_lb[j], min(operational_ub[j], val)))

        return (ind,)

    return _mutate
