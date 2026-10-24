"""
Genetic Algorithm Evolutionary Operators (Bounded Mutation & Crossover).
"""

import random
from typing import Callable, List, Tuple

from .config import VariableBounds


def build_bounded_mutation(
    design_bounds: VariableBounds,
    operational_bounds: VariableBounds,
    eta_m: float = 20.0,
    indpb: float = 0.1,
    num_days: int = 7,
) -> Callable:
    """
    Polynomial bounded mutation maintaining strict physical limits.
    """
    all_lb = list(design_bounds.lb)
    all_ub = list(design_bounds.ub)
    all_types = list(design_bounds.types)

    for _ in range(num_days):
        all_lb.extend(operational_bounds.lb)
        all_ub.extend(operational_bounds.ub)
        all_types.extend(operational_bounds.types)

    def mutate(individual: List[float]) -> Tuple[List[float]]:
        for i in range(len(individual)):
            if random.random() <= indpb:
                x = individual[i]
                xl = all_lb[i]
                xu = all_ub[i]
                delta_1 = (x - xl) / (xu - xl + 1e-12)
                delta_2 = (xu - x) / (xu - xl + 1e-12)
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val**mut_pow

                x = x + delta_q * (xu - xl)
                x = max(xl, min(xu, x))

                if all_types[i] is int:
                    x = float(round(x))

                individual[i] = x
        return (individual,)

    return mutate


def build_bounded_crossover(
    design_bounds: VariableBounds,
    operational_bounds: VariableBounds,
    eta_c: float = 20.0,
    num_days: int = 7,
) -> Callable:
    """
    Simulated Binary Crossover (SBX) with bound enforcement.
    """
    all_lb = list(design_bounds.lb)
    all_ub = list(design_bounds.ub)
    all_types = list(design_bounds.types)

    for _ in range(num_days):
        all_lb.extend(operational_bounds.lb)
        all_ub.extend(operational_bounds.ub)
        all_types.extend(operational_bounds.types)

    def crossover(ind1: List[float], ind2: List[float]) -> Tuple[List[float], List[float]]:
        for i in range(len(ind1)):
            if random.random() <= 0.5:
                y1 = min(ind1[i], ind2[i])
                y2 = max(ind1[i], ind2[i])
                xl = all_lb[i]
                xu = all_ub[i]

                if abs(y1 - y2) > 1e-12:
                    rand = random.random()
                    beta = 1.0 + (2.0 * (y1 - xl) / (y2 - y1 + 1e-12))
                    alpha = 2.0 - beta ** -(eta_c + 1.0)
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                    beta = 1.0 + (2.0 * (xu - y2) / (y2 - y1 + 1e-12))
                    alpha = 2.0 - beta ** -(eta_c + 1.0)
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                    c1 = max(xl, min(xu, c1))
                    c2 = max(xl, min(xu, c2))

                    if all_types[i] is int:
                        c1 = float(round(c1))
                        c2 = float(round(c2))

                    ind1[i] = c1
                    ind2[i] = c2
        return ind1, ind2

    return crossover
