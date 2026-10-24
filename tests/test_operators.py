"""
Tests for bounded polynomial mutation and SBX crossover.
"""

from solar_opt.config import PipelineConfig
from solar_opt.operators import build_bounded_crossover, build_bounded_mutation


def test_build_bounded_mutation_respects_limits():
    cfg = PipelineConfig()
    mut = build_bounded_mutation(cfg.design, cfg.operational, eta_m=20.0, indpb=1.0, num_days=7)

    ind = [10000.0, 15.0, 4.0, 6.0, 150.0] + [5.0, 350.0, 280.0] * 7
    (mutated,) = mut(list(ind))

    # Verify length
    assert len(mutated) == 26

    # Verify bounds
    for i, (lb, ub) in enumerate(zip(cfg.design.lb, cfg.design.ub)):
        assert lb <= mutated[i] <= ub

    # ColperSCA must be integer
    assert mutated[2] == round(mutated[2])


def test_build_bounded_crossover_respects_limits():
    cfg = PipelineConfig()
    cx = build_bounded_crossover(cfg.design, cfg.operational, eta_c=20.0, num_days=7)

    ind1 = [8500.0, 12.0, 3.0, 5.0, 80.0] + [3.0, 310.0, 260.0] * 7
    ind2 = [11500.0, 22.0, 8.0, 7.0, 180.0] + [10.0, 370.0, 290.0] * 7

    c1, c2 = cx(list(ind1), list(ind2))
    assert len(c1) == 26
    assert len(c2) == 26

    for i, (lb, ub) in enumerate(zip(cfg.design.lb, cfg.design.ub)):
        assert lb <= c1[i] <= ub
        assert lb <= c2[i] <= ub
