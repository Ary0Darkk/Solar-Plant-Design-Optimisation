"""
Tests for bounded genetic mutation operator.
"""

from solar_opt.operators import build_bounded_mutation


def test_build_bounded_mutation_respects_limits():
    d_lb = [8000, 10, 2, 4, 50]
    d_ub = [12000, 25, 10, 8, 200]
    op_lb = [2, 300, 250]
    op_ub = [12, 380, 300]

    # Force 100% mutation probability to stress test bounding
    mut_fn = build_bounded_mutation(d_lb, d_ub, op_lb, op_ub, num_days=7, indpb=1.0)

    # Initial individual at upper bounds
    ind = [12000, 25, 10, 8, 200] + [12, 380, 300] * 7
    for _ in range(50):
        (mutated,) = mut_fn(list(ind))
        # Ensure design genes remain strictly bounded
        for i in range(5):
            assert d_lb[i] <= mutated[i] <= d_ub[i]
        # Ensure operational genes remain strictly bounded
        for d in range(7):
            for j in range(3):
                val = mutated[5 + d * 3 + j]
                assert op_lb[j] <= val <= op_ub[j]
