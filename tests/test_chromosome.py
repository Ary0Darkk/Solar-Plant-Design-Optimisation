"""
Tests for chromosome generation, DEAP types, and serialization.
"""

from deap import base, creator

from solar_opt.chromosome import (
    build_gene_generator,
    deserialize_pop,
    init_deap_types,
    serialize_pop,
)


def test_init_deap_types():
    init_deap_types()
    assert hasattr(creator, "FitnessMax")
    assert hasattr(creator, "Individual")


def test_gene_generator_dimensions():
    d_lb = [8000, 10, 2, 4, 50]
    d_ub = [12000, 25, 10, 8, 200]
    op_lb = [2, 300, 250]
    op_ub = [12, 380, 300]
    gen_fn = build_gene_generator(d_lb, d_ub, op_lb, op_ub, num_days=7)

    ind = gen_fn()
    assert len(ind) == 5 + 7 * 3  # 26 genes
    # Verify design bounds
    for i in range(5):
        assert d_lb[i] <= ind[i] <= d_ub[i]
    # Verify operational bounds
    for d in range(7):
        for j in range(3):
            gene = ind[5 + d * 3 + j]
            assert op_lb[j] <= gene <= op_ub[j]


def test_serialize_and_deserialize_population():
    init_deap_types()
    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual([1, 2, 3]))

    pop = [toolbox.individual(), toolbox.individual()]
    pop[0].fitness.values = (150.0,)
    pop[1].fitness.values = (250.0,)

    serialized = serialize_pop(pop)
    assert len(serialized) == 2
    assert serialized[0] == ([1, 2, 3], (150.0,))

    reconstructed = deserialize_pop(serialized, toolbox)
    assert len(reconstructed) == 2
    assert reconstructed[0] == [1, 2, 3]
    assert reconstructed[0].fitness.values == (150.0,)
