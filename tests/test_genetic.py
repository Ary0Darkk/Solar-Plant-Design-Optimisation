"""
Unit tests for genetic algorithm primitives, operators, and serialization.
"""

from deap import base, creator, tools

from optimized_pipeline.checkpoint import compute_signature
from optimized_pipeline.chromosome import (
    build_gene_generator,
    deserialize_pop,
    init_deap_types,
    serialize_pop,
)
from optimized_pipeline.config import DESIGN_BOUNDS, OPERATIONAL_BOUNDS
from optimized_pipeline.operators import build_bounded_mutation


def test_individual_generation():
    """Verify that generated individuals strictly respect gene length and bounds."""
    init_deap_types()
    num_days = 7
    generator = build_gene_generator(
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        num_days=num_days,
    )

    ind = generator()
    expected_len = 5 + 3 * num_days
    assert len(ind) == expected_len

    # Check design bounds
    for i in range(5):
        assert DESIGN_BOUNDS.lb[i] <= ind[i] <= DESIGN_BOUNDS.ub[i]

    # Check operational bounds
    for d in range(num_days):
        start = 5 + d * 3
        for j in range(3):
            assert OPERATIONAL_BOUNDS.lb[j] <= ind[start + j] <= OPERATIONAL_BOUNDS.ub[j]


def test_custom_mutation_bounds():
    """Verify that mutation preserves integer types and never violates bounds."""
    init_deap_types()
    num_days = 7
    mutator = build_bounded_mutation(
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        num_days=num_days,
        indpb=1.0,  # force mutate all genes
    )

    generator = build_gene_generator(
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        num_days=num_days,
    )

    ind = generator()
    (mutated,) = mutator(ind)

    # Check that mutated individual is within bounds
    for i in range(5):
        assert DESIGN_BOUNDS.lb[i] <= mutated[i] <= DESIGN_BOUNDS.ub[i]
        assert isinstance(mutated[i], int)

    for d in range(num_days):
        start = 5 + d * 3
        for j in range(3):
            assert OPERATIONAL_BOUNDS.lb[j] <= mutated[start + j] <= OPERATIONAL_BOUNDS.ub[j]
            assert isinstance(mutated[start + j], int)


def test_checkpoint_key_deterministic():
    """Verify that checkpoint key is consistent and deterministic."""
    sig1 = compute_signature(
        season="winter",
        design_names=DESIGN_BOUNDS.names,
        operational_names=OPERATIONAL_BOUNDS.names,
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        pop_size=112,
        num_generations=100,
        cxpb=0.8,
        mutpb=0.2,
        indpb=0.1,
    )
    sig2 = compute_signature(
        season="winter",
        design_names=DESIGN_BOUNDS.names,
        operational_names=OPERATIONAL_BOUNDS.names,
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        pop_size=112,
        num_generations=100,
        cxpb=0.8,
        mutpb=0.2,
        indpb=0.1,
    )
    assert sig1 == sig2
    assert len(sig1) == 12


def test_serialization_roundtrip():
    """Verify serialize_pop and deserialize_pop preserve individuals."""
    init_deap_types()
    toolbox = base.Toolbox()
    generator = build_gene_generator(
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        num_days=2,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, generator)

    ind1 = toolbox.individual()
    ind1.fitness.values = (12345.67,)
    ind2 = toolbox.individual()
    ind2.fitness.values = (-9876.54,)
    original_pop = [ind1, ind2]

    serialized = serialize_pop(original_pop)
    recovered = deserialize_pop(serialized, toolbox)

    assert len(recovered) == 2
    assert recovered[0] == original_pop[0]
    assert recovered[0].fitness.values == (12345.67,)
    assert recovered[1] == original_pop[1]
    assert recovered[1].fitness.values == (-9876.54,)
