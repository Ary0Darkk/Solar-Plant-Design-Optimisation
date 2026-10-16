"""
Unit tests for genetic algorithm primitives, operators, and serialization.
"""

from deap import base, creator, tools

from solar_opt.config import DEFAULT_DESIGN_BOUNDS, DEFAULT_OPERATIONAL_BOUNDS
from solar_opt.genetic.checkpoint import compute_checkpoint_key
from solar_opt.genetic.chromosome import (
    create_individual_generator,
    deserialize_population,
    serialize_population,
    setup_deap_primitives,
)
from solar_opt.genetic.operators import create_bounded_integer_mutation


def test_individual_generation():
    """Verify that generated individuals strictly respect gene length and bounds."""
    setup_deap_primitives()
    num_days = 7
    generator = create_individual_generator(
        design_lb=DEFAULT_DESIGN_BOUNDS.lb,
        design_ub=DEFAULT_DESIGN_BOUNDS.ub,
        operational_lb=DEFAULT_OPERATIONAL_BOUNDS.lb,
        operational_ub=DEFAULT_OPERATIONAL_BOUNDS.ub,
        num_days=num_days,
    )

    ind = generator()
    expected_len = 5 + 3 * num_days
    assert len(ind) == expected_len

    # Check design bounds
    for i in range(5):
        assert DEFAULT_DESIGN_BOUNDS.lb[i] <= ind[i] <= DEFAULT_DESIGN_BOUNDS.ub[i]

    # Check operational bounds
    for d in range(num_days):
        start = 5 + d * 3
        for j in range(3):
            assert DEFAULT_OPERATIONAL_BOUNDS.lb[j] <= ind[start + j] <= DEFAULT_OPERATIONAL_BOUNDS.ub[j]


def test_custom_mutation_bounds():
    """Verify that mutation preserves integer types and never violates bounds."""
    setup_deap_primitives()
    num_days = 7
    mutator = create_bounded_integer_mutation(
        design_lb=DEFAULT_DESIGN_BOUNDS.lb,
        design_ub=DEFAULT_DESIGN_BOUNDS.ub,
        operational_lb=DEFAULT_OPERATIONAL_BOUNDS.lb,
        operational_ub=DEFAULT_OPERATIONAL_BOUNDS.ub,
        num_days=num_days,
        indpb=1.0,  # force mutate all genes
    )

    generator = create_individual_generator(
        design_lb=DEFAULT_DESIGN_BOUNDS.lb,
        design_ub=DEFAULT_DESIGN_BOUNDS.ub,
        operational_lb=DEFAULT_OPERATIONAL_BOUNDS.lb,
        operational_ub=DEFAULT_OPERATIONAL_BOUNDS.ub,
        num_days=num_days,
    )

    ind = generator()
    (mutated,) = mutator(ind)

    # Check that mutated individual is within bounds
    for i in range(5):
        assert DEFAULT_DESIGN_BOUNDS.lb[i] <= mutated[i] <= DEFAULT_DESIGN_BOUNDS.ub[i]
        assert isinstance(mutated[i], int)

    for d in range(num_days):
        start = 5 + d * 3
        for j in range(3):
            assert DEFAULT_OPERATIONAL_BOUNDS.lb[j] <= mutated[start + j] <= DEFAULT_OPERATIONAL_BOUNDS.ub[j]
            assert isinstance(mutated[start + j], int)


def test_checkpoint_key_deterministic():
    """Verify that checkpoint key is consistent and deterministic."""
    k1 = compute_checkpoint_key(
        season="winter",
        design_vars=DEFAULT_DESIGN_BOUNDS.names,
        operational_vars=DEFAULT_OPERATIONAL_BOUNDS.names,
        design_types=DEFAULT_DESIGN_BOUNDS.types,
        operational_types=DEFAULT_OPERATIONAL_BOUNDS.types,
        design_lb=DEFAULT_DESIGN_BOUNDS.lb,
        design_ub=DEFAULT_DESIGN_BOUNDS.ub,
        operational_lb=DEFAULT_OPERATIONAL_BOUNDS.lb,
        operational_ub=DEFAULT_OPERATIONAL_BOUNDS.ub,
        pop_size=112,
        num_generations=100,
        cxpb=0.8,
        mutpb=0.2,
        indpb=0.1,
    )
    k2 = compute_checkpoint_key(
        season="winter",
        design_vars=DEFAULT_DESIGN_BOUNDS.names,
        operational_vars=DEFAULT_OPERATIONAL_BOUNDS.names,
        design_types=DEFAULT_DESIGN_BOUNDS.types,
        operational_types=DEFAULT_OPERATIONAL_BOUNDS.types,
        design_lb=DEFAULT_DESIGN_BOUNDS.lb,
        design_ub=DEFAULT_DESIGN_BOUNDS.ub,
        operational_lb=DEFAULT_OPERATIONAL_BOUNDS.lb,
        operational_ub=DEFAULT_OPERATIONAL_BOUNDS.ub,
        pop_size=112,
        num_generations=100,
        cxpb=0.8,
        mutpb=0.2,
        indpb=0.1,
    )
    assert k1 == k2
    assert len(k1) == 12


def test_serialization_roundtrip():
    """Verify serialize_population and deserialize_population preserve individuals."""
    setup_deap_primitives()
    toolbox = base.Toolbox()
    generator = create_individual_generator(
        design_lb=DEFAULT_DESIGN_BOUNDS.lb,
        design_ub=DEFAULT_DESIGN_BOUNDS.ub,
        operational_lb=DEFAULT_OPERATIONAL_BOUNDS.lb,
        operational_ub=DEFAULT_OPERATIONAL_BOUNDS.ub,
        num_days=2,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, generator)

    ind1 = toolbox.individual()
    ind1.fitness.values = (12345.67,)
    ind2 = toolbox.individual()
    ind2.fitness.values = (-9876.54,)
    original_pop = [ind1, ind2]

    serialized = serialize_population(original_pop)
    recovered = deserialize_population(serialized, toolbox)

    assert len(recovered) == 2
    assert recovered[0] == original_pop[0]
    assert recovered[0].fitness.values == (12345.67,)
    assert recovered[1] == original_pop[1]
    assert recovered[1].fitness.values == (-9876.54,)
