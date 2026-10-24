"""
Tests for 26-gene chromosome generator, bounds, and DEAP types.
"""

from solar_opt.chromosome import (
    build_toolbox,
    decode_individual,
    deserialize_pop,
    generate_gene_vector,
    init_deap_types,
    serialize_pop,
)
from solar_opt.config import PipelineConfig


def test_init_deap_types():
    init_deap_types()
    from deap import creator

    assert hasattr(creator, "FitnessMax")
    assert hasattr(creator, "Individual")


def test_gene_generator_dimensions():
    cfg = PipelineConfig()
    genes = generate_gene_vector(cfg.design, cfg.operational, num_days=7)
    assert len(genes) == 26
    # Check design bounds
    for i, (lb, ub) in enumerate(zip(cfg.design.lb, cfg.design.ub)):
        assert lb <= genes[i] <= ub

    # ColperSCA must be an integer count
    colpersca = genes[2]
    assert colpersca == round(colpersca)


def test_serialize_and_deserialize_population():
    cfg = PipelineConfig()
    toolbox = build_toolbox(cfg.design, cfg.operational, num_days=7)
    pop = toolbox.population(n=10)
    for ind in pop:
        ind.fitness.values = (1500.0,)

    serialized = serialize_pop(pop)
    assert len(serialized) == 10
    assert serialized[0][1] == 1500.0

    restored = deserialize_pop(serialized)
    assert len(restored) == 10
    assert restored[0].fitness.values == (1500.0,)
    assert list(restored[0]) == list(pop[0])


def test_decode_individual():
    cfg = PipelineConfig()
    ind = [10000.0, 15.0, 4.0, 6.0, 150.0] + [5.0, 350.0, 280.0] * 7
    decoded = decode_individual(ind, cfg.design.names, cfg.operational.names, day_offset=0)
    assert decoded["specified_total_aperture"] == 10000.0
    assert decoded["Row_Distance"] == 15.0
    assert decoded["ColperSCA"] == 4
    assert decoded["W_aperture"] == 6.0
    assert decoded["L_SCA"] == 150.0
    assert decoded["m_dot"] == 5.0
    assert decoded["T_startup"] == 350.0
    assert decoded["T_shutdown"] == 280.0
