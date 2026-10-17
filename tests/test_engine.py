"""
Unit tests for core optimization engine and worker task evaluation.
"""

import multiprocessing as mp

import pytest

from optimized_pipeline.chromosome import (
    build_gene_generator,
    init_deap_types,
)
from optimized_pipeline.config import get_pipeline_config
from optimized_pipeline.engine import (
    _eval_single_task,
    evaluate_batch_memory_optimized,
)
from optimized_pipeline.parallel import partition_batches
from optimized_pipeline.weather import get_season_days


def test_partition_batches():
    """Verify that items are properly divided into discrete chunks."""
    items = list(range(25))
    batches = partition_batches(items, batch_size=10)
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5
    assert [x for b in batches for x in b] == items

    with pytest.raises(ValueError):
        partition_batches(items, batch_size=0)


def test_eval_single_task_valid():
    """Verify _eval_single_task produces a scalar objective for valid inputs."""
    summer_days = get_season_days("summer")
    first_day = summer_days[0]  # March 9th: local_index=0, day_index=67

    overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 16.0,
        "W_aperture": 6.0,
        "ColperSCA": 4,
        "L_SCA": 100,
        "m_dot": 5,
        "T_startup": 350,
        "T_shutdown": 280,
    }
    task_args = (0, first_day["local_index"], overrides, first_day["day_index"])
    ind_idx, local_idx, obj, is_pen = _eval_single_task(task_args)

    assert ind_idx == 0
    assert local_idx == first_day["local_index"]
    assert is_pen is False
    assert obj is not None
    assert isinstance(obj, float)


def test_eval_single_task_penalized():
    """Verify _eval_single_task flags invalid/infeasible inputs."""
    summer_days = get_season_days("summer")
    first_day = summer_days[0]  # March 9th: local_index=0, day_index=67

    # Violates vehicular maintenance corridor: Row_Distance (6m) - W_aperture (6m) = 0 < 3.5m
    overrides = {
        "specified_total_aperture": 9000,
        "Row_Distance": 6.0,
        "W_aperture": 6.0,
        "ColperSCA": 4,
        "L_SCA": 100,
        "m_dot": 5,
        "T_startup": 350,
        "T_shutdown": 280,
    }
    task_args = (1, first_day["local_index"], overrides, first_day["day_index"])
    ind_idx, local_idx, obj, is_pen = _eval_single_task(task_args)

    assert ind_idx == 1
    assert local_idx == first_day["local_index"]
    assert is_pen is True
    assert obj is None


def test_evaluate_batch_memory_optimized_pool():
    """Verify evaluate_batch_memory_optimized evaluates individuals using a multiprocessing Pool."""
    from deap import base, creator, tools

    init_deap_types()
    config = get_pipeline_config()
    num_days = 7

    generator = build_gene_generator(
        design_lb=config.design.lb,
        design_ub=config.design.ub,
        operational_lb=config.operational.lb,
        operational_ub=config.operational.ub,
        num_days=num_days,
    )
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generator)

    ind = toolbox.individual()
    batch = [ind]

    ctx = mp.get_context()
    with ctx.Pool(processes=2) as pool:
        evaluated_pop, stats = evaluate_batch_memory_optimized(
            batch=batch,
            pool=pool,
            season="winter",
            config=config,
            rank=0,
        )

    assert len(evaluated_pop) == 1
    assert hasattr(evaluated_pop[0], "fitness")
    assert len(evaluated_pop[0].fitness.values) == 1
    assert stats["total"] == 7
