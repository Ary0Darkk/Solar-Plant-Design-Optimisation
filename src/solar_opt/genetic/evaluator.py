"""
Parallel evaluation of candidate populations and seasonal fitness aggregation.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..economic.objective import calculate_daily_objective
from ..simulation.pysam_runner import run_simulation
from ..simulation.weather import get_season_days

logger = logging.getLogger("NYS_Optimisation")


def run_one_simulation_task(task_args: Tuple[int, int, Dict[str, Any], int, float, float]) -> Tuple[int, int, Optional[float], bool]:
    """
    Worker task executing a single 24-hour PySAM simulation and daily objective evaluation.

    Args:
        task_args: (ind_index, local_day_index, overrides, day_index, min_price, max_price)

    Returns:
        (ind_index, local_day_index, daily_objective or None, is_penalized: bool)
    """
    ind_idx, local_day_idx, overrides, day_index, min_price, max_price = task_args

    sim_result, penalty_applied = run_simulation(overrides=overrides, day_index=day_index)

    if penalty_applied or sim_result is None:
        return ind_idx, local_day_idx, None, True

    try:
        daily_obj = calculate_daily_objective(
            hourly_energy=sim_result["hourly_energy"],
            field_htf_pump_power=sim_result["field_htf_pump_power"],
            pc_htf_pump_power=sim_result["pc_htf_pump_power"],
            field_collector_tracking_power=sim_result["field_collector_tracking_power"],
            pc_startup_thermal_power=sim_result["pc_startup_thermal_power"],
            field_piping_thermal_loss=sim_result["field_piping_thermal_loss"],
            receiver_thermal_loss=sim_result["receiver_thermal_loss"],
            day_index=day_index,
            min_price=min_price,
            max_price=max_price,
        )
        return ind_idx, local_day_idx, daily_obj, False

    except Exception as exc:
        logger.warning(
            "Objective evaluation failed for individual %d, day %d: %s",
            ind_idx,
            day_index,
            exc,
        )
        return ind_idx, local_day_idx, None, True


def evaluate_batch(
    batch: Sequence[Any],
    local_pool: Any,
    season: str,
    design_vars: Sequence[str],
    operational_vars: Sequence[str],
    penalty_value: float = -1e13,
    min_price: float = 3.0,
    max_price: float = 8.0,
    rank: int = 0,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Evaluate a batch of candidate individuals across all representative days for the season.

    Returns:
        (evaluated_individuals, {"successful": count, "penalized": count, "total": count})
    """
    season_days = get_season_days(season)
    num_days = len(season_days)
    num_design = len(design_vars)
    num_operational = len(operational_vars)

    tasks: List[Tuple[int, int, Dict[str, Any], int, float, float]] = []

    for ind_idx, ind in enumerate(batch):
        # Extract 5 season-wide design parameters
        design_overrides = {name: ind[i] for i, name in enumerate(design_vars)}

        # Extract 3 daily operational parameters per representative day
        for day_info in season_days:
            local_idx = day_info["local_index"]
            day_offset = num_design + local_idx * num_operational
            daily_operational = {
                name: ind[day_offset + op_idx]
                for op_idx, name in enumerate(operational_vars)
            }

            combined_overrides = {**design_overrides, **daily_operational}
            tasks.append(
                (
                    ind_idx,
                    local_idx,
                    combined_overrides,
                    day_info["day_index"],
                    min_price,
                    max_price,
                )
            )

    logger.debug("[Rank %d] Executing %d simulation tasks for %d individuals", rank, len(tasks), len(batch))

    # Run tasks across multiprocessing pool
    chunksize = max(1, len(tasks) // (getattr(local_pool, "_processes", 4) * 4))
    task_results = list(local_pool.imap_unordered(run_one_simulation_task, tasks, chunksize=chunksize))

    # Aggregate results per individual
    results_by_ind: Dict[int, List[Tuple[int, Optional[float], bool]]] = {i: [] for i in range(len(batch))}
    for ind_idx, local_day_idx, daily_obj, penalized in task_results:
        results_by_ind[ind_idx].append((local_day_idx, daily_obj, penalized))

    successful_count = 0
    penalized_count = 0

    for ind_idx, ind in enumerate(batch):
        day_results = results_by_ind[ind_idx]
        any_penalized = any(pen for _, _, pen in day_results)

        if any_penalized or len(day_results) < num_days:
            ind.fitness.values = (penalty_value,)
            penalized_count += len(day_results)
        else:
            total_fitness = sum(obj for _, obj, _ in day_results if obj is not None)
            ind.fitness.values = (float(total_fitness),)
            successful_count += len(day_results)

    stats = {
        "successful": successful_count,
        "penalized": penalized_count,
        "total": successful_count + penalized_count,
    }

    return list(batch), stats
