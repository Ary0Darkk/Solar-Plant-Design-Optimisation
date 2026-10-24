"""
Unified Command-Line Interface (CLI) Entry Point.
"""

import argparse
import os
from datetime import datetime
from typing import Any, List, Optional

import mlflow

from .config import get_pipeline_config
from .engine import run_optimization, worker_event_loop
from .parallel import get_communicator
from .tracker import init_tracking, log_run_params


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line flags."""
    parser = argparse.ArgumentParser(
        description="Concentrated Solar Power (CSP) Genetic Algorithm Optimization using PySAM and DEAP."
    )
    parser.add_argument(
        "--season",
        type=str,
        default="summer",
        choices=["winter", "summer", "monsoon", "post"],
        help="Season to optimize (default: summer)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of GA generations override (default: from config, usually 100)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Population size override (default: from config, usually 112)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes per rank (default: auto-detected)",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=None,
        help="Recycle worker process after N simulations (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 41)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="Minimum electricity tariff in INR/kWh (default: 3.0)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Maximum electricity tariff in INR/kWh (default: 8.0)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Scheduler Job ID (auto-detected from PBS_JOBID or SLURM_JOB_ID)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="CSP_Seasonal_GA",
        help="MLflow / DagsHub experiment name (default: CSP_Seasonal_GA)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable remote DagsHub/MLflow experiment tracking",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Execute a single 24-hour PySAM simulation and display results",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=1,
        help="Representative day index (1-7) for single simulation (default: 1)",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log single simulation results as a run in MLflow",
    )

    return parser.parse_args(argv)


def resolve_worker_count(workers_override: Optional[int]) -> int:
    """Detect available CPU cores and allocate appropriate worker concurrency."""
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    if workers_override is not None:
        return max(1, min(workers_override, cpu_count))

    is_hpc = bool(os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"))
    if not is_hpc:
        return min(4, cpu_count)

    return cpu_count


def execute_single_simulation(args: argparse.Namespace, config: Any) -> None:
    """Execute a single 24-hour PySAM simulation and display/log metrics."""
    import numpy as np

    from .objective import compute_daily_objective_value
    from .simulation import get_simulator
    from .tariff import get_shared_tariff
    from .weather import get_season_days

    season_days = get_season_days(args.season, config.seasons)
    day_idx_1based = max(1, min(args.day, len(season_days)))
    day_meta = season_days[day_idx_1based - 1]
    day_index = day_meta["day_index"]

    print("\n" + "=" * 65)
    print("☀️  PySAM Concentrated Solar Power - Single Day Simulation")
    print(
        f"    Season: {args.season.upper()} | Day {day_idx_1based} ({day_meta['month']:02d}-{day_meta['day']:02d}) [Day Index: {day_index}]"
    )
    print("=" * 65)

    sim = get_simulator(config.simulation)
    tariff = get_shared_tariff(config.tariff)

    design_overrides = {
        "specified_total_aperture": 8745,
        "Row_Distance": 23,
        "ColperSCA": 2,
        "W_aperture": 6,
        "L_SCA": 152,
        "m_dot": 4,
        "T_startup": 313,
        "T_shutdown": 292,
    }

    sim_slice, penalized = sim.run_daily_simulation(design_overrides, day_index=day_index)
    if penalized or sim_slice is None:
        print("\n❌ [ERROR] Simulation returned physically unfeasible state or failed constraints.")
        return

    tariff_slice = tariff[day_index * 24 : (day_index + 1) * 24]
    net_profit, gross_rev, par_cost, loss_cost = compute_daily_objective_value(
        sim_slice, tariff_slice, land_area=design_overrides["specified_total_aperture"]
    )

    print("\n📋 Plant Layout & Geometry:")
    print(f"  • Solar Aperture:      {design_overrides['specified_total_aperture']:,.1f} m²")
    print(f"  • Row Pitch:           {design_overrides['Row_Distance']:.1f} m")
    print(f"  • Collectors per SCA:  {design_overrides['ColperSCA']}")
    print(f"  • Aperture Width:      {design_overrides['W_aperture']:.1f} m")
    print(
        f"  • HTF Start / Stop:    {design_overrides['T_startup']}°C / {design_overrides['T_shutdown']}°C"
    )

    print("\n📊 24-Hour Energy & Economic Metrics:")
    print(f"  • Gross Energy Gen:    {float(np.sum(sim_slice.hourly_energy)):.2f} MWh")
    print(f"  • Parasitic Load:      {float(np.sum(sim_slice.hourly_parasitic)):.2f} MWh")
    print(f"  • Thermal Piping Loss: {float(np.sum(sim_slice.hourly_heat_loss)):.2f} MWt")
    print(f"  • Gross Revenue:       ₹{gross_rev:,.2f}")
    print(f"  • Parasitic Cost:      ₹{par_cost:,.2f}")
    print(f"  • Heat Loss Cost:      ₹{loss_cost:,.2f}")
    print(f"  💰 Net Daily Profit:   ₹{net_profit:,.2f}")
    print("=" * 65 + "\n")

    if args.log_mlflow:
        try:
            active = init_tracking(experiment_name=args.experiment)
            if active:
                with mlflow.start_run(run_name=f"PySAM_Sim_{args.season}_day_{args.day}"):
                    mlflow.set_tags(
                        {
                            "engine": "PySAM Physical Trough",
                            "mode": "single_simulation",
                            "season": args.season,
                        }
                    )
                    mlflow.log_params(design_overrides)
                    mlflow.log_metrics(
                        {
                            "daily_net_profit_inr": round(net_profit, 2),
                            "daily_gross_revenue_inr": round(gross_rev, 2),
                            "daily_energy_mwh": round(float(np.sum(sim_slice.hourly_energy)), 2),
                        }
                    )
                    print(f"✅ MLflow Remote Run Logged: {mlflow.get_tracking_uri()}")
            else:
                print("⚠️  Remote MLflow tracking inactive (no credentials or URI provided).")
        except Exception as exc:
            print(f"⚠️  MLflow logging skipped: {exc}")


def main(argv: Optional[List[str]] = None) -> None:
    """Main execution entry point."""
    args = parse_arguments(argv)
    config = get_pipeline_config(args.config)

    if getattr(args, "simulate", False):
        execute_single_simulation(args, config)
        return

    if args.generations is not None:
        config.ga.num_generations = args.generations
    if args.pop_size is not None:
        config.ga.pop_size = args.pop_size
    if args.max_tasks_per_child is not None:
        config.ga.max_tasks_per_child = args.max_tasks_per_child
    if args.seed is not None:
        config.ga.random_seed = args.seed
    if args.min_price is not None:
        config.tariff.min_price = args.min_price
    if args.max_price is not None:
        config.tariff.max_price = args.max_price

    workers = resolve_worker_count(args.workers)
    comm = get_communicator()
    rank = comm.Get_rank()

    if rank == 0:
        if not args.no_tracking:
            init_tracking(experiment_name=args.experiment)

        with (
            mlflow.start_run(
                run_name=f"CSP_GA_{args.season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if not args.no_tracking
            else open(os.devnull)
        ):
            log_run_params(
                {
                    "season": args.season,
                    "pop_size": config.ga.pop_size,
                    "generations": config.ga.num_generations,
                    "workers": workers,
                }
            )
            run_optimization(args.season, config=config, workers=workers, comm=comm)
    else:
        worker_event_loop(comm)
