import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime

# Ensure Windows consoles encode UTF-8 (prevent MLflow emoji crashes)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import dagshub
import mlflow

try:
    from mpi4py import MPI
except Exception:
    MPI = None


# ---------------------------------------------------------
# Standalone communicator fallback
# ---------------------------------------------------------


class StandaloneComm:
    """Mock MPI communicator for running without MPI in single-node environments."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, data, root: int = 0):
        return data

    def send(self, data, dest: int, tag: int = 0) -> None:
        pass

    def recv(self, source: int = 0, tag: int = 0):
        return None


from config import CONFIG
from deap_ga_optimiser import (
    run_deap_ga_optimisation,
    worker_loop,
)
from setup_custom_logger import setup_custom_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Season-wise CSP genetic algorithm optimisation"
    )

    parser.add_argument(
        "--season",
        default="summer",
        choices=["winter", "summer", "monsoon", "post"],
        help="Seasonal optimisation case to run (default: summer)",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Optional override for number of generations",
    )

    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Optional override for population size",
    )

    return parser.parse_args()


def log_experiment_parameters(season, mpi_size):
    design = CONFIG["design"]
    operational = CONFIG["operational"]

    # Actual multiprocessing workers configured per MPI rank
    workers_per_rank = int(os.environ.get("LOCAL_WORKERS", "4"))

    mlflow.log_params(
        {
            "season": season,
            "random_seed": CONFIG.get("random_seed"),
            "penalty": CONFIG.get("penalty"),
            "population_size": CONFIG.get("pop_size"),
            "num_generations": CONFIG.get("num_generations"),
            "cxpb": CONFIG.get("cxpb"),
            "mutpb": CONFIG.get("mutpb"),
            "indpb": CONFIG.get("indpb"),
            "tournament_size": CONFIG.get("tournament_size"),
            "hall_of_fame_size": CONFIG.get("hall_of_fame_size"),
            "num_days": len(CONFIG["SEASONS"][season]),
            # MPI configuration
            "mpi_ranks": mpi_size,
            # Actual configured parallelism
            "workers_per_rank": workers_per_rank,
            "configured_total_workers": mpi_size * workers_per_rank,
            # Intended HPC allocation (reference only)
            "hpc_cores_per_node": 112,
            "design_variables": json.dumps(design["overrides"]),
            "operational_variables": json.dumps(operational["overrides"]),
            "design_lower_bounds": json.dumps(design["lb"]),
            "design_upper_bounds": json.dumps(design["ub"]),
            "operational_lower_bounds": json.dumps(operational["lb"]),
            "operational_upper_bounds": json.dumps(operational["ub"]),
            "electricity_min_price": CONFIG.get("min_price", 3.0),
            "electricity_max_price": CONFIG.get("max_price", 8.0),
        }
    )


def main():
    args = parse_args()
    season = args.season

    if args.generations is not None:
        CONFIG["num_generations"] = args.generations
    if args.pop_size is not None:
        CONFIG["pop_size"] = args.pop_size
        if CONFIG.get("tournament_size", 7) > args.pop_size:
            CONFIG["tournament_size"] = max(2, args.pop_size // 2)
        if CONFIG.get("hall_of_fame_size", 5) > args.pop_size:
            CONFIG["hall_of_fame_size"] = max(1, args.pop_size // 2)

    # ---------------------------------------------------------
    # MPI / Standalone Communicator
    # ---------------------------------------------------------

    if MPI is not None:
        comm = MPI.COMM_WORLD
    else:
        comm = StandaloneComm()

    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------------------------------------------------------
    # Synchronized Rank Logger
    # ---------------------------------------------------------

    # Synchronize run timestamp from Rank 0 to ensure matching log sessions
    session_timestamp = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if rank == 0 else None
    )
    session_timestamp = comm.bcast(session_timestamp, root=0)

    logger = setup_custom_logger(rank=rank, timestamp=session_timestamp)
    logger.info("NYS-Optimisation started!")

    logger.info(f"[Rank {rank}] Starting optimisation for season: {season}")

    # =========================================================
    # MASTER
    # =========================================================

    if rank == 0:
        # -----------------------------------------------------
        # MLflow / DagsHub configuration
        # -----------------------------------------------------

        try:
            dagshub.init(
                repo_owner="aryanvj787",
                repo_name="NYS-Design-Optimisation-using-PySAM",
                mlflow=True,
            )
        except Exception as exc:
            logger.warning(
                "Could not initialize DagsHub (%s). Falling back to local MLflow.",
                exc,
            )

        mlflow.set_experiment("CSP_Seasonal_GA")

        with mlflow.start_run(run_name=f"GA_{season}"):
            # Tags = metadata describing the run
            mlflow.set_tags(
                {
                    "project": "CSP Plant Optimisation",
                    "season": season,
                    "scheduler": "PBS",
                    "parallelization": "MPI + multiprocessing",
                    "algorithm": "DEAP Genetic Algorithm",
                    "simulation": "PySAM",
                }
            )

            # Parameters = fixed configuration
            log_experiment_parameters(
                season=season,
                mpi_size=size,
            )

            logger.info(
                f"[Rank 0] MLflow run started: {mlflow.active_run().info.run_id}"
            )

            # -------------------------------------------------
            # Local multiprocessing pool
            # -------------------------------------------------

            num_workers = int(os.environ.get("LOCAL_WORKERS", "4"))
            mp_context = mp.get_context()

            try:
                # Periodic recycling (maxtasksperchild) releases PySAM C-memory
                with mp_context.Pool(
                    processes=num_workers,
                    maxtasksperchild=100,
                ) as local_pool:
                    run_deap_ga_optimisation(
                        season=season,
                        local_pool=local_pool,
                        comm=comm,
                        rank=rank,
                        mpi_size=size,
                        logger=logger,
                    )

            finally:
                # Tell remote MPI ranks to stop
                for worker in range(1, size):
                    comm.send(None, dest=worker, tag=1)

            logger.info("[Rank 0] MLflow run completed")

    # =========================================================
    # WORKERS
    # =========================================================

    else:
        worker_loop(season=season, comm=comm, rank=rank, logger=logger)

    logger.info("Optimisation Completed!")


if __name__ == "__main__":
    main()
