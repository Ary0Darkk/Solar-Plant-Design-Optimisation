import os
import argparse
import multiprocessing as mp
import json

import mlflow
import dagshub
from mpi4py import MPI


from setup_custom_logger import setup_custom_logger

from deap_ga_optimiser import (
    run_deap_ga_optimisation,
    worker_loop,
)

from config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Season-wise CSP genetic algorithm optimisation"
    )

    parser.add_argument(
        "--season",
        required=True,
        choices=["winter", "summer", "monsoon", "post"],
        help="Seasonal optimisation case to run",
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
        }
    )


def main():
    args = parse_args()
    season = args.season

    # ---------------------------------------------------------
    # MPI
    # ---------------------------------------------------------

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------------------------------------------------------
    # Race-safe Logger
    # ---------------------------------------------------------

    logger = setup_custom_logger(rank=rank)
    logger.info("NYS-Optimisation started!")

    logger.info(f"[Rank {rank}] Starting optimisation for season: {season}")

    # =========================================================
    # MASTER
    # =========================================================

    if rank == 0:
        # -----------------------------------------------------
        # MLflow configuration
        # -----------------------------------------------------

        dagshub.init(
            repo_owner="aryanvj787",
            repo_name="NYS-Design-Optimisation-using-PySAM",
            mlflow=True,
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
            local_pool = mp.Pool(processes=num_workers)

            try:
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

                local_pool.close()
                local_pool.join()

            logger.info("[Rank 0] MLflow run completed")

    # =========================================================
    # WORKERS
    # =========================================================

    else:
        worker_loop(season=season, comm=comm, rank=rank)

    logger.info("Optimisation Completed!")


if __name__ == "__main__":
    main()
