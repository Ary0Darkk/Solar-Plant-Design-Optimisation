#!/usr/bin/env python3
"""
Utility script to list, clean, or delete runs inside a specific MLflow experiment on remote DagsHub.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

DEFAULT_EXPERIMENT = "CSP_Seasonal_GA"


def list_experiments(tracking_uri: str, username: str = "aryanvj787", token: str = "") -> int:
    """List all experiments and their run counts."""
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ["DAGSHUB_USER_TOKEN"] = token

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        experiments = client.search_experiments()
    except Exception as exc:
        print(f"Error querying experiments ({tracking_uri}): {exc}")
        return 1

    print(f"\nConnected to: {tracking_uri}")
    print(f"Found {len(experiments)} experiment(s):\n")
    print(f"{'ID':<6} {'Experiment Name':<30} {'Stage':<10} {'Runs'}")
    print("-" * 60)
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
        count = getattr(runs, "total", len(runs))
        print(f"{exp.experiment_id:<6} {exp.name:<30} {exp.lifecycle_stage:<10} {count}")
    print()
    return 0


def delete_runs(
    tracking_uri: str,
    experiment_name: str,
    username: str = "aryanvj787",
    token: str = "",
    dry_run: bool = False,
) -> int:
    """Delete all runs inside an experiment."""
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ["DAGSHUB_USER_TOKEN"] = token

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment '{experiment_name}' not found at {tracking_uri}")
        return 1

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
    print(f"\nExperiment: {experiment_name} (ID: {exp.experiment_id})")
    print(f"Total runs found: {len(runs)}")

    if not runs:
        print("No runs to delete.")
        return 0

    if dry_run:
        print(f"[DRY-RUN] Would delete {len(runs)} run(s).")
        return 0

    deleted = 0
    for r in runs:
        try:
            client.delete_run(r.info.run_id)
            deleted += 1
        except Exception as exc:
            print(f"  Failed to delete run {r.info.run_id}: {exc}")

    print(f"Successfully deleted {deleted}/{len(runs)} runs.")
    return 0


# Alias for backwards compatibility
clean_runs = delete_runs


def main():
    parser = argparse.ArgumentParser(description="MLflow Run & Experiment Manager")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--delete", action="store_true", help="Delete runs in experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate deletion")
    parser.add_argument("--token", type=str, default="", help="DagsHub Token override")
    parser.add_argument("--uri", type=str, default="", help="Custom MLflow Tracking URI")

    args = parser.parse_args()

    token = (
        args.token
        or os.environ.get("DAGSHUB_USER_TOKEN")
        or os.environ.get("MLFLOW_TRACKING_PASSWORD")
        or ""
    )
    repo = os.environ.get("DAGSHUB_REPO_NAME", "NYS-Design-Optimisation-using-PySAM")

    if args.uri:
        uri = args.uri
    else:
        uri = f"https://dagshub.com/{repo}.mlflow"

    if args.list or not args.delete:
        sys.exit(list_experiments(uri, token=token))
    else:
        sys.exit(delete_runs(uri, args.experiment, token=token, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
