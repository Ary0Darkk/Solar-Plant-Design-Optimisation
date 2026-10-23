#!/usr/bin/env python3
"""
Utility script to clean or delete runs inside a specific MLflow experiment
(either on local SQLite or remote DagsHub).

Usage:
    # Delete local runs in CSP_Plant_Optimization:
    uv run python clean_mlflow_runs.py --local

    # Delete remote DagsHub runs:
    uv run python clean_mlflow_runs.py --token <YOUR_DAGSHUB_TOKEN>
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

DEFAULT_EXPERIMENT = "CSP_Plant_Optimization"


def clean_runs(
    tracking_uri: str,
    experiment_name: str,
    username: str = "aryanvj787",
    token: str = "",
) -> int:
    """Delete all runs inside an MLflow experiment."""
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ["DAGSHUB_USER_TOKEN"] = token

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        exp = client.get_experiment_by_name(experiment_name)
    except Exception as exc:
        print(f"Error connecting to MLflow tracking server ({tracking_uri}): {exc}")
        return 1

    if exp is None:
        print(f"Experiment '{experiment_name}' not found on {tracking_uri}.")
        return 1

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    if not runs:
        print(f"No active runs found in experiment '{experiment_name}' (ID: {exp.experiment_id}).")
        return 0

    print(f"Found {len(runs)} run(s) in experiment '{experiment_name}' (ID: {exp.experiment_id}).")
    deleted_count = 0
    for r in runs:
        try:
            client.delete_run(r.info.run_id)
            print(f"  [DELETED] Run ID: {r.info.run_id} ({r.info.run_name})")
            deleted_count += 1
        except Exception as err:
            print(f"  [ERROR] Failed to delete run {r.info.run_id}: {err}")

    print(f"Successfully deleted {deleted_count}/{len(runs)} runs.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Delete runs inside a specific MLflow experiment.")
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment name to clean (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Clean runs in local results/mlflow.db SQLite database",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="DagsHub Access Token (if omitted, reads from DAGSHUB_USER_TOKEN or MLFLOW_TRACKING_PASSWORD)",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Custom MLflow tracking URI",
    )

    args = parser.parse_args()

    if args.local:
        db_path = Path("results/mlflow.db").resolve()
        uri = f"sqlite:///{db_path.as_posix()}"
        print(f"Cleaning local runs in {uri}...")
        sys.exit(clean_runs(tracking_uri=uri, experiment_name=args.experiment))

    token = (
        args.token
        or os.environ.get("DAGSHUB_USER_TOKEN")
        or os.environ.get("MLFLOW_TRACKING_PASSWORD")
    )

    uri = args.uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///results/mlflow.db")

    if not token and not uri.startswith("sqlite"):
        print(
            "ERROR: DagsHub token required to delete remote runs.\n"
            "Please provide your token:\n"
            "  1. Via CLI: uv run python clean_mlflow_runs.py --token <YOUR_TOKEN>\n"
            "  2. Or in .env: DAGSHUB_USER_TOKEN=<YOUR_TOKEN>\n"
            "To clean local runs instead, pass --local."
        )
        sys.exit(1)

    print(f"Connecting to MLflow at {uri}...")
    sys.exit(clean_runs(tracking_uri=uri, experiment_name=args.experiment, token=token or ""))


if __name__ == "__main__":
    main()
