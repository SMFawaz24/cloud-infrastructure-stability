"""
mlflow_tracking/export_model.py
--------------------------------
Export the trained Isolation Forest model from MLflow to a .pkl file.

Run this ONCE after mlflow_tracking/train.py has completed at least one run.
The exported file (isolation_forest_model.pkl) should be copied into the
huggingface/ directory before uploading to Hugging Face Spaces.

Usage
-----
    python mlflow_tracking/export_model.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.sklearn
import joblib

EXPERIMENT_NAME = "Cloud_Stability_Analysis"
OUTPUT_PATH     = "isolation_forest_model.pkl"


def export_model():
    """
    Locate the most recent MLflow run, load the Isolation Forest model
    from its artifacts, and serialise it to a joblib .pkl file.
    """
    print("[export]  Connecting to MLflow tracking server...")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"[export]  ERROR: Experiment '{EXPERIMENT_NAME}' not found.")
        print("          Run mlflow_tracking/train.py first.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("[export]  ERROR: No runs found.")
        print("          Run mlflow_tracking/train.py first.")
        sys.exit(1)

    run_id    = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/isolation_forest_model"

    print(f"[export]  Loading model from run: {run_id}")
    model = mlflow.sklearn.load_model(model_uri)

    joblib.dump(model, OUTPUT_PATH)
    print(f"[export]  Model saved to {OUTPUT_PATH}")
    print(f"          Copy this file into the huggingface/ folder.")


if __name__ == "__main__":
    export_model()
