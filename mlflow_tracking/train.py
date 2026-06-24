"""
mlflow_tracking/train.py
------------------------
MLflow-integrated pipeline for the Cloud Infrastructure Stability Analysis project.

Wraps the full detection pipeline in a single MLflow run, logging:
    - Parameters : n_data_points, random_seed, zscore_threshold,
                   iforest_contamination, detection_methods
    - Metrics    : zscore_recall, iforest_recall, detection counts,
                   true positive counts, injected_anomalies
    - Artifacts  : 4-panel stability chart (PNG)
    - Model      : Trained Isolation Forest saved to the MLflow Model Registry
                   under the name "CloudStabilityIForest"

Usage
-----
    python mlflow_tracking/train.py

Then view results:
    mlflow ui
    # Open http://127.0.0.1:5000
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.sklearn

from src.detection import (
    generate_metrics,
    zscore_detection,
    isolation_forest_detection,
    early_warning_system,
    compute_summary,
)
from src.visualize import plot_metrics

# ── Configuration ─────────────────────────────────────────────────
EXPERIMENT_NAME = "Cloud_Stability_Analysis"
RUN_NAME        = "Cloud_Stability_Run"
CHART_PATH      = "stability_chart.png"

N_POINTS        = 200
RANDOM_SEED     = 42
ZSCORE_THRESH   = 2.5
CONTAMINATION   = 0.03


def run_experiment():
    """Execute the full pipeline inside a single MLflow tracking run."""

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n" + "=" * 62)
    print("  Cloud Infrastructure Stability Analysis")
    print("  MLflow Experiment Tracking")
    print("=" * 62 + "\n")

    with mlflow.start_run(run_name=RUN_NAME):

        # ── Log parameters ───────────────────────────────────────────
        mlflow.log_param("n_data_points",          N_POINTS)
        mlflow.log_param("random_seed",            RANDOM_SEED)
        mlflow.log_param("zscore_threshold",       ZSCORE_THRESH)
        mlflow.log_param("iforest_contamination",  CONTAMINATION)
        mlflow.log_param("detection_methods",      "ZScore + IsolationForest")

        # ── Step 1: Generate data ────────────────────────────────────
        df, injected_mask = generate_metrics(
            n_points=N_POINTS, random_seed=RANDOM_SEED
        )

        # ── Step 2: Z-Score detection ────────────────────────────────
        zscore_flags, _ = zscore_detection(df, threshold=ZSCORE_THRESH)

        # ── Step 3: Isolation Forest detection ───────────────────────
        iforest_flags, iforest_model = isolation_forest_detection(
            df, contamination=CONTAMINATION
        )

        # ── Step 4: Early Warning System alerts ──────────────────────
        early_warning_system(df, zscore_flags,  method_name="Z-Score Detection")
        early_warning_system(df, iforest_flags, method_name="Isolation Forest")

        # ── Step 5: Save chart and log as artifact ───────────────────
        plot_metrics(df, zscore_flags, iforest_flags,
                     injected_mask, save_path=CHART_PATH)
        mlflow.log_artifact(CHART_PATH)

        # ── Step 6: Compute and log metrics ──────────────────────────
        summary = compute_summary(df, injected_mask, zscore_flags, iforest_flags)
        for key, value in summary.items():
            mlflow.log_metric(key, value)

        # ── Step 7: Log Isolation Forest model to Model Registry ─────
        mlflow.sklearn.log_model(
            sk_model              = iforest_model,
            artifact_path         = "isolation_forest_model",
            registered_model_name = "CloudStabilityIForest",
        )

        run_id = mlflow.active_run().info.run_id
        print(f"[mlflow]  Run ID    : {run_id}")
        print(f"[mlflow]  Experiment: {EXPERIMENT_NAME}")
        print("\nTo view results run:  mlflow ui")
        print("Then open:            http://127.0.0.1:5000\n")

    # Clean up local chart copy (already logged to MLflow)
    if os.path.exists(CHART_PATH):
        os.remove(CHART_PATH)


if __name__ == "__main__":
    run_experiment()
