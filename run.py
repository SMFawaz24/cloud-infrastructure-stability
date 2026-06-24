"""
run.py
------
Standalone entry point for the Cloud Infrastructure Stability Analysis project.

Runs the full pipeline without MLflow tracking:
    1. Simulate 200 minutes of cloud server metrics with 6 injected anomalies
    2. Detect anomalies using Z-Score statistical analysis
    3. Detect anomalies using Isolation Forest ML model
    4. Print Early Warning System alerts to the terminal
    5. Save a 4-panel time-series chart to outputs/stability_chart.png

Usage
-----
    python run.py
"""

import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.detection import (
    generate_metrics,
    zscore_detection,
    isolation_forest_detection,
    early_warning_system,
    compute_summary,
)
from src.visualize import plot_metrics

OUTPUTS_DIR   = "outputs"
CHART_PATH    = os.path.join(OUTPUTS_DIR, "stability_chart.png")

# Detection configuration
N_POINTS      = 200
RANDOM_SEED   = 42
ZSCORE_THRESH = 2.5
CONTAMINATION = 0.03


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("\n" + "=" * 62)
    print("  Cloud Infrastructure Stability Analysis")
    print("  Early Warning System")
    print("=" * 62 + "\n")

    # 1. Generate simulated metrics
    df, injected_mask = generate_metrics(
        n_points=N_POINTS, random_seed=RANDOM_SEED
    )

    # 2. Z-Score detection
    zscore_flags, _ = zscore_detection(df, threshold=ZSCORE_THRESH)

    # 3. Isolation Forest detection
    iforest_flags, _ = isolation_forest_detection(df, contamination=CONTAMINATION)

    # 4. Early Warning System alerts
    early_warning_system(df, zscore_flags,  method_name="Z-Score Detection")
    early_warning_system(df, iforest_flags, method_name="Isolation Forest")

    # 5. Summary statistics
    compute_summary(df, injected_mask, zscore_flags, iforest_flags)

    # 6. Save chart
    plot_metrics(df, zscore_flags, iforest_flags, injected_mask,
                 save_path=CHART_PATH)

    print(f"Done. Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
