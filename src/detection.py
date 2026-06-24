"""
detection.py
------------
Core anomaly detection logic shared across the entire project.

Provides:
    - generate_metrics()          : Simulate cloud infrastructure time-series data
    - zscore_detection()          : Statistical anomaly detection via Z-Score
    - isolation_forest_detection(): ML-based anomaly detection via Isolation Forest
    - early_warning_system()      : Alert printer with severity classification
    - compute_summary()           : Recall and detection count statistics
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

# Metric columns used consistently across all modules
METRIC_COLS = ["cpu_usage", "memory_usage", "disk_usage", "network_latency_ms"]


# DATA GENERATION

def generate_metrics(n_points: int = 200, random_seed: int = 42):
    """
    Simulate time-series data for four cloud infrastructure metrics.

    Normal baseline values are drawn from Gaussian distributions centred
    around realistic operating means.  Six anomaly events are then injected
    at evenly-spaced positions across the timeline with a small random jitter
    so they appear naturally distributed.  Each event affects only one or
    two metrics, which is consistent with real-world incident patterns
    (e.g. a memory leak raises CPU and memory but not disk or latency).

    Parameters
    ----------
    n_points    : int  — number of one-minute time steps to generate
    random_seed : int  — seed for reproducibility

    Returns
    -------
    df           : pd.DataFrame  — indexed by timestamp, columns = METRIC_COLS
    anomaly_mask : np.ndarray    — boolean array, True where anomaly was injected
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start="2024-01-01", periods=n_points, freq="1min")

    cpu     = np.random.normal(loc=45, scale=8,  size=n_points)
    memory  = np.random.normal(loc=60, scale=6,  size=n_points)
    disk    = np.random.normal(loc=55, scale=5,  size=n_points)
    latency = np.random.normal(loc=30, scale=10, size=n_points)

    # Spread 6 anomalies evenly across the timeline
    n_anomalies  = 6
    min_gap      = n_points // (n_anomalies + 1)
    anomaly_mask = np.zeros(n_points, dtype=bool)
    anomaly_idx  = []

    candidate = min_gap
    for _ in range(n_anomalies):
        jitter = np.random.randint(-5, 6)
        idx    = int(np.clip(candidate + jitter, 5, n_points - 5))
        anomaly_idx.append(idx)
        anomaly_mask[idx] = True
        candidate += min_gap

    for idx in anomaly_idx:
        event = np.random.choice(
            ["cpu_memory", "latency_only", "disk_cpu"],
            p=[0.4, 0.3, 0.3]
        )
        if event == "cpu_memory":
            cpu[idx]    += np.random.uniform(40, 55)
            memory[idx] += np.random.uniform(25, 40)
        elif event == "latency_only":
            latency[idx] += np.random.uniform(150, 250)
        else:
            disk[idx] += np.random.uniform(25, 38)
            cpu[idx]  += np.random.uniform(20, 35)

    cpu     = np.clip(cpu,     0, 100)
    memory  = np.clip(memory,  0, 100)
    disk    = np.clip(disk,    0, 100)
    latency = np.clip(latency, 0, 500)

    df = pd.DataFrame({
        "cpu_usage"          : cpu,
        "memory_usage"       : memory,
        "disk_usage"         : disk,
        "network_latency_ms" : latency,
    }, index=timestamps)

    print(f"[data]  Generated {n_points} data points with {n_anomalies} injected anomalies.")
    return df, anomaly_mask


# ─────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────

def zscore_detection(df: pd.DataFrame, threshold: float = 2.5):
    """
    Z-Score statistical anomaly detection.

    Computes the Z-Score for every metric across the full dataset.
    A reading is flagged if any metric's absolute Z-Score exceeds
    the threshold, meaning it is more than `threshold` standard
    deviations away from the column mean.

    Parameters
    ----------
    df        : pd.DataFrame — metric data
    threshold : float        — standard deviation cutoff (default 2.5)

    Returns
    -------
    anomaly_flags : pd.Series   — boolean, True where anomaly detected
    z_scores      : pd.DataFrame— per-metric Z-Score values
    """
    z_scores      = df[METRIC_COLS].apply(stats.zscore)
    anomaly_flags = (z_scores.abs() > threshold).any(axis=1)

    print(f"[zscore]  threshold={threshold} | "
          f"detected {anomaly_flags.sum()} / {len(df)} anomalies.")
    return anomaly_flags, z_scores


def isolation_forest_detection(df: pd.DataFrame, contamination: float = 0.03):
    """
    Isolation Forest ML anomaly detection.

    Builds an ensemble of random isolation trees.  Points that require
    fewer splits to isolate (shorter average path length) are scored as
    anomalous.  The contamination parameter sets the expected proportion
    of outliers in the data.

    Parameters
    ----------
    df            : pd.DataFrame — metric data
    contamination : float        — expected outlier fraction (default 0.03)

    Returns
    -------
    anomaly_flags : pd.Series           — boolean, True where anomaly detected
    model         : IsolationForest     — the fitted model (for serialisation)
    """
    model         = IsolationForest(contamination=contamination, random_state=42)
    predictions   = model.fit_predict(df[METRIC_COLS])
    anomaly_flags = pd.Series(predictions == -1, index=df.index)

    print(f"[iforest] contamination={contamination} | "
          f"detected {anomaly_flags.sum()} / {len(df)} anomalies.")
    return anomaly_flags, model


# ─────────────────────────────────────────────────────────────────
# EARLY WARNING SYSTEM
# ─────────────────────────────────────────────────────────────────

def _severity(row: pd.Series) -> str:
    """Classify alert severity from a single metric reading."""
    if row["cpu_usage"] > 90 or row["memory_usage"] > 90:
        return "CRITICAL"
    if row["cpu_usage"] > 75 or row["memory_usage"] > 75:
        return "HIGH"
    return "MODERATE"


def early_warning_system(
    df: pd.DataFrame,
    anomaly_flags: pd.Series,
    method_name: str = "Detection",
) -> list:
    """
    Print timestamped alerts for every flagged reading and return them
    as a list of dicts for downstream use (e.g. logging to MLflow).

    Parameters
    ----------
    df            : pd.DataFrame — full metric dataset
    anomaly_flags : pd.Series   — boolean flags from a detector
    method_name   : str         — label displayed in the header

    Returns
    -------
    alerts : list[dict]  — one dict per alert
    """
    SEP = "=" * 62
    print(f"\n{SEP}")
    print(f"  EARLY WARNING SYSTEM  |  {method_name}")
    print(SEP)

    alerts        = []
    anomalous     = df[anomaly_flags]

    if anomalous.empty:
        print("  No anomalies detected. System is stable.")
    else:
        for ts, row in anomalous.iterrows():
            severity = _severity(row)
            print(f"\n  [ALERT]  {ts.strftime('%Y-%m-%d %H:%M')}")
            print(f"    CPU Usage        : {row['cpu_usage']:.1f}%")
            print(f"    Memory Usage     : {row['memory_usage']:.1f}%")
            print(f"    Disk Usage       : {row['disk_usage']:.1f}%")
            print(f"    Network Latency  : {row['network_latency_ms']:.1f} ms")
            print(f"    Severity         : {severity}")
            alerts.append({
                "timestamp"          : ts.isoformat(),
                "cpu_usage"          : round(float(row["cpu_usage"]), 2),
                "memory_usage"       : round(float(row["memory_usage"]), 2),
                "disk_usage"         : round(float(row["disk_usage"]), 2),
                "network_latency_ms" : round(float(row["network_latency_ms"]), 2),
                "severity"           : severity,
                "method"             : method_name,
            })

    print(f"\n{SEP}\n")
    return alerts


# ─────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────

def compute_summary(
    df: pd.DataFrame,
    injected_mask: np.ndarray,
    zscore_flags: pd.Series,
    iforest_flags: pd.Series,
) -> dict:
    """
    Compute detection performance statistics.

    Returns a dictionary suitable for direct use with mlflow.log_metric().

    Parameters
    ----------
    df            : pd.DataFrame
    injected_mask : np.ndarray  — ground truth anomaly positions
    zscore_flags  : pd.Series   — Z-Score detections
    iforest_flags : pd.Series   — Isolation Forest detections

    Returns
    -------
    summary : dict
    """
    injected_series = pd.Series(injected_mask, index=df.index)
    n_injected      = int(injected_mask.sum())
    n_zscore        = int(zscore_flags.sum())
    n_iforest       = int(iforest_flags.sum())
    tp_z            = int((zscore_flags  & injected_series).sum())
    tp_if           = int((iforest_flags & injected_series).sum())

    summary = {
        "total_data_points"       : len(df),
        "injected_anomalies"      : n_injected,
        "zscore_detections"       : n_zscore,
        "iforest_detections"      : n_iforest,
        "zscore_true_positives"   : tp_z,
        "iforest_true_positives"  : tp_if,
        "zscore_recall"           : round(tp_z  / n_injected, 4) if n_injected else 0,
        "iforest_recall"          : round(tp_if / n_injected, 4) if n_injected else 0,
    }

    print("DETECTION SUMMARY")
    print("-" * 50)
    for k, v in summary.items():
        print(f"  {k:<30} : {v}")
    print("-" * 50 + "\n")

    return summary
