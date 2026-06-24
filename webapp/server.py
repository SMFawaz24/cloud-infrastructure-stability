"""
webapp/server.py
----------------
Flask backend for the Cloud Infrastructure Stability Analysis dashboard.

Runs the full detection pipeline once at startup and exposes JSON API
endpoints that the frontend JavaScript calls to render charts, statistics,
alerts, and analytics.

Usage
-----
    pip install flask
    python webapp/server.py

    Then open:  http://127.0.0.1:5050
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template

from src.detection import (
    generate_metrics,
    zscore_detection,
    isolation_forest_detection,
    early_warning_system,
    compute_summary,
    METRIC_COLS,
)
from src.detection import _severity

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Run the pipeline once at startup ─────────────────────────────

print("\n[server]  Running detection pipeline...")

N_POINTS      = 200
RANDOM_SEED   = 42
ZSCORE_THRESH = 2.5
CONTAMINATION = 0.03

DF, INJECTED_MASK        = generate_metrics(n_points=N_POINTS, random_seed=RANDOM_SEED)
ZSCORE_FLAGS, Z_SCORES   = zscore_detection(DF, threshold=ZSCORE_THRESH)
IFOREST_FLAGS, IF_MODEL  = isolation_forest_detection(DF, contamination=CONTAMINATION)
SUMMARY                  = compute_summary(DF, INJECTED_MASK, ZSCORE_FLAGS, IFOREST_FLAGS)
INJECTED_SERIES          = pd.Series(INJECTED_MASK, index=DF.index)

print("[server]  Pipeline complete. Starting web server...\n")


# ── Helper ───────────────────────────────────────────────────────

def _ts_labels():
    """Return ISO timestamp strings for every data point."""
    return [ts.strftime("%H:%M") for ts in DF.index]


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    """KPI cards shown at the top of the dashboard."""
    both_flags = ZSCORE_FLAGS & IFOREST_FLAGS
    return jsonify({
        "total_points"      : int(SUMMARY["total_data_points"]),
        "injected_anomalies": int(SUMMARY["injected_anomalies"]),
        "zscore_detected"   : int(SUMMARY["zscore_detections"]),
        "iforest_detected"  : int(SUMMARY["iforest_detections"]),
        "both_detected"     : int(both_flags.sum()),
        "zscore_recall_pct" : round(SUMMARY["zscore_recall"] * 100, 1),
        "iforest_recall_pct": round(SUMMARY["iforest_recall"] * 100, 1),
        "zscore_precision"  : round(
            SUMMARY["zscore_true_positives"] / SUMMARY["zscore_detections"] * 100, 1
        ) if SUMMARY["zscore_detections"] else 0,
        "iforest_precision" : round(
            SUMMARY["iforest_true_positives"] / SUMMARY["iforest_detections"] * 100, 1
        ) if SUMMARY["iforest_detections"] else 0,
    })


@app.route("/api/timeseries")
def api_timeseries():
    """Full time-series data for all four metrics with anomaly markers."""
    labels = _ts_labels()
    return jsonify({
        "labels"        : labels,
        "cpu"           : [round(v, 2) for v in DF["cpu_usage"].tolist()],
        "memory"        : [round(v, 2) for v in DF["memory_usage"].tolist()],
        "disk"          : [round(v, 2) for v in DF["disk_usage"].tolist()],
        "latency"       : [round(v, 2) for v in DF["network_latency_ms"].tolist()],
        "injected"      : [bool(v) for v in INJECTED_SERIES.tolist()],
        "zscore_flags"  : [bool(v) for v in ZSCORE_FLAGS.tolist()],
        "iforest_flags" : [bool(v) for v in IFOREST_FLAGS.tolist()],
    })


@app.route("/api/alerts")
def api_alerts():
    """All detected alerts with full metric values and severity."""
    alerts = []
    both   = ZSCORE_FLAGS | IFOREST_FLAGS

    for ts, row in DF[both].iterrows():
        z_flag = bool(ZSCORE_FLAGS[ts])
        i_flag = bool(IFOREST_FLAGS[ts])
        alerts.append({
            "timestamp"  : ts.strftime("%Y-%m-%d %H:%M"),
            "cpu"        : round(float(row["cpu_usage"]), 1),
            "memory"     : round(float(row["memory_usage"]), 1),
            "disk"       : round(float(row["disk_usage"]), 1),
            "latency"    : round(float(row["network_latency_ms"]), 1),
            "severity"   : _severity(row),
            "zscore"     : z_flag,
            "iforest"    : i_flag,
            "ground_truth": bool(INJECTED_SERIES[ts]),
        })

    return jsonify({"alerts": alerts, "count": len(alerts)})


@app.route("/api/statistics")
def api_statistics():
    """Descriptive statistics for each metric (mean, std, min, max, percentiles)."""
    stats_out = {}
    for col in METRIC_COLS:
        s = DF[col]
        stats_out[col] = {
            "mean"  : round(float(s.mean()), 2),
            "std"   : round(float(s.std()),  2),
            "min"   : round(float(s.min()),  2),
            "max"   : round(float(s.max()),  2),
            "p25"   : round(float(s.quantile(0.25)), 2),
            "p50"   : round(float(s.quantile(0.50)), 2),
            "p75"   : round(float(s.quantile(0.75)), 2),
            "p95"   : round(float(s.quantile(0.95)), 2),
        }
    return jsonify(stats_out)


@app.route("/api/zscores")
def api_zscores():
    """Z-Score values for every metric at every time step."""
    labels = _ts_labels()
    return jsonify({
        "labels"   : labels,
        "cpu"      : [round(v, 3) for v in Z_SCORES["cpu_usage"].tolist()],
        "memory"   : [round(v, 3) for v in Z_SCORES["memory_usage"].tolist()],
        "disk"     : [round(v, 3) for v in Z_SCORES["disk_usage"].tolist()],
        "latency"  : [round(v, 3) for v in Z_SCORES["network_latency_ms"].tolist()],
        "threshold": ZSCORE_THRESH,
    })


@app.route("/api/correlation")
def api_correlation():
    """Pearson correlation matrix between the four metrics."""
    corr = DF[METRIC_COLS].corr().round(3)
    return jsonify({
        "labels": ["CPU", "Memory", "Disk", "Latency"],
        "matrix": corr.values.tolist(),
    })


@app.route("/api/severity_breakdown")
def api_severity_breakdown():
    """Count of alerts by severity level for each detection method."""
    def count_by_severity(flags):
        counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0}
        for _, row in DF[flags].iterrows():
            counts[_severity(row)] += 1
        return counts

    return jsonify({
        "zscore" : count_by_severity(ZSCORE_FLAGS),
        "iforest": count_by_severity(IFOREST_FLAGS),
    })


if __name__ == "__main__":
    print("=" * 62)
    print("  Cloud Infrastructure Stability Analysis")
    print("  Dashboard  —  http://127.0.0.1:5050")
    print("=" * 62 + "\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
