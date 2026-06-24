"""
app.py  —  Cloud Infrastructure Stability Analysis
===================================================
Local web application entry point.

Runs a Gradio web interface for the Early Warning System.
No internet connection or Hugging Face account required.

Usage
-----
    python app.py

Then open your browser at:  http://127.0.0.1:7860

Authors
-------
    Shreyans Modi       RA2311026010720
    Syed Mohammad Fawaz RA2311026010780
"""

import os
import sys
import warnings

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

MODEL_PATH    = "isolation_forest_model.pkl"
COLS          = ["cpu_usage", "memory_usage", "disk_usage", "network_latency_ms"]
ZSCORE_THRESH = 1.8     # interactive threshold — more sensitive than batch (2.5)
MIN_HISTORY   = 5       # minimum readings before Z-Score activates
MAX_HISTORY   = 50      # rolling window kept per session

BG_DARK   = "#0f1117"
BG_PANEL  = "#1a1d27"
GRID_COL  = "#2a2d3a"

METRIC_STYLE = {
    "cpu_usage"          : ("CPU Usage (%)",        "steelblue",      (0, 110)),
    "memory_usage"       : ("Memory Usage (%)",     "mediumseagreen", (0, 110)),
    "disk_usage"         : ("Disk Usage (%)",       "goldenrod",      (0, 110)),
    "network_latency_ms" : ("Network Latency (ms)", "orchid",         (0, 520)),
}


# ─────────────────────────────────────────────────────────────────
# MODEL  —  load saved .pkl or train fresh on simulated normal data
# ─────────────────────────────────────────────────────────────────

def _load_or_train_model() -> IsolationForest:
    if os.path.exists(MODEL_PATH):
        print(f"[model]  Loaded from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    print("[model]  No saved model found. Training on simulated normal data...")
    np.random.seed(42)
    normal_data = pd.DataFrame({
        "cpu_usage"          : np.random.normal(45, 8,  500),
        "memory_usage"       : np.random.normal(60, 6,  500),
        "disk_usage"         : np.random.normal(55, 5,  500),
        "network_latency_ms" : np.random.normal(30, 10, 500),
    })
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(normal_data)
    joblib.dump(model, MODEL_PATH)
    print(f"[model]  Saved to {MODEL_PATH}")
    return model


IFOREST_MODEL: IsolationForest = _load_or_train_model()


# ─────────────────────────────────────────────────────────────────
# BASELINE  —  pre-seed Z-Score history so it's active from reading 1
# ─────────────────────────────────────────────────────────────────

def _build_baseline(n: int = 10) -> list:
    np.random.seed(42)
    return [
        {
            "cpu_usage"          : float(np.random.normal(45, 4)),
            "memory_usage"       : float(np.random.normal(60, 3)),
            "disk_usage"         : float(np.random.normal(55, 2)),
            "network_latency_ms" : float(np.random.normal(30, 5)),
        }
        for _ in range(n)
    ]


BASELINE = _build_baseline()


# ─────────────────────────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────

def _zscore_check(history_df: pd.DataFrame, new_row: pd.DataFrame) -> bool:
    """Flag if new reading is more than ZSCORE_THRESH std-devs from history mean."""
    if len(history_df) < MIN_HISTORY:
        return False
    combined = pd.concat([history_df[COLS], new_row[COLS]], ignore_index=True)
    z        = combined.apply(stats.zscore).iloc[-1]
    return bool((z.abs() > ZSCORE_THRESH).any())


def _iforest_check(new_row: pd.DataFrame) -> bool:
    """Flag if the Isolation Forest model classifies reading as anomalous (-1)."""
    return bool(IFOREST_MODEL.predict(new_row[COLS])[0] == -1)


def _severity(cpu: float, memory: float) -> str:
    if cpu > 90 or memory > 90:
        return "CRITICAL"
    if cpu > 75 or memory > 75:
        return "HIGH"
    return "MODERATE"


# ─────────────────────────────────────────────────────────────────
# CHART GENERATION
# ─────────────────────────────────────────────────────────────────

def _plot_live(history_df: pd.DataFrame,
               zscore_anom: bool,
               iforest_anom: bool):
    """Render the 4-panel live metric chart for the current session."""

    if history_df is None or len(history_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.5, "Submit a reading to start the live chart.",
                ha="center", va="center", fontsize=13, color="white")
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(5, 1, height_ratios=[0.15, 1, 1, 1, 1], hspace=0.55)

    # Legend strip
    legend_ax = fig.add_subplot(gs[0])
    legend_ax.set_facecolor(BG_DARK)
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=11, linewidth=0, label="Z-Score Alert"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="orange",
                   markersize=12, linewidth=0, label="Isolation Forest Alert"),
        ],
        loc="center", ncol=2, fontsize=11, frameon=True,
        facecolor=BG_PANEL, edgecolor="#555555", labelcolor="white",
        handletextpad=0.6, columnspacing=2.5,
    )

    last_idx = len(history_df) - 1

    for i, (col, (label, line_color, ylim)) in enumerate(METRIC_STYLE.items()):
        ax = fig.add_subplot(gs[i + 1])
        ax.set_facecolor(BG_PANEL)
        ax.plot(history_df.index, history_df[col],
                color=line_color, linewidth=1.4, alpha=0.85)

        latest = history_df[col].iloc[-1]
        if zscore_anom:
            ax.scatter([last_idx], [latest], color="red", marker="o",
                       s=140, edgecolors="white", linewidths=0.7, zorder=4)
        if iforest_anom:
            ax.scatter([last_idx], [latest], color="orange", marker="X",
                       s=150, edgecolors="white", linewidths=0.7, zorder=5)

        ax.set_ylabel(label, color="white", fontsize=9, labelpad=5)
        ax.set_ylim(ylim)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333344")
        ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)

        if i < len(METRIC_STYLE) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Reading Number", color="white", fontsize=9)

    fig.suptitle(
        "Cloud Infrastructure  —  Live Metric Monitor  |  Last 50 Readings",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────
# CORE ANALYZE FUNCTION  —  called on every button press
# ─────────────────────────────────────────────────────────────────

def analyze(cpu: float, memory: float, disk: float, latency: float,
            history_state: list):
    """
    Process one metric reading.

    Appends to the rolling session history, runs Z-Score and Isolation Forest
    detectors, builds a severity-graded status message, updates the live chart,
    and returns the updated gr.State so history persists across calls.
    """
    # Restore history from Gradio state (list-of-dicts -> DataFrame)
    history_df = (
        pd.DataFrame(history_state, columns=COLS)
        if history_state
        else pd.DataFrame(columns=COLS)
    )

    new_row = pd.DataFrame([{
        "cpu_usage"          : float(cpu),
        "memory_usage"       : float(memory),
        "disk_usage"         : float(disk),
        "network_latency_ms" : float(latency),
    }])

    # Append and keep rolling window
    history_df = pd.concat([history_df, new_row], ignore_index=True).tail(MAX_HISTORY)

    # Detectors — pass history without the new row so the new row is what is evaluated
    zscore_flag  = _zscore_check(history_df.iloc[:-1], new_row)
    iforest_flag = _iforest_check(new_row)

    # Status message
    reading_num = len(history_df)
    lines = [
        f"  Reading #{reading_num}",
        f"  CPU: {cpu:.1f}%   Memory: {memory:.1f}%   "
        f"Disk: {disk:.1f}%   Latency: {latency:.1f} ms",
        "",
    ]

    if not zscore_flag and not iforest_flag:
        lines.append("  STATUS : STABLE — All metrics within normal range.")
    else:
        lines.append("  STATUS : ANOMALY DETECTED — Early Warning Triggered!")
        lines.append("")
        if zscore_flag:
            lines.append("  [Z-Score]   Reading is statistically abnormal.")
        if iforest_flag:
            lines.append("  [IForest]   Isolation Forest flagged this as an outlier.")
        lines.append("")
        sev = _severity(cpu, memory)
        if sev == "CRITICAL":
            lines.append("  Severity : CRITICAL — Immediate action required.")
        elif sev == "HIGH":
            lines.append("  Severity : HIGH — Investigate soon.")
        else:
            lines.append("  Severity : MODERATE — Monitor closely.")

    status_text   = "\n".join(lines)
    fig           = _plot_live(history_df, zscore_flag, iforest_flag)
    updated_state = history_df[COLS].to_dict(orient="records")

    return status_text, fig, updated_state


# ─────────────────────────────────────────────────────────────────
# GRADIO UI LAYOUT
# ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Cloud Infrastructure Stability Analysis",
    theme=gr.themes.Base(),
) as demo:

    # Each session gets its own rolling history via gr.State
    history_state = gr.State(BASELINE.copy())

    gr.Markdown(
        "# Cloud Infrastructure Stability Analysis\n"
        "### Early Warning System — Anomaly Detection\n\n"
        "Enter current cloud server metrics and click **Analyze**. "
        "The system detects anomalies using **Z-Score** and **Isolation Forest** "
        "and updates the live chart with each new reading.\n\n"
        "---"
    )

    with gr.Row():

        # ── Left: inputs ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Input Metrics")

            cpu_in  = gr.Slider(0, 100, value=45, step=0.5, label="CPU Usage (%)")
            mem_in  = gr.Slider(0, 100, value=60, step=0.5, label="Memory Usage (%)")
            disk_in = gr.Slider(0, 100, value=55, step=0.5, label="Disk Usage (%)")
            lat_in  = gr.Slider(0, 500, value=30, step=1,   label="Network Latency (ms)")

            gr.Markdown("---\n**Preset scenarios — one click runs the analysis:**")

            with gr.Row():
                btn_normal = gr.Button("Normal Reading")
                btn_cpu    = gr.Button("CPU Spike")
            with gr.Row():
                btn_lat    = gr.Button("Latency Spike")
                btn_crit   = gr.Button("Critical Event")

            btn_analyze = gr.Button("Analyze", variant="primary", size="lg")

        # ── Right: outputs ────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Detection Result")
            status_out = gr.Textbox(label="", lines=11, interactive=False)
            chart_out  = gr.Plot(label="Live Metric Chart")

    # ── Preset buttons (set sliders + run analyze in one click) ──

    def _preset(cpu, mem, disk, lat, state):
        return analyze(cpu, mem, disk, lat, state)

    btn_normal.click(
        fn=lambda s: _preset(45, 60, 55, 30, s),
        inputs=[history_state],
        outputs=[status_out, chart_out, history_state],
    ).then(fn=lambda: (45, 60, 55, 30),
           outputs=[cpu_in, mem_in, disk_in, lat_in])

    btn_cpu.click(
        fn=lambda s: _preset(95, 88, 56, 32, s),
        inputs=[history_state],
        outputs=[status_out, chart_out, history_state],
    ).then(fn=lambda: (95, 88, 56, 32),
           outputs=[cpu_in, mem_in, disk_in, lat_in])

    btn_lat.click(
        fn=lambda s: _preset(47, 61, 54, 280, s),
        inputs=[history_state],
        outputs=[status_out, chart_out, history_state],
    ).then(fn=lambda: (47, 61, 54, 280),
           outputs=[cpu_in, mem_in, disk_in, lat_in])

    btn_crit.click(
        fn=lambda s: _preset(98, 95, 91, 320, s),
        inputs=[history_state],
        outputs=[status_out, chart_out, history_state],
    ).then(fn=lambda: (98, 95, 91, 320),
           outputs=[cpu_in, mem_in, disk_in, lat_in])

    # ── Manual analyze button ─────────────────────────────────────
    btn_analyze.click(
        fn=analyze,
        inputs=[cpu_in, mem_in, disk_in, lat_in, history_state],
        outputs=[status_out, chart_out, history_state],
    )

    gr.Markdown(
        "---\n"
        "**How it works**\n\n"
        "- **Z-Score** compares each reading against the session history. "
        "Flags any metric more than 1.8 standard deviations from the mean. "
        "Active from the first reading (pre-seeded with baseline data).\n"
        "- **Isolation Forest** is a machine learning model trained on normal "
        "operating data. Flags readings that look unlike anything it was trained on. "
        "Works from the very first reading.\n"
        "- When both detectors agree, confidence in the alert is higher.\n\n"
        "---\n"
        "*Shreyans Modi (RA2311026010720)  |  "
        "Syed Mohammad Fawaz (RA2311026010780)*"
    )


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  Cloud Infrastructure Stability Analysis")
    print("  Early Warning System  —  Local Web App")
    print("=" * 62)
    print("\n  Open your browser at:  http://127.0.0.1:7860\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,        # set True to get a public Gradio share link
        show_error=True,
    )
