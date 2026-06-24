"""
visualize.py
------------
Chart generation for the Cloud Infrastructure Stability Analysis project.

Provides:
    - plot_metrics()  : Full 4-panel time-series chart (batch mode)
    - plot_live()     : Live session chart for the Gradio web app
"""

import matplotlib
matplotlib.use("Agg")                     # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

# ── Colour theme ─────────────────────────────────────────────────
BG_DARK  = "#0f1117"
BG_PANEL = "#1a1d27"
GRID_COL = "#2a2d3a"

METRIC_STYLE = {
    "cpu_usage"          : ("CPU Usage (%)",        "steelblue",      (0, 110)),
    "memory_usage"       : ("Memory Usage (%)",     "mediumseagreen", (0, 110)),
    "disk_usage"         : ("Disk Usage (%)",       "goldenrod",      (0, 110)),
    "network_latency_ms" : ("Network Latency (ms)", "orchid",         (0, 520)),
}


def _styled_ax(ax, label: str, ylim: tuple) -> None:
    """Apply consistent dark-theme styling to a subplot axis."""
    ax.set_facecolor(BG_PANEL)
    ax.set_ylabel(label, color="white", fontsize=9.5, labelpad=6)
    ax.set_ylim(ylim)
    ax.tick_params(colors="#aaaaaa", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#333344")
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)


def _legend_strip(ax) -> None:
    """Render the shared legend in a dedicated top strip axis."""
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#9b59b6",
               markeredgecolor="#9b59b6", markersize=18, linewidth=0,
               label="Injected Anomaly (ground truth)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="red", markersize=11, linewidth=0,
               label="Z-Score Alert"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="orange",
               markeredgecolor="orange", markersize=12, linewidth=0,
               label="Isolation Forest Alert"),
    ]
    ax.legend(
        handles=handles, loc="center", ncol=3, fontsize=11, frameon=True,
        facecolor=BG_PANEL, edgecolor="#555555", labelcolor="white",
        handletextpad=0.6, columnspacing=2.5,
    )


def plot_metrics(
    df: pd.DataFrame,
    zscore_flags: pd.Series,
    iforest_flags: pd.Series,
    injected_mask: np.ndarray,
    save_path: str = "outputs/stability_chart.png",
) -> str:
    """
    Render the full 4-panel anomaly detection chart and save to disk.

    Marker layers (drawn in this order so nothing is hidden):
      1. Purple stars  — ground truth injected anomalies (largest, drawn first)
      2. Red circles   — Z-Score detections
      3. Orange X      — Isolation Forest detections (on top)

    Parameters
    ----------
    df            : pd.DataFrame  — full metric dataset
    zscore_flags  : pd.Series     — boolean detections from Z-Score
    iforest_flags : pd.Series     — boolean detections from Isolation Forest
    injected_mask : np.ndarray    — ground truth boolean array
    save_path     : str           — file path to write the PNG

    Returns
    -------
    save_path : str  — the path where the chart was saved
    """
    injected_series = pd.Series(injected_mask, index=df.index)

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(5, 1, height_ratios=[0.18, 1, 1, 1, 1], hspace=0.55)

    _legend_strip(fig.add_subplot(gs[0]))

    for i, (col, (label, line_color, ylim)) in enumerate(METRIC_STYLE.items()):
        ax = fig.add_subplot(gs[i + 1])
        _styled_ax(ax, label, ylim)

        ax.plot(df.index, df[col], color=line_color, linewidth=1.3, alpha=0.85)

        # Ground truth — purple stars, largest, drawn first
        ax.scatter(df.index[injected_series], df[col][injected_series],
                   color="#9b59b6", marker="*", s=350,
                   edgecolors="white", linewidths=0.6, zorder=3, alpha=1.0)
        # Z-Score — red circles
        ax.scatter(df.index[zscore_flags], df[col][zscore_flags],
                   color="red", marker="o", s=90,
                   edgecolors="white", linewidths=0.5, zorder=4, alpha=0.95)
        # Isolation Forest — orange X, on top
        ax.scatter(df.index[iforest_flags], df[col][iforest_flags],
                   color="orange", marker="X", s=100,
                   edgecolors="white", linewidths=0.5, zorder=5, alpha=0.95)

        if i == len(METRIC_STYLE) - 1:
            ax.set_xlabel("Timestamp", color="white", fontsize=10)
        else:
            ax.set_xticklabels([])

    fig.suptitle(
        "Cloud Infrastructure Stability Analysis\n"
        "Time-Series Metrics with Anomaly Detection  (Z-Score + Isolation Forest)",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[chart]  Saved to {save_path}")
    return save_path


def plot_live(
    history_df: pd.DataFrame,
    zscore_anomaly: bool,
    iforest_anomaly: bool,
):
    """
    Render the live session chart for the Gradio web application.

    Draws the rolling history and marks only the latest reading if
    an anomaly was flagged by either detector.

    Parameters
    ----------
    history_df      : pd.DataFrame  — rolling history (up to 50 readings)
    zscore_anomaly  : bool          — whether latest reading was flagged by Z-Score
    iforest_anomaly : bool          — whether latest reading was flagged by Isolation Forest

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
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

    # Simplified legend for live view (no ground truth column)
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
        _styled_ax(ax, label, ylim)

        ax.plot(history_df.index, history_df[col],
                color=line_color, linewidth=1.4, alpha=0.85)

        latest_val = history_df[col].iloc[-1]
        if zscore_anomaly:
            ax.scatter([last_idx], [latest_val],
                       color="red", marker="o", s=140,
                       edgecolors="white", linewidths=0.7, zorder=4)
        if iforest_anomaly:
            ax.scatter([last_idx], [latest_val],
                       color="orange", marker="X", s=150,
                       edgecolors="white", linewidths=0.7, zorder=5)

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
