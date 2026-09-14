"""
Report Figure Generator
=======================
Draws the figures the project report needs that the training pipeline does not
already produce.

The pipeline writes the model-evaluation figures — ROC, precision-recall,
confusion, calibration, comparison bars, class balance, correlations — directly
from the fitted models. This script adds the two analysis figures that come from
the data and the cost model, plus the architecture diagram.

Everything here is plotted from measured values. An earlier version of this file
drew ROC curves from a closed-form expression and labelled them with an invented
AUC; if a figure cannot be computed now, it is skipped with a warning instead.

Usage:
    python scripts/generate_project_figures.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("figures")

OUTPUT_DIR = project_root / "reports" / "figures"

# Mirrors the dashboard tokens so figures and screens read as one system.
BG = "#0e1217"
SURFACE = "#151a21"
INK = "#e8eaed"
INK_MUTED = "#8b96a6"
GRID = "#232b36"
BLUE = "#3987e5"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"


def _style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply the shared dark styling to an axis."""
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="600", pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.5, color=GRID)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    """Write a figure to the reports directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Wrote %s", path.relative_to(project_root))


def plot_tool_wear_profile(metrics: Dict[str, Any]) -> None:
    """Plot the measured failure rate against accumulated tool wear."""
    profile = metrics["tool_wear"]["profile"]
    finding = metrics["tool_wear"]["finding"]
    if not profile:
        logger.warning("No tool-wear profile available; skipping.")
        return

    midpoints, rates, assets = [], [], []
    for row in profile:
        left, right = row["bucket"].strip("()[]").split(", ")
        midpoints.append((float(left) + float(right)) / 2)
        rates.append(row["failure_rate"])
        assets.append(row["assets"])

    threshold = finding.get("threshold") if finding else None
    colours = [
        CRITICAL if threshold is not None and m > threshold else BLUE for m in midpoints
    ]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.bar(midpoints, rates, width=16, color=colours, zorder=3)

    if threshold is not None:
        ax.axvline(threshold, color=WARNING, linestyle="--", linewidth=1.4, zorder=4)
        ax.text(
            threshold + 4, max(rates) * 0.92,
            f"replace at {threshold} min",
            color=WARNING, fontsize=10, fontweight="600",
        )

    _style_axes(
        ax,
        "Failure rate rises sharply once tooling passes its service limit",
        "Accumulated tool wear (minutes)",
        "Failure rate (%)",
    )

    if finding:
        ax.text(
            0.015, 0.95,
            f"{finding['avoidable']} of {finding['total']} {finding['kind']} "
            f"({finding['share']:.0f}%) occur past this point",
            transform=ax.transAxes, color=INK, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,
                      edgecolor=GRID, linewidth=1),
        )

    fig.text(
        0.5, -0.02,
        f"Measured across {sum(assets):,} assets. Each bar is a 20-minute bucket.",
        ha="center", color=INK_MUTED, fontsize=9,
    )
    _save(fig, "tool_wear_vs_failure_rate.png")


def plot_cost_comparison(metrics: Dict[str, Any]) -> None:
    """Plot annual cost with and without the model, from the cost model."""
    cost = metrics.get("cost")
    if not cost:
        logger.warning("No cost metrics available; skipping.")
        return

    labels = ["Run to failure", "With the model"]
    values = [cost["reactive_cost"], cost["predictive_cost"]]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    bars = ax.bar(labels, values, width=0.5, color=[CRITICAL, GOOD], zorder=3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value * 1.02,
            f"${value:,.0f}", ha="center", color=INK, fontsize=12, fontweight="600",
        )

    _style_axes(ax, "Annual failure-related cost", "", "Cost (USD)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.get_xaxis().set_tick_params(labelsize=11, colors=INK)

    ax.annotate(
        f"${cost['avoided_cost']:,.0f} avoided\n{cost['reduction_pct']:.0f}% reduction",
        xy=(1, values[1]), xytext=(0.45, max(values) * 0.72),
        color=GOOD, fontsize=11, fontweight="600", ha="center",
        arrowprops=dict(color=GOOD, arrowstyle="->", linewidth=1.5),
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,
                  edgecolor=GOOD, linewidth=1),
    )

    confusion = cost["confusion"]
    fig.text(
        0.5, -0.04,
        f"Scaled from the {cost['test_split_size']:,}-asset test split "
        f"(x{cost['scale_factor']}). "
        f"Model caught {confusion['tp']} failures, missed {confusion['fn']}, "
        f"raised {confusion['fp']} false alarms. "
        f"Inputs are configurable in config/config.yaml.",
        ha="center", color=INK_MUTED, fontsize=8.5, wrap=True,
    )
    _save(fig, "cost_comparison.png")


def plot_architecture() -> None:
    """Draw the system architecture diagram.

    A diagram is a drawing of the design, not a claim about results, so it is
    the one figure here that is not computed from data.
    """
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, title, lines, accent=BLUE):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=SURFACE, edgecolor=accent,
                          linewidth=1.4, zorder=2)
        )
        ax.add_patch(
            plt.Rectangle((x, y + h - 0.06), w, 0.06, facecolor=accent,
                          edgecolor="none", zorder=3)
        )
        ax.text(x + 0.18, y + h - 0.34, title, color=INK, fontsize=10,
                fontweight="600", zorder=4)
        for i, line in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.66 - i * 0.28, line, color=INK_MUTED,
                    fontsize=8.6, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.3))

    ax.text(0.2, 6.15, "Platform architecture", color=INK, fontsize=15, fontweight="600")
    ax.text(0.2, 5.82, "From plant sensor to maintenance instruction",
            color=INK_MUTED, fontsize=10)

    box(0.2, 3.5, 2.7, 1.9, "1. Ingestion",
        ["MQTT / OPC-UA telemetry", "PLC REST push", "CSV and Excel upload",
         "Database query", "Watched folder"])
    box(3.3, 3.5, 2.7, 1.9, "2. Preparation",
        ["Schema validation", "Range checks", "5 engineered features",
         "Fitted scaler", "Leakage columns dropped"])
    box(6.4, 3.5, 2.7, 1.9, "3. Model",
        ["Candidate comparison", "Randomised search", "Selected on F1",
         "Isotonic calibration", "Probability out"], accent=GOOD)
    box(9.5, 3.5, 3.3, 1.9, "4. Interpretation",
        ["SHAP attribution", "Risk score 0-100", "Risk banding",
         "Rule-based actions", "Station mapping"], accent=WARNING)

    arrow(2.9, 4.45, 3.3, 4.45)
    arrow(6.0, 4.45, 6.4, 4.45)
    arrow(9.1, 4.45, 9.5, 4.45)

    box(0.2, 0.9, 6.0, 2.1, "Dashboard",
        ["Asset Health  ·  Risk Assessment  ·  Data Explorer",
         "Model Diagnostics  ·  Defect Inspection",
         "Model & Cost Analysis  ·  Batch Analysis  ·  Alerts",
         "", "Streamlit, eight modules"])
    box(6.4, 0.9, 6.4, 2.1, "Artifacts",
        ["models/  fitted models, scaler, feature metadata",
         "reports/figures/  evaluation plots",
         "reports/results/  metrics as JSON",
         "", "Regenerated by scripts/train_pipeline.py"])

    arrow(3.2, 3.5, 3.2, 3.0)
    arrow(9.6, 3.5, 9.6, 3.0)

    _save(fig, "system_architecture_diagram.png")


def main(argv=None) -> int:
    """Generate the report figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture-only",
        action="store_true",
        help="Draw only the architecture diagram, which needs no trained model.",
    )
    args = parser.parse_args(argv)

    plot_architecture()

    if args.architecture_only:
        return 0

    from src.reporting.metrics import collect_metrics

    try:
        metrics = collect_metrics(project_root)
    except FileNotFoundError:
        logger.error(
            "No model artifacts found. Run `python scripts/train_pipeline.py "
            "--mode baseline` first. Data-derived figures were skipped."
        )
        return 1

    plot_tool_wear_profile(metrics)
    plot_cost_comparison(metrics)
    logger.info("Done. Model-evaluation figures come from the training pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
