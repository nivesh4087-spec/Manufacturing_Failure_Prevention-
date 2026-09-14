"""
Agile Tracking Charts
=====================
Draws the sprint charts referenced by the Agile/Scrum submission document.

The sprint data lives in :data:`SPRINTS` so the document and the charts cannot
drift apart — edit it here and regenerate rather than editing a chart by hand.

Usage:
    python scripts/generate_agile_charts.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("agile")

OUTPUT_DIR = project_root / "reports" / "figures"

# Shared with the dashboard theme so every figure in the project matches.
BG = "#0e1217"
SURFACE = "#151a21"
INK = "#e8eaed"
INK_MUTED = "#8b96a6"
GRID = "#232b36"
BLUE = "#3987e5"
ORANGE = "#d95926"
GOOD = "#0ca30c"
WARNING = "#fab219"

#: Sprint 1 daily tracking. Ideal is a straight burn from the committed total;
#: actual is what the board showed at the end of each day.
SPRINT_1 = {
    "name": "Sprint 1 — Foundation and core architecture",
    "committed_points": 42,
    "days": list(range(0, 11)),
    "actual_remaining": [42, 42, 39, 34, 26, 21, 16, 13, 10, 5, 0],
    "completed": [
        "Planning and backlog refinement",
        "Architecture and environment setup",
        "Story 1 — executive dashboard (3)",
        "Story 2 — data explorer (5)",
        "Story 3 — risk predictor (8)",
        "Story 4 — batch upload (5)",
        "Story 5 — SHAP explainability (5)",
        "Story 6 — model benchmarking (3)",
        "Story 7 — threshold alerting (3)",
        "Story 8 — quality inspection (5)",
        "Stories 9 and 10 — technical debt (5)",
    ],
}

#: Sprint 2 covers the hardening pass: the defects found in Sprint 1's output
#: and the work to make the platform's claims verifiable.
SPRINT_2 = {
    "name": "Sprint 2 — Hardening and verification",
    "committed_points": 34,
    "days": list(range(0, 11)),
    "actual_remaining": [34, 34, 29, 24, 21, 16, 12, 8, 5, 2, 0],
    "completed": [
        "Planning",
        "Repository hygiene, dependency audit",
        "Story 11 — crash and dependency fixes (5)",
        "Story 12 — interface rebuild (5)",
        "Story 13 — state persistence fix (3)",
        "Story 14 — cold-start onboarding (5)",
        "Story 15 — caching and performance (4)",
        "Story 16 — remove hardcoded content (4)",
        "Story 17 — training profiles (3)",
        "Story 18 — measured reporting (3)",
        "Story 19 — documentation (2)",
    ],
}

SPRINTS: List[Dict] = [SPRINT_1, SPRINT_2]

#: Cumulative flow over Sprint 2, in story counts per board column.
CUMULATIVE_FLOW = {
    "days": list(range(0, 11)),
    "Done": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "In progress": [0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0],
    "To do": [9, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0],
}


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply the shared dark styling."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="600", pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, linewidth=0.6, alpha=0.5, color=GRID)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Wrote %s", path.relative_to(project_root))


def plot_burndown(sprint: Dict, filename: str) -> None:
    """Draw one sprint's burndown against the ideal line."""
    days = np.array(sprint["days"])
    actual = np.array(sprint["actual_remaining"])
    total = sprint["committed_points"]
    ideal = np.linspace(total, 0, len(days))

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=BG)

    ax.plot(days, ideal, color=INK_MUTED, linestyle="--", linewidth=1.6,
            label="Ideal burn", zorder=3)
    ax.plot(days, actual, color=BLUE, linewidth=2.4, marker="o", markersize=7,
            markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Actual remaining", zorder=4)

    # Shade where the team ran behind the ideal line — the honest reading of a
    # burndown is where the gap opens, not just whether it closed by the end.
    ax.fill_between(days, ideal, actual, where=(actual >= ideal),
                    color=WARNING, alpha=0.16, interpolate=True,
                    label="Behind ideal", zorder=2)
    ax.fill_between(days, ideal, actual, where=(actual < ideal),
                    color=GOOD, alpha=0.16, interpolate=True,
                    label="Ahead of ideal", zorder=2)

    _style(ax, sprint["name"], "Sprint day", "Story points remaining")
    ax.set_xticks(days)
    ax.set_ylim(-1, total * 1.08)
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, fontsize=9.5,
              framealpha=1)

    velocity = total / (len(days) - 1)
    ax.text(
        0.985, 0.94,
        f"{total} points committed\n{velocity:.1f} points per day",
        transform=ax.transAxes, color=INK, fontsize=9.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID),
    )
    _save(fig, filename)


def plot_velocity() -> None:
    """Compare committed against delivered points per sprint."""
    names = [s["name"].split("—")[0].strip() for s in SPRINTS]
    committed = [s["committed_points"] for s in SPRINTS]
    delivered = [s["committed_points"] - s["actual_remaining"][-1] for s in SPRINTS]

    x = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    ax.bar(x - width / 2, committed, width, label="Committed", color=INK_MUTED, zorder=3)
    ax.bar(x + width / 2, delivered, width, label="Delivered", color=GOOD, zorder=3)

    for i, (c, d) in enumerate(zip(committed, delivered)):
        ax.text(i - width / 2, c + 0.6, str(c), ha="center", color=INK, fontsize=10)
        ax.text(i + width / 2, d + 0.6, str(d), ha="center", color=INK, fontsize=10)

    _style(ax, "Velocity by sprint", "", "Story points")
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=INK, fontsize=10.5)
    ax.set_ylim(0, max(committed) * 1.18)
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, fontsize=9.5,
              framealpha=1)
    _save(fig, "sprint_velocity.png")


def plot_cumulative_flow() -> None:
    """Draw the Sprint 2 cumulative flow diagram."""
    days = CUMULATIVE_FLOW["days"]
    columns = ["Done", "In progress", "To do"]
    colours = [GOOD, BLUE, INK_MUTED]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.stackplot(
        days,
        *[CUMULATIVE_FLOW[c] for c in columns],
        labels=columns,
        colors=colours,
        alpha=0.82,
        edgecolor=SURFACE,
        linewidth=1.5,
    )

    _style(ax, "Sprint 2 cumulative flow", "Sprint day", "Stories")
    ax.set_xticks(days)
    ax.set_xlim(0, max(days))
    ax.legend(loc="upper left", facecolor=SURFACE, edgecolor=GRID,
              labelcolor=INK, fontsize=9.5, framealpha=1)
    ax.text(
        0.985, 0.06,
        "A steady Done band and a thin In-progress band\nindicate work was "
        "finished rather than accumulated.",
        transform=ax.transAxes, color=INK_MUTED, fontsize=9, ha="right", va="bottom",
    )
    _save(fig, "cumulative_flow.png")


def main() -> int:
    """Generate every agile tracking chart."""
    plot_burndown(SPRINT_1, "burndown_chart.png")
    plot_burndown(SPRINT_2, "burndown_sprint2.png")
    plot_velocity()
    plot_cumulative_flow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
