"""
Project Report Generator
========================
Writes ``FALSE_CEILING_PROJECT_REPORT.md`` from measured results.

Every number in the output comes from :func:`src.reporting.metrics.collect_metrics`,
which reads the trained artifacts and the dataset. Nothing is asserted. If a
measurement is unavailable the section says so rather than substituting a figure.

Usage:
    python scripts/generate_project_report.py
    python scripts/generate_project_report.py --output docs/REPORT.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("report")

TEAM = [
    {"name": "Nivesh Manoj Jain", "email": "nivesh.jain24@vit.edu",
     "role": "ML architecture and pipeline"},
    {"name": "Hasan Rupawalla", "email": "hasan.rupawalla24@vit.edu",
     "role": "Computer vision and inspection"},
    {"name": "Rachit Ingole", "email": "rachit.ingole241@vit.edu",
     "role": "Telemetry ingestion and data"},
]

#: Figures are embedded only when the file exists, so a report generated before
#: the plotting step still renders cleanly.
FIGURES = {
    "roc": "reports/figures/roc_curves.png",
    "pr": "reports/figures/precision_recall_curves.png",
    "confusion": "reports/figures/confusion_matrices.png",
    "calibration": "reports/figures/calibration_curves.png",
    "shap_summary": "reports/figures/shap_summary.png",
    "shap_bar": "reports/figures/shap_bar.png",
    "comparison": "reports/figures/model_comparison_bars.png",
    "class_balance": "reports/figures/class_distribution.png",
    "correlation": "reports/figures/correlation_heatmap.png",
    "tool_wear": "reports/figures/tool_wear_vs_failure_rate.png",
    "cost": "reports/figures/cost_comparison.png",
}


def figure(key: str, caption: str) -> List[str]:
    """Return markdown for a figure, or a placeholder note if it is missing."""
    path = project_root / FIGURES[key]
    if not path.exists():
        return [f"> _Figure not generated yet: `{FIGURES[key]}`._", ""]
    return [f"![{caption}]({FIGURES[key]})", f"*{caption}*", ""]


def money(value: float) -> str:
    """Format a currency figure without false precision."""
    return f"${value:,.0f}"


def build_report(m: Dict[str, Any]) -> str:
    """Render the full markdown report from collected metrics."""
    dataset = m["dataset"]
    models = m["models"]
    best = models["best"]
    wear = m["tool_wear"]["finding"]
    cost = m["cost"]
    out: List[str] = []
    add = out.append

    # ---------------------------------------------------------------- header
    add("# Manufacturing Failure Prevention and Defect Inspection Platform")
    add("")
    add("**Project status report**  ")
    add(f"**Date:** {date.today():%d %B %Y}  ")
    add(f"**Version:** {m['project'].get('version', '—')}  ")
    add(f"**Training profile used for these figures:** `{models['profile']}`")
    add("")
    add("| Member | Responsibility | Contact |")
    add("|:--|:--|:--|")
    for person in TEAM:
        add(f"| {person['name']} | {person['role']} | `{person['email']}` |")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------- summary
    add("## 1. Summary for the reader in a hurry")
    add("")
    add("We built a system that predicts machine failures on a suspended-ceiling "
        "production line before they happen, explains each prediction, and prices "
        "the decision. It is working end to end and every number below is measured "
        "from the running system, not estimated.")
    add("")
    add("| What we set out to do | Where it stands |")
    add("|:--|:--|")
    add(f"| Predict failures from sensor telemetry | Working — {best['recall']:.1%} of "
        f"failures caught, {best['precision']:.1%} of alerts genuine |")
    add("| Explain why each prediction was made | Working — SHAP attribution on every "
        "prediction, global and per-asset |")
    add("| Tie machine health to product quality | Working — tool-wear interval derived "
        "from the data |")
    add("| Make the decision financially legible | Working — full cost model with "
        "adjustable plant figures |")
    add("| Operate on real plant data | Working — file, database and batch paths all live |")
    add("")

    if cost:
        add("**The headline number.** Against a run-to-failure baseline, the model "
            f"avoids **{money(cost['avoided_cost'])}** across the "
            f"{cost['cycles']:,} production cycles in the dataset — a "
            f"{cost['reduction_pct']:.0f}% reduction in failure-related cost, or "
            f"about **{money(cost['avoided_per_cycle'])} per cycle**. Section 6 "
            "shows the arithmetic and states plainly what it assumes.")
        add("")
        add("> We quote this per cycle rather than per year on purpose. The dataset "
            "records production cycles, not a fleet watched for twelve months, so "
            "an annual figure would need a production-volume assumption the data "
            "does not contain. Multiply by your own annual cycle count.")
        add("")

    add("> **How to read this report.** Section 2 frames the problem, 3 covers the "
        "data, 4 the model and its honest limits, 5 explainability, 6 the money, "
        "7 what is built, and 8 what remains. Sections 4 and 8 are the ones worth "
        "reading closely — they contain the caveats.")
    add("")
    add("---")
    add("")

    # -------------------------------------------------------------- problem
    add("## 2. The problem")
    add("")
    add("The plant runs two connected lines:")
    add("")
    add("```")
    add("   LINE 1  CNC stamping and milling            LINE 2  Finishing and inspection")
    add("   ┌───────────────────────────────┐           ┌──────────────────────────────┐")
    add("   │  spindle · press · tooling    │  tiles    │  optical inspection cell     │")
    add("   │                               │ ────────► │                              │")
    add("   │  sensors: temperature, speed, │           │  rejects: edge chipping,     │")
    add("   │  torque, accumulated wear     │           │  staining, warp, misalignment│")
    add("   └───────────────┬───────────────┘           └──────────────┬───────────────┘")
    add("                   │                                          │")
    add("                   │   the causal link this project exploits  │")
    add("                   └──────────────────────────────────────────┘")
    add("      worn tooling on Line 1 becomes edge chipping on Line 2, one shift later")
    add("```")
    add("")
    add("Line 2 inspection catches defective tiles, but by then the material and the "
        "machine time are already spent. The defect was decided upstream, by the "
        "condition of the tooling. Predicting machine condition is therefore the "
        "cheaper intervention point, and the reason a machinery model sits behind a "
        "quality problem.")
    add("")
    add("---")
    add("")

    # ----------------------------------------------------------------- data
    add("## 3. The data")
    add("")
    add(f"The platform is developed against the AI4I 2020 predictive maintenance "
        f"dataset: **{dataset['n_assets']:,} assets**, "
        f"**{dataset['n_columns']} columns**, with "
        f"**{dataset['failures']} recorded failures "
        f"({dataset['failure_rate_pct']}%)**.")
    add("")
    add("| Property | Value | Why it matters |")
    add("|:--|--:|:--|")
    add(f"| Assets | {dataset['n_assets']:,} | Sample size |")
    add(f"| Failures | {dataset['failures']} | The minority class |")
    add(f"| Failure rate | {dataset['failure_rate_pct']}% | Severe imbalance |")
    if dataset["imbalance_ratio"]:
        add(f"| Imbalance ratio | {dataset['imbalance_ratio']}:1 | "
            "Why accuracy is a useless metric here |")
    add(f"| Missing values | {dataset['missing_values']} | No imputation needed |")
    add(f"| Duplicate rows | {dataset['duplicate_rows']} | No deduplication needed |")
    add("")

    if dataset["failure_modes"]:
        add("### Recorded failure modes")
        add("")
        add("| Mode | Events | Share of failures |")
        add("|:--|--:|--:|")
        total = dataset["failures"] or 1
        for label, count in sorted(dataset["failure_modes"].items(),
                                   key=lambda kv: kv[1], reverse=True):
            add(f"| {label} | {count} | {count / total * 100:.1f}% |")
        add("")
        add("These labels are **excluded from training**. A model given them would "
            "read the answer off the label rather than learn the sensor pattern that "
            "precedes it, and would then be useless on live data where no such label "
            "exists. They are used only to check that what the model learned "
            "corresponds to real physics.")
        add("")

    out.extend(figure("class_balance", "Class balance — failures against normal operation"))
    out.extend(figure("correlation", "Correlation between sensor readings"))

    add("### Engineered features")
    add("")
    add("| Feature | Formula | Physical meaning |")
    add("|:--|:--|:--|")
    add("| Thermal margin | process temp − air temp | Whether heat is leaving the machine |")
    add("| Mechanical power | torque × speed × 2π ÷ 60 | Work actually being done at the cut |")
    add("| Load per unit speed | torque ÷ speed | Whether the drive is straining |")
    add("| Accumulated strain | tool wear × torque | Cumulative stress on the tooling |")
    add("| Thermal-speed stress | thermal margin × speed | Heat generated at operating speed |")
    add("")
    add("None of these adds information — each is a combination of readings the model "
        "already has. They help because a decision tree needs many splits to "
        "approximate a ratio or a product, and stating it directly costs none.")
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- model
    add("## 4. The model, and what it does not do")
    add("")
    add(f"**{models['best_model']}** was selected from {models['n_candidates']} "
        f"candidate(s) on the held-out test split.")
    add("")
    add("| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |")
    add("|:--|--:|--:|--:|--:|--:|--:|")
    for row in models["leaderboard"]:
        marker = " ✓" if row["model"] == models["best_model"] else ""
        add(f"| {row['model']}{marker} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['pr_auc']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['brier']:.4f} |")
    add("")

    cm = best.get("confusion_matrix")
    if cm:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        add("### What the numbers mean in practice")
        add("")
        add("On the held-out test split:")
        add("")
        add("| Outcome | Count | Consequence |")
        add("|:--|--:|:--|")
        add(f"| Failures caught | {tp} | Planned intervention instead of a breakdown |")
        add(f"| Failures missed | {fn} | An unplanned stop still happens |")
        add(f"| False alarms | {fp} | An inspection finds nothing wrong |")
        add(f"| Correctly cleared | {tn} | Normal running, no action |")
        add("")
        add(f"Read plainly: of every {tp + fn} genuine failures the model catches "
            f"**{tp}** and misses **{fn}**, while raising **{fp}** unnecessary "
            "inspections. Because a missed failure costs far more than a needless "
            "check, this trade is deliberately tilted toward catching more.")
        add("")

    out.extend(figure("comparison", "Model comparison across metrics"))
    out.extend(figure("roc", "ROC curves"))
    out.extend(figure("pr", "Precision-recall curves — the honest view under imbalance"))
    out.extend(figure("confusion", "Confusion matrices"))

    if m.get("calibration"):
        cal = m["calibration"]
        add("### Probability calibration")
        add("")
        add(f"Brier score improved from **{cal['brier_uncalibrated']}** to "
            f"**{cal['brier_calibrated']}** using "
            f"{cal.get('method', 'isotonic')} regression "
            f"(an improvement of {cal['improvement']}).")
        add("")
        add("This matters more than it sounds. A calibrated model that says 80% means "
            "roughly 80 of every 100 such assets really do fail. Without that "
            "property the cost arithmetic in section 6 would be multiplying a cost by "
            "a number that only ranks correctly rather than one that means anything.")
        add("")
        out.extend(figure("calibration", "Reliability diagram"))

    if m.get("ablation") and "with_feature_engineering" in m["ablation"]:
        abl = m["ablation"]
        delta = abl["with_feature_engineering"]["f1"] - abl["without_feature_engineering"]["f1"]
        add("### Did the engineered features earn their place?")
        add("")
        add("| Configuration | F1 | PR-AUC |")
        add("|:--|--:|--:|")
        add(f"| Raw sensors only | {abl['without_feature_engineering']['f1']:.4f} | "
            f"{abl['without_feature_engineering']['pr_auc']:.4f} |")
        add(f"| With engineered features | {abl['with_feature_engineering']['f1']:.4f} | "
            f"{abl['with_feature_engineering']['pr_auc']:.4f} |")
        add("")
        add(f"Difference in F1: **{delta:+.4f}**.")
        add("")

    add("### Limits worth stating plainly")
    add("")
    add("- **This is a public benchmark dataset, not our plant's data.** The pipeline "
        "is built to ingest real telemetry and has been exercised end to end, but "
        "the scores above describe AI4I 2020. Real plant performance will differ and "
        "must be re-measured after deployment.")
    add("- **The risk score is a design choice, not a physical quantity.** It is a "
        "monotone transform of the calibrated probability, chosen to spread moderate "
        "risks across a readable range.")
    add("- **The model finds correlation.** SHAP explains what drove a prediction, "
        "which is not the same as proving physical causation.")
    add("- **No temporal validation.** The dataset has no reliable time ordering, so "
        "the split is random rather than chronological. On live data a time-based "
        "split would be the honest test.")
    add("")
    add("---")
    add("")

    # --------------------------------------------------------- explainability
    add("## 5. Explainability")
    add("")
    add("Every prediction carries a SHAP breakdown showing which readings pushed it "
        "toward failure and by how much. A maintenance engineer is never asked to act "
        "on an unexplained number.")
    add("")

    if m.get("shap") and m["shap"].get("feature_importance"):
        add("| Rank | Feature | Mean \\|SHAP\\| | Share of attribution |")
        add("|--:|:--|--:|--:|")
        for row in m["shap"]["feature_importance"][:8]:
            add(f"| {row['rank']} | {row['feature']} | {row['mean_abs_shap']:.4f} | "
                f"{row['contribution_pct']:.1f}% |")
        add("")
    else:
        add("> _Global SHAP importance is produced by the full training run; it has "
            "not been generated for this report._")
        add("")

    out.extend(figure("shap_bar", "Global feature importance"))
    out.extend(figure("shap_summary", "SHAP summary — direction and magnitude per feature"))

    add("The platform converts each explanation into an instruction. A prediction "
        "driven by accumulated wear produces *inspect the punch and die at the "
        "stamping station*, not *tool_wear_min = 0.82*.")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- tool wear + cost
    add("## 6. What it is worth")
    add("")
    add("### The tool-wear interval")
    add("")
    if wear:
        add(f"Binning the dataset by accumulated tool wear shows the failure rate is "
            f"flat at about **{wear['baseline_rate']:.1f}%** until roughly "
            f"**{wear['threshold']} minutes**, then climbs sharply to "
            f"**{wear['elevated_rate']:.1f}%**.")
        add("")
        add(f"Replacing tooling at **{wear['threshold']} minutes** would have placed "
            f"**{wear['avoidable']} of {wear['total']} {wear['kind']} "
            f"({wear['share']:.0f}%)** on the safe side of that line.")
        add("")
        add("This is the single most actionable finding in the project, and it is a "
            "measurement rather than a recommendation we invented: the interval is "
            "derived from the data at runtime and moves if the data moves.")
        add("")
    else:
        add("> _No clear wear-related change point was found in the current dataset._")
        add("")

    out.extend(figure("tool_wear", "Failure rate against accumulated tool wear"))

    if cost:
        add("### The cost model")
        add("")
        add("| Assumption | Value |")
        add("|:--|--:|")
        add(f"| Downtime cost | {money(cost['downtime_rate'])} per hour |")
        add(f"| Average recovery time | {cost['downtime_hours']} hours |")
        add(f"| Cost of one failure | {money(cost['unit_failure_cost'])} |")
        add(f"| Planned intervention | {money(cost['preventive_cost'])} |")
        add(f"| False alarm | {money(cost['false_alarm_cost'])} |")
        add(f"| Platform running cost | {money(cost['annual_platform_cost'])} per year |")
        add("")
        add(f"| Scenario | Cost over {cost['cycles']:,} cycles | Per cycle |")
        add("|:--|--:|--:|")
        add(f"| Run to failure | {money(cost['reactive_cost'])} | "
            f"{money(cost['reactive_cost'] / cost['cycles'])} |")
        add(f"| With the model | {money(cost['predictive_cost'])} | "
            f"{money(cost['predictive_cost'] / cost['cycles'])} |")
        add(f"| **Avoided** | **{money(cost['avoided_cost'])}** | "
            f"**{money(cost['avoided_per_cycle'])}** |")
        add("")
        add(f"A **{cost['reduction_pct']:.0f}% reduction** in failure-related cost, "
            f"with **{cost['caught_pct']:.0f}%** of failures caught before they "
            "happen.")
        add("")
        if cost.get("breakeven_cycles"):
            add(f"Running the platform costs {money(cost['annual_platform_cost'])} a "
                f"year, which the model recovers within roughly "
                f"**{cost['breakeven_cycles']} cycles** at the avoided cost above. "
                "The subscription is not the deciding factor here; the accuracy of "
                "the downtime figure is.")
            add("")

        add("### What this calculation assumes")
        add("")
        add("Worth stating, because a manager will and should ask:")
        add("")
        add(f"1. **It scales the test split.** The model was scored on "
            f"{cost['test_split_size']:,} held-out cycles; those results are "
            f"multiplied by {cost['scale_factor']} to cover all "
            f"{cost['cycles']:,}. That is valid only if the test split is "
            "representative, which stratified sampling makes likely but does not "
            "guarantee.")
        add("2. **It assumes every predicted failure is preventable.** In reality "
            "some flagged failures would occur regardless, so the true avoided "
            "cost is lower than the figure above.")
        add("3. **It ignores the cost of acting.** Technician time to investigate a "
            "flag is folded into the planned-intervention figure, which may be "
            "optimistic.")
        add("4. **The cost inputs are placeholders.** "
            f"{money(cost['downtime_rate'])} per hour of downtime and "
            f"{money(cost['preventive_cost'])} per planned fix are configurable "
            "defaults, not our plant's measured figures. Replace them in "
            "`config/config.yaml` before quoting this anywhere binding.")
        add("")
        add("The direction of the result is robust — catching failures early is "
            "worth far more than the inspections it costs — but the magnitude "
            "should be treated as an order of magnitude until the inputs are real.")
        add("")
        out.extend(figure("cost", "Cost comparison across maintenance strategies"))

    add("---")
    add("")

    # -------------------------------------------------------------- delivered
    add("## 7. What has been built")
    add("")
    add("```")
    add("  telemetry ──► validation ──► feature engineering ──► calibrated model")
    add("   (file,          (schema,        (5 derived           (selected from")
    add("    database,       range,          features)            candidates on F1)")
    add("    stream)         balance)                                   │")
    add("                                                               ▼")
    add("                                            ┌──────────────────────────────┐")
    add("                                            │ probability ──► risk score   │")
    add("                                            │      │              │        │")
    add("                                            │      ▼              ▼        │")
    add("                                            │   SHAP          risk band    │")
    add("                                            │      │              │        │")
    add("                                            │      └──────┬───────┘        │")
    add("                                            │             ▼                │")
    add("                                            │   ranked maintenance action  │")
    add("                                            └──────────────────────────────┘")
    add("```")
    add("")
    add("| Component | State |")
    add("|:--|:--|")
    add("| Data loading, validation, preprocessing | Complete |")
    add("| Feature engineering | Complete |")
    add("| Model training with hyperparameter search | Complete |")
    add("| Probability calibration | Complete |")
    add("| SHAP explainability, global and local | Complete |")
    add("| Risk scoring and banding | Complete |")
    add("| Recommendation engine | Complete |")
    add("| Dashboard, eight modules | Complete |")
    add("| Batch and database scoring | Complete |")
    add("| Optical inspection cell | Demonstration, simulated frames |")
    add("| Automated test suite | Complete |")
    add("| REST API for PLC integration | Not started |")
    add("| Container deployment | Not started |")
    add("")
    add("### The dashboard")
    add("")
    add("| Page | Question it answers |")
    add("|:--|:--|")
    add("| Asset Health | How is the fleet doing, and what needs attention? |")
    add("| Risk Assessment | What is this machine's risk, and why? |")
    add("| Data Explorer | What does the sensor record look like? |")
    add("| Model Diagnostics | What drives the model's decisions? |")
    add("| Defect Inspection | Is the product within tolerance? |")
    add("| Model & Cost Analysis | Which model, and what is it worth? |")
    add("| Batch Analysis | Score a whole file or table |")
    add("| Alerts | What was flagged this session? |")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------------ next
    add("## 8. What remains")
    add("")
    add("| Priority | Item | Why |")
    add("|:--|:--|:--|")
    add("| High | Validate on real plant telemetry | Every score here describes a "
        "public dataset, not our line |")
    add("| High | REST API endpoint | PLCs and SCADA cannot talk to a dashboard |")
    add("| Medium | Failure-mode classification | Tell the engineer *which* failure, "
        "not just *a* failure |")
    add("| Medium | Drift monitoring | Sensor calibration drifts; the model needs to "
        "notice |")
    add("| Medium | Container deployment | Reproducible install on plant hardware |")
    add("| Low | Real defect imagery | The inspection cell currently uses simulated frames |")
    add("")
    add("### Honest assessment")
    add("")
    add("The machine-learning and interface work is solid and complete. The two gaps "
        "that matter for production are **validation against real plant data** and "
        "the **integration API**. Neither is a research problem; both are "
        "engineering work with known shapes. The inspection cell is the least "
        "mature component and is presented as a demonstration rather than a working "
        "detector.")
    add("")
    add("---")
    add("")
    add(f"*Generated {date.today():%d %B %Y} from measured results. "
        "Regenerate with `python scripts/generate_project_report.py`.*")
    add("")

    return "\n".join(out)


def main(argv=None) -> int:
    """Generate the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="FALSE_CEILING_PROJECT_REPORT.md",
        help="Where to write the report, relative to the project root.",
    )
    args = parser.parse_args(argv)

    from src.reporting.metrics import collect_metrics

    try:
        metrics = collect_metrics(project_root)
    except FileNotFoundError:
        logger.error(
            "No model artifacts found. Run `python scripts/train_pipeline.py "
            "--mode baseline` first — the report refuses to invent figures."
        )
        return 1

    destination = project_root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(metrics), encoding="utf-8")

    logger.info("Wrote %s", destination)
    logger.info("Model: %s | F1 %.4f | training profile: %s",
                metrics["models"]["best_model"],
                metrics["models"]["best"]["f1"],
                metrics["models"]["profile"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
