"""
Report Metrics
==============
Collects every figure a report quotes, measured from the artifacts on disk.

The previous report generator asserted its numbers as string literals — a model
score, a saving, a defect reduction — none of which were computed from anything.
That is the failure mode this module exists to prevent: a report can only state
what :func:`collect_metrics` actually measured, and the function raises rather
than inventing a value when an artifact is missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _results_dir(config: Dict[str, Any], project_root: Path) -> Path:
    return project_root / config["artifacts"]["results_dir"]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning ``None`` when it has not been generated."""
    if not path.exists():
        logger.info("Not present, skipping: %s", path.name)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dataset_facts(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the dataset the models were trained on.

    Args:
        df: The raw dataset.
        config: Project configuration.

    Returns:
        Row counts, class balance and the recorded failure-mode breakdown.
    """
    target = config["data"]["target_column"]
    n = len(df)
    failures = int(df[target].sum())

    modes = {}
    labels = {
        "TWF": "Tool wear failure",
        "HDF": "Heat dissipation failure",
        "PWF": "Power failure",
        "OSF": "Overstrain failure",
        "RNF": "Random failure",
    }
    for code, label in labels.items():
        if code in df.columns:
            modes[label] = int(df[code].sum())

    tiers = {}
    if "Type" in df.columns:
        share = df["Type"].value_counts(normalize=True) * 100
        tiers = {str(k): round(float(v), 1) for k, v in share.items()}

    return {
        "n_assets": n,
        "n_columns": len(df.columns),
        "failures": failures,
        "failure_rate_pct": round(failures / n * 100, 2) if n else 0.0,
        "imbalance_ratio": round((n - failures) / failures, 1) if failures else None,
        "failure_modes": modes,
        "tier_mix_pct": tiers,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def model_facts(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Extract measured model performance from the saved artifacts.

    Args:
        artifacts: Loaded model artifacts.

    Returns:
        The leaderboard and the selected model's headline scores.

    Raises:
        ValueError: If the artifacts carry no test results to report.
    """
    results: List[Dict[str, Any]] = artifacts.get("test_results", [])
    if not results:
        raise ValueError(
            "Model artifacts contain no test results. Run the training pipeline "
            "before generating a report."
        )

    best_name = artifacts.get("best_model_name", results[0]["model"])
    best = next((r for r in results if r["model"] == best_name), results[0])

    return {
        "profile": artifacts.get("training_profile", "unknown"),
        "best_model": best_name,
        "n_candidates": len(results),
        "leaderboard": [
            {
                "model": r["model"],
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
                "pr_auc": round(r["pr_auc"], 4),
                "roc_auc": round(r["roc_auc"], 4),
                "brier": round(r["brier_score"], 4),
            }
            for r in results
        ],
        "best": {
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "f1": round(best["f1"], 4),
            "pr_auc": round(best["pr_auc"], 4),
            "roc_auc": round(best["roc_auc"], 4),
            "brier": round(best["brier_score"], 4),
            "confusion_matrix": best.get("confusion_matrix"),
        },
        "features": list(artifacts.get("feature_names", [])),
    }


def tool_wear_facts(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Measure the tool-wear failure relationship and the interval it implies."""
    from app.pages.ceiling_inspection import recommended_change_point, tool_wear_failure_profile

    profile_fn = getattr(tool_wear_failure_profile, "__wrapped__", tool_wear_failure_profile)
    profile = profile_fn(df, config["data"]["target_column"])
    finding = recommended_change_point(profile)

    return {
        "finding": finding,
        "profile": [
            {
                "bucket": str(row["bucket"]),
                "assets": int(row["assets"]),
                "failures": int(row["failures"]),
                "failure_rate": round(float(row["failure_rate"]), 2),
            }
            for _, row in profile.iterrows()
        ],
    }


def cost_facts(
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    n_assets: int,
) -> Dict[str, Any]:
    """Price the selected model's confusion matrix using the configured costs.

    Deliberately states the result **per dataset** rather than per year. The
    dataset is a set of production cycles, not a fleet observed for twelve
    months, so annualising it would require a production-volume assumption the
    data does not contain. Callers that know their real cycle rate can scale the
    per-cycle figure themselves; the platform subscription is reported separately
    for the same reason, since it is the one genuinely annual quantity.

    Args:
        artifacts: Loaded model artifacts.
        config: Project configuration, for the ``business`` block.
        n_assets: Number of cycles in the dataset.

    Returns:
        The cost comparison, or an empty dict if no confusion matrix is stored.
    """
    results = artifacts.get("test_results", [])
    best_name = artifacts.get("best_model_name")
    best = next((r for r in results if r["model"] == best_name), None)
    if not best or not best.get("confusion_matrix"):
        return {}

    business = config.get("business", {})
    downtime_rate = business.get("downtime_cost_per_hour", 10000.0)
    downtime_hours = business.get("avg_downtime_hours", 4.0)
    preventive = business.get("preventive_action_cost", 1500.0)
    false_alarm = business.get("false_alarm_cost", 1500.0)
    licence = business.get("license_cost_monthly", 2500.0) * 12

    cm = best["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    tested = tn + fp + fn + tp
    scale = n_assets / tested if tested else 1.0

    failure_cost = downtime_rate * downtime_hours

    # Stated over the whole dataset: the test split scaled to the full set of
    # cycles. No time unit is attached, because the data carries none.
    reactive = (tp + fn) * failure_cost * scale
    predictive = (
        tp * preventive * scale + fp * false_alarm * scale + fn * failure_cost * scale
    )
    avoided = reactive - predictive

    return {
        "unit_failure_cost": failure_cost,
        "downtime_rate": downtime_rate,
        "downtime_hours": downtime_hours,
        "preventive_cost": preventive,
        "false_alarm_cost": false_alarm,
        "annual_platform_cost": licence,
        "cycles": n_assets,
        "test_split_size": tested,
        "scale_factor": round(scale, 2),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "reactive_cost": round(reactive, 2),
        "predictive_cost": round(predictive, 2),
        "avoided_cost": round(avoided, 2),
        "avoided_per_cycle": round(avoided / n_assets, 2) if n_assets else 0.0,
        "reduction_pct": round(avoided / reactive * 100, 1) if reactive else 0.0,
        "caught_pct": round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0.0,
        # How many cycles the platform must cover before it pays for itself.
        "breakeven_cycles": (
            int(licence / (avoided / n_assets)) if n_assets and avoided > 0 else None
        ),
    }


def collect_metrics(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Gather every measured figure a report needs.

    Args:
        project_root: Repository root. Defaults to this package's grandparent.

    Returns:
        A nested dict of measurements: ``dataset``, ``models``, ``tool_wear``,
        ``cost``, ``calibration``, ``ablation`` and ``shap``.

    Raises:
        FileNotFoundError: If model artifacts have not been generated.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    from src.data.loader import load_config, load_dataset
    from src.models.trainer import load_model_artifacts

    config = load_config(str(project_root / "config" / "config.yaml"))
    artifacts = load_model_artifacts(config, project_root)
    df = load_dataset(config)

    results_dir = _results_dir(config, project_root)
    dataset = dataset_facts(df, config)

    metrics: Dict[str, Any] = {
        "project": config["project"],
        "dataset": dataset,
        "models": model_facts(artifacts),
        "tool_wear": tool_wear_facts(df, config),
        "cost": cost_facts(artifacts, config, dataset["n_assets"]),
        "calibration": _load_json(results_dir / "calibration_results.json"),
        "ablation": _load_json(results_dir / "ablation_results.json"),
        "shap": _load_json(results_dir / "shap_analysis.json"),
        "inspection": config.get("inspection", {}),
        "risk_thresholds": config["risk"]["thresholds"],
    }
    return metrics


__all__ = [
    "collect_metrics",
    "dataset_facts",
    "model_facts",
    "tool_wear_facts",
    "cost_facts",
]
