"""
Cached Data Access
==================
Streamlit-aware wrappers around the expensive pipeline calls.

Preprocessing the dataset and scoring ten thousand rows are both fast enough to
do once and far too slow to redo on every widget interaction — and Streamlit
reruns the whole script on every interaction. Everything expensive is funnelled
through this module so the caching decisions live in one place.

Argument names beginning with an underscore are excluded from Streamlit's cache
key. Models, scalers and config dicts are unhashable, so they are passed that way
and the cache is keyed on the cheap arguments alongside them.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Preparing evaluation split...")
def get_test_split(_df: pd.DataFrame, _config: Mapping[str, Any], cache_key: str = "default"):
    """Return the held-out test split, preprocessed exactly as in training.

    The model-comparison page rebuilds ROC, PR, calibration and cost curves from
    this split. Without caching, every slider nudge re-ran the full preprocessing
    pipeline — splitting, engineering and scaling ten thousand rows — which made
    the cost sliders feel broken.

    Args:
        _df: Raw dataset.
        _config: Project configuration.
        cache_key: Bump to force a recompute.

    Returns:
        ``(X_test, y_test)``.
    """
    from src.preprocessing.pipeline import run_preprocessing_pipeline

    processed = run_preprocessing_pipeline(_df, dict(_config))
    return processed["X_test"], processed["y_test"]


@st.cache_data(show_spinner="Scoring fleet...")
def get_scored_fleet(
    _df: pd.DataFrame,
    _artifacts: Mapping[str, Any],
    _config: Mapping[str, Any],
    cache_key: str = "default",
) -> pd.DataFrame:
    """Score every asset in the dataset with the calibrated model.

    Args:
        _df: Raw dataset.
        _artifacts: Loaded model artifacts.
        _config: Project configuration.
        cache_key: Bump to force a recompute.

    Returns:
        The dataset joined with ``failure_probability``, ``risk_score``,
        ``risk_category`` and ``prediction``.
    """
    from src.inference.batch import score_raw_dataset

    return score_raw_dataset(_df, dict(_artifacts), dict(_config))


@st.cache_data(show_spinner=False)
def get_fleet_summary(
    _scored: pd.DataFrame,
    _config: Mapping[str, Any],
    target_column: str,
    cache_key: str = "default",
) -> Dict[str, Any]:
    """Derive the headline fleet figures from a scored frame.

    Every number the Asset Health page displays comes from here, so none of them
    are hardcoded and all of them move when the data or the model changes.

    Args:
        _scored: Output of :func:`get_scored_fleet`.
        _config: Project configuration.
        target_column: Name of the ground-truth failure column, if present.
        cache_key: Bump to force a recompute.

    Returns:
        A dict of summary statistics.
    """
    thresholds = _config["risk"]["thresholds"]
    n_assets = len(_scored)

    band_counts = _scored["risk_category"].value_counts().to_dict()
    n_elevated = int((_scored["risk_score"] > thresholds["low"]).sum())
    n_critical = int(band_counts.get("CRITICAL RISK", 0))

    summary: Dict[str, Any] = {
        "n_assets": n_assets,
        "band_counts": band_counts,
        "n_elevated": n_elevated,
        "n_critical": n_critical,
        # Share of the fleet the model places in the routine band. This replaces
        # the old hardcoded "system health index".
        "health_index": round(100.0 * (n_assets - n_elevated) / n_assets, 1) if n_assets else 0.0,
        "mean_risk": round(float(_scored["risk_score"].mean()), 1),
        "p95_risk": round(float(_scored["risk_score"].quantile(0.95)), 1),
    }

    if target_column in _scored.columns:
        actual = int(_scored[target_column].sum())
        summary["actual_failures"] = actual
        summary["actual_failure_rate"] = round(100.0 * actual / n_assets, 2) if n_assets else 0.0
        # Recall at the deployed decision threshold, stated plainly rather than
        # quoted from a report.
        flagged = _scored["prediction"] == "FAILURE"
        true_positives = int((flagged & (_scored[target_column] == 1)).sum())
        summary["caught_failures"] = true_positives
        summary["catch_rate"] = round(100.0 * true_positives / actual, 1) if actual else 0.0
        summary["flagged"] = int(flagged.sum())

    return summary


@st.cache_data(show_spinner=False)
def get_type_mix(_df: pd.DataFrame, column: str = "Type") -> Dict[str, float]:
    """Return the product-tier mix as percentages, computed from the data.

    Args:
        _df: Raw dataset.
        column: Name of the product-type column.

    Returns:
        Tier label -> percentage share. Empty if the column is absent.
    """
    if column not in _df.columns:
        return {}
    counts = _df[column].value_counts(normalize=True) * 100
    return {str(k): round(float(v), 1) for k, v in counts.items()}


def active_dataset(load_raw_dataset_fn) -> Tuple[pd.DataFrame, str]:
    """Return whichever dataset the user is currently working with.

    A batch upload or database sync replaces the bundled telemetry everywhere, so
    pages should ask for the dataset through this rather than calling the raw
    loader directly.

    Args:
        load_raw_dataset_fn: The cached raw-dataset loader from ``app/main.py``.

    Returns:
        ``(dataframe, source_label)``.
    """
    uploaded: Optional[pd.DataFrame] = st.session_state.get("uploaded_dataset")
    if uploaded is not None and not uploaded.empty:
        return uploaded, "Uploaded batch"
    return load_raw_dataset_fn(), "Plant telemetry"


__all__ = [
    "get_test_split",
    "get_scored_fleet",
    "get_fleet_summary",
    "get_type_mix",
    "active_dataset",
]
