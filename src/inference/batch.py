"""
Batch Scoring
=============
Vectorised scoring of many telemetry rows at once.

Both the fleet overview and the upload page need the same journey: raw columns
in, engineered and scaled features, calibrated probability out, then a risk score
and band. That logic used to live inside a Streamlit callback, which made it
untestable and impossible to reuse. It lives here instead — no Streamlit import,
so it can be unit-tested and called from a script or an API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Internal feature names the model is trained on, in the order a user supplies them.
BASE_FEATURES = (
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)


def normalise_type_column(values: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    """Coerce a product-type column to its integer encoding.

    Accepts the raw ``L``/``M``/``H`` letters, the already-encoded integers, or a
    mix of both. Anything unrecognised falls back to the middle tier rather than
    dropping the row, since type is the least influential feature.

    Args:
        values: The column as supplied by the user.
        config: Project configuration, for ``preprocessing.type_encoding``.

    Returns:
        An integer Series aligned to ``values``.
    """
    encoding = config.get("preprocessing", {}).get("type_encoding", {"L": 0, "M": 1, "H": 2})
    default = encoding.get("M", 1)

    def convert(value: Any) -> int:
        if isinstance(value, str):
            return encoding.get(value.strip().upper(), default)
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            return int(value)
        return default

    return values.map(convert).astype(int)


def build_feature_frame(
    df: pd.DataFrame,
    column_mapping: Mapping[str, str],
    config: Mapping[str, Any],
    feature_stats: Mapping[str, float],
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Map, engineer and align raw columns into the model's feature matrix.

    Args:
        df: The source rows.
        column_mapping: Internal feature name -> source column name.
        config: Project configuration.
        feature_stats: Percentile thresholds fitted during training. Reusing the
            training statistics is what keeps engineered flags comparable.
        feature_names: The exact feature order the model expects.

    Returns:
        A DataFrame whose columns are ``feature_names``, in that order.
    """
    from src.features.engineer import engineer_features

    mapped = pd.DataFrame(index=df.index)
    for internal_name, source_col in column_mapping.items():
        if internal_name == "type":
            mapped[internal_name] = normalise_type_column(df[source_col], config)
        else:
            mapped[internal_name] = pd.to_numeric(df[source_col], errors="coerce")

    # A NaN here means a value the user supplied could not be parsed. Fill with
    # the column median so one malformed cell does not void the whole batch, and
    # say so in the log rather than silently substituting zero.
    for col in mapped.columns:
        n_missing = int(mapped[col].isna().sum())
        if n_missing:
            fill = float(mapped[col].median()) if mapped[col].notna().any() else 0.0
            logger.warning("Column %s had %d unparseable values; filled with %.3f", col, n_missing, fill)
            mapped[col] = mapped[col].fillna(fill)

    if "type" not in mapped.columns:
        mapped["type"] = config.get("preprocessing", {}).get("type_encoding", {}).get("M", 1)

    engineered, _ = engineer_features(mapped, config, fit_stats=dict(feature_stats))

    feature_names = list(feature_names)
    for col in feature_names:
        if col not in engineered.columns:
            logger.warning("Feature %s was absent after engineering; defaulting to 0.", col)
            engineered[col] = 0.0

    return engineered[feature_names]


def score_frame(
    df: pd.DataFrame,
    column_mapping: Mapping[str, str],
    model: Any,
    scaler: Any,
    config: Mapping[str, Any],
    feature_stats: Mapping[str, float],
    feature_names: Iterable[str],
    numerical_cols: Iterable[str],
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Score a batch of telemetry rows.

    Args:
        df: Source rows.
        column_mapping: Internal feature name -> source column name.
        model: A fitted classifier exposing ``predict_proba``.
        scaler: The scaler fitted during training.
        config: Project configuration.
        feature_stats: Training-time feature statistics.
        feature_names: Model feature order.
        numerical_cols: Columns the scaler was fitted on.
        decision_threshold: Probability at or above which a row is called a
            failure. Exposed because the cost-optimal threshold is rarely 0.5 on
            an imbalanced problem.

    Returns:
        A DataFrame indexed like ``df`` with ``failure_probability`` (0-1),
        ``risk_score`` (0-100), ``risk_category`` and ``prediction``.
    """
    from src.preprocessing.pipeline import apply_scaler
    from src.risk.scoring import compute_risk_score, get_risk_category

    X = build_feature_frame(df, column_mapping, config, feature_stats, feature_names)
    X_scaled = apply_scaler(X, scaler, dict(config), feature_cols=list(numerical_cols))

    probabilities = model.predict_proba(X_scaled)[:, 1]
    risk_scores = np.array([compute_risk_score(float(p), dict(config)) for p in probabilities])
    categories = [get_risk_category(float(s), dict(config))["label"] for s in risk_scores]

    return pd.DataFrame(
        {
            "failure_probability": probabilities,
            "risk_score": np.round(risk_scores, 1),
            "risk_category": categories,
            "prediction": np.where(probabilities >= decision_threshold, "FAILURE", "NO FAILURE"),
        },
        index=df.index,
    )


def default_mapping_for_raw_dataset(df: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, str]:
    """Derive a column mapping for the project's own raw dataset.

    The raw AI4I column headers carry units (``Air temperature [K]``); config
    already records how those map to internal names, so reuse it rather than
    restating the mapping.

    Args:
        df: The raw dataset.
        config: Project configuration, for ``data.feature_names``.

    Returns:
        Internal feature name -> the matching column present in ``df``.
    """
    rename_map = config["data"]["feature_names"]
    mapping: Dict[str, str] = {}

    for raw_col, internal in rename_map.items():
        if internal in mapping or internal == "machine_failure":
            continue
        if raw_col in df.columns:
            mapping[internal] = raw_col

    # Fall back to any column already carrying the internal name.
    for internal in (*BASE_FEATURES, "type"):
        if internal not in mapping and internal in df.columns:
            mapping[internal] = internal

    return mapping


def score_raw_dataset(
    df: pd.DataFrame,
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str = "best_model_calibrated",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score the project's own dataset with the trained model.

    Convenience wrapper for the dashboard's fleet views, which need a risk
    distribution over real assets rather than a hardcoded health figure.

    Args:
        df: The raw dataset.
        artifacts: Loaded model artifacts.
        config: Project configuration.
        model_key: Which artifact to score with. Defaults to the calibrated model
            so the probabilities are meaningful as probabilities.
        sample_size: Score a random subset of this size instead of everything.
        random_state: Seed for that subset.

    Returns:
        ``df`` joined with the scoring columns from :func:`score_frame`.
    """
    frame = df if sample_size is None or sample_size >= len(df) else df.sample(
        n=sample_size, random_state=random_state
    )

    model = artifacts.get(model_key) or artifacts["best_model"]
    scored = score_frame(
        frame,
        default_mapping_for_raw_dataset(frame, config),
        model,
        artifacts["scaler"],
        config,
        artifacts["feature_stats"],
        artifacts["feature_names"],
        artifacts["numerical_cols"],
    )
    return frame.join(scored)


__all__ = [
    "BASE_FEATURES",
    "normalise_type_column",
    "build_feature_frame",
    "score_frame",
    "score_raw_dataset",
    "default_mapping_for_raw_dataset",
]
