"""
Tests for the shared batch inference helpers.

These cover the path the dashboard and the upload page both depend on, so a
regression here would break scoring everywhere at once.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_config
from src.inference.batch import (
    build_feature_frame,
    default_mapping_for_raw_dataset,
    normalise_type_column,
    score_frame,
)


@pytest.fixture(scope="module")
def config():
    """Project configuration."""
    return load_config("config/config.yaml")


@pytest.fixture
def raw_frame():
    """A small frame using the dataset's own column spellings."""
    return pd.DataFrame(
        {
            "Air temperature [K]": [298.1, 302.0, 300.5],
            "Process temperature [K]": [308.6, 314.0, 311.0],
            "Rotational speed [rpm]": [1551, 1100, 1400],
            "Torque [Nm]": [42.8, 72.0, 55.0],
            "Tool wear [min]": [0, 220, 150],
            "Type": ["M", "L", "H"],
        }
    )


class TestNormaliseTypeColumn:
    """Product-type coercion."""

    def test_maps_letters(self, config):
        result = normalise_type_column(pd.Series(["L", "M", "H"]), config)
        assert result.tolist() == [0, 1, 2]

    def test_is_case_and_space_insensitive(self, config):
        result = normalise_type_column(pd.Series([" l ", "m", "H"]), config)
        assert result.tolist() == [0, 1, 2]

    def test_passes_through_integers(self, config):
        result = normalise_type_column(pd.Series([0, 1, 2]), config)
        assert result.tolist() == [0, 1, 2]

    def test_unknown_value_falls_back_to_middle_tier(self, config):
        # Type is the least influential feature, so an unrecognised grade should
        # not discard the row.
        result = normalise_type_column(pd.Series(["Z", None]), config)
        assert result.tolist() == [1, 1]

    def test_returns_integer_dtype(self, config):
        assert normalise_type_column(pd.Series(["L", "H"]), config).dtype.kind == "i"


class TestDefaultMapping:
    """Mapping the project's own dataset columns."""

    def test_maps_every_base_feature(self, raw_frame, config):
        mapping = default_mapping_for_raw_dataset(raw_frame, config)
        for name in ("air_temp_k", "process_temp_k", "rotational_speed_rpm",
                     "torque_nm", "tool_wear_min", "type"):
            assert name in mapping

    def test_never_maps_the_target(self, raw_frame, config):
        assert "machine_failure" not in default_mapping_for_raw_dataset(raw_frame, config)

    def test_accepts_internal_names_directly(self, config):
        frame = pd.DataFrame(
            {
                "air_temp_k": [298.0], "process_temp_k": [308.0],
                "rotational_speed_rpm": [1500], "torque_nm": [40.0],
                "tool_wear_min": [10], "type": [1],
            }
        )
        mapping = default_mapping_for_raw_dataset(frame, config)
        assert mapping["torque_nm"] == "torque_nm"


class TestBuildFeatureFrame:
    """Feature assembly and alignment."""

    def test_returns_features_in_model_order(self, raw_frame, config):
        mapping = default_mapping_for_raw_dataset(raw_frame, config)
        names = ["torque_nm", "air_temp_k", "temp_diff", "power"]
        stats = {"high_torque_threshold": 50.0, "low_speed_threshold": 1200.0}
        out = build_feature_frame(raw_frame, mapping, config, stats, names)
        assert list(out.columns) == names

    def test_fills_unparseable_values_rather_than_dropping_rows(self, config):
        frame = pd.DataFrame(
            {
                "air_temp_k": [298.0, "bad", 300.0],
                "process_temp_k": [308.0, 309.0, 310.0],
                "rotational_speed_rpm": [1500, 1400, 1300],
                "torque_nm": [40.0, 45.0, 50.0],
                "tool_wear_min": [10, 20, 30],
                "type": [1, 1, 1],
            }
        )
        mapping = {c: c for c in frame.columns}
        stats = {"high_torque_threshold": 50.0, "low_speed_threshold": 1200.0}
        out = build_feature_frame(frame, mapping, config, stats, ["air_temp_k"])
        assert len(out) == 3
        assert out["air_temp_k"].notna().all()

    def test_missing_engineered_column_defaults_to_zero(self, raw_frame, config):
        mapping = default_mapping_for_raw_dataset(raw_frame, config)
        stats = {"high_torque_threshold": 50.0, "low_speed_threshold": 1200.0}
        out = build_feature_frame(raw_frame, mapping, config, stats, ["not_a_real_feature"])
        assert (out["not_a_real_feature"] == 0.0).all()


class TestScoreFrame:
    """End-to-end scoring against a stub model."""

    class _StubModel:
        """Returns a fixed probability per row, highest first."""

        def __init__(self, probabilities):
            self._probabilities = np.asarray(probabilities)

        def predict_proba(self, X):
            p = self._probabilities[: len(X)]
            return np.column_stack([1 - p, p])

    class _StubScaler:
        """Identity scaler — isolates the test from fitted transformer state."""

        def transform(self, X):
            return np.asarray(X, dtype=float)

    def _score(self, raw_frame, config, probabilities, **kwargs):
        mapping = default_mapping_for_raw_dataset(raw_frame, config)
        stats = {"high_torque_threshold": 50.0, "low_speed_threshold": 1200.0}
        names = ["air_temp_k", "process_temp_k", "rotational_speed_rpm",
                 "torque_nm", "tool_wear_min", "type"]
        return score_frame(
            raw_frame, mapping, self._StubModel(probabilities), self._StubScaler(),
            config, stats, names, names, **kwargs
        )

    def test_returns_expected_columns(self, raw_frame, config):
        out = self._score(raw_frame, config, [0.9, 0.5, 0.1])
        assert list(out.columns) == [
            "failure_probability", "risk_score", "risk_category", "prediction"
        ]

    def test_preserves_the_source_index(self, raw_frame, config):
        frame = raw_frame.set_axis([10, 20, 30])
        out = self._score(frame, config, [0.9, 0.5, 0.1])
        assert out.index.tolist() == [10, 20, 30]

    def test_probabilities_pass_through_unchanged(self, raw_frame, config):
        out = self._score(raw_frame, config, [0.9, 0.5, 0.1])
        assert out["failure_probability"].tolist() == pytest.approx([0.9, 0.5, 0.1])

    def test_risk_score_is_monotonic_in_probability(self, raw_frame, config):
        out = self._score(raw_frame, config, [0.9, 0.5, 0.1])
        scores = out["risk_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_risk_scores_stay_in_range(self, raw_frame, config):
        out = self._score(raw_frame, config, [1.0, 0.5, 0.0])
        assert out["risk_score"].between(0, 100).all()

    def test_default_threshold_splits_at_half(self, raw_frame, config):
        out = self._score(raw_frame, config, [0.9, 0.5, 0.1])
        assert out["prediction"].tolist() == ["FAILURE", "FAILURE", "NO FAILURE"]

    def test_threshold_is_configurable(self, raw_frame, config):
        # The cost-optimal threshold is rarely 0.5 on an imbalanced problem, so
        # callers must be able to move it.
        out = self._score(raw_frame, config, [0.9, 0.5, 0.1], decision_threshold=0.95)
        assert out["prediction"].tolist() == ["NO FAILURE"] * 3

    def test_categories_come_from_config(self, raw_frame, config):
        out = self._score(raw_frame, config, [0.99, 0.5, 0.001])
        valid = {c["label"] for c in config["risk"]["categories"].values()}
        assert set(out["risk_category"]).issubset(valid)
