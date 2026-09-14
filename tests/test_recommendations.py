"""
Unit Tests — Recommendation Engine
=====================================
Tests recommendation generation and formatting.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_config
from src.recommendations.engine import (
    generate_recommendations,
    format_recommendations_text,
    get_recommendation_summary,
)


@pytest.fixture
def config():
    return load_config(str(Path(__file__).resolve().parent.parent / "config" / "config.yaml"))


@pytest.fixture
def high_risk_explanation():
    """SHAP explanation with high-risk factors."""
    return {
        "top_factors": [
            {
                "feature": "tool_wear_min",
                "value": 200.0,
                "shap_value": 0.5,
                "impact": "increases risk",
                "abs_importance": 0.5,
            },
            {
                "feature": "torque_nm",
                "value": 70.0,
                "shap_value": 0.35,
                "impact": "increases risk",
                "abs_importance": 0.35,
            },
            {
                "feature": "temp_diff",
                "value": 12.0,
                "shap_value": 0.2,
                "impact": "increases risk",
                "abs_importance": 0.2,
            },
        ],
    }


@pytest.fixture
def low_risk_explanation():
    """SHAP explanation with no significant risk factors."""
    return {
        "top_factors": [
            {
                "feature": "air_temp_k",
                "value": 298.0,
                "shap_value": -0.1,
                "impact": "decreases risk",
                "abs_importance": 0.1,
            },
        ],
    }


class TestGenerateRecommendations:
    """Test recommendation generation."""

    def test_generates_for_high_risk(self, high_risk_explanation, config):
        recs = generate_recommendations(high_risk_explanation, config)
        assert len(recs) > 0

    def test_tool_wear_recommendation(self, high_risk_explanation, config):
        recs = generate_recommendations(high_risk_explanation, config)
        tool_rec = next((r for r in recs if r["rule"] == "tool_wear"), None)
        assert tool_rec is not None
        assert len(tool_rec["recommendations"]) > 0

    def test_no_recs_for_low_risk(self, low_risk_explanation, config):
        recs = generate_recommendations(low_risk_explanation, config)
        assert len(recs) == 0

    def test_priority_ordering(self, high_risk_explanation, config):
        recs = generate_recommendations(high_risk_explanation, config)
        if len(recs) >= 2:
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            for i in range(len(recs) - 1):
                p1 = priority_order.get(recs[i]["priority"], 99)
                p2 = priority_order.get(recs[i + 1]["priority"], 99)
                assert p1 <= p2

    def test_has_plant_context(self, high_risk_explanation, config):
        recs = generate_recommendations(high_risk_explanation, config)
        for rec in recs:
            assert "plant_context" in rec


class TestFormatRecommendations:
    """Test recommendation formatting."""

    def test_format_with_recs(self, high_risk_explanation, config):
        recs = generate_recommendations(high_risk_explanation, config)
        text = format_recommendations_text(recs)
        assert "RECOMMENDATION" in text.upper()
        assert "DISCLAIMER" in text.upper()

    def test_format_empty(self):
        text = format_recommendations_text([])
        assert "No significant risk" in text


class TestGetSummary:
    """Test recommendation summary."""

    def test_summary_with_recs(self, high_risk_explanation, config):
        recs = generate_recommendations(high_risk_explanation, config)
        summary = get_recommendation_summary(recs)
        assert len(summary) > 0

    def test_summary_empty(self):
        summary = get_recommendation_summary([])
        assert "No immediate action" in summary[0]
