"""
Recommendation Engine Module
=============================
Rule-based recommendation engine that maps SHAP feature contributions
to actionable maintenance recommendations.

IMPORTANT: All recommendations are labeled as "AI-generated decision-support
recommendations" — NOT certified engineering procedures.

Recommendations are triggered based on which features contribute most
strongly to a failure prediction, as identified by SHAP values.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ============================================================================
# Recommendation Rules
# ============================================================================

# Maps feature name patterns to recommendation rules
RECOMMENDATION_RULES = {
    "tool_wear": {
        "feature_patterns": ["tool_wear_min", "tool_wear_severity", "strain"],
        "condition": "increases risk",
        "title": "Tool Wear Alert",
        "icon": "🔧",
        "recommendations": [
            "Inspect tool condition and measure wear level.",
            "Consider scheduling tool replacement if wear exceeds maintenance threshold.",
            "Check if tool alignment has shifted due to prolonged use.",
            "Review tool replacement schedule and adjust if needed.",
        ],
        "priority": "high",
        "plant_context": (
            "Line 1, stamping station: measure the punch and die edge on the "
            "tile press. Worn tooling is the direct cause of the edge chipping "
            "and corner cracks that Line 2 inspection rejects."
        ),
    },
    "torque": {
        "feature_patterns": ["torque_nm", "torque_per_rpm", "power_factor", "power"],
        "condition": "increases risk",
        "title": "Load / Torque Alert",
        "icon": "⚙️",
        "recommendations": [
            "Inspect load conditions and check for mechanical resistance.",
            "Verify that material feed is consistent and not causing jams.",
            "Check drive belts, gears, and bearings for excess friction.",
            "Reduce operational load if safe to do so.",
        ],
        "priority": "high",
        "plant_context": (
            "Line 1, press drive: rising torque at the same feed rate means the "
            "board is resisting the cut. Check for binding in the feed rollers "
            "and for board density above specification."
        ),
    },
    "temperature": {
        "feature_patterns": [
            "air_temp_k", "process_temp_k", "temp_diff", "temp_rpm_interaction"
        ],
        "condition": "increases risk",
        "title": "Thermal Alert",
        "icon": "🌡️",
        "recommendations": [
            "Inspect thermal conditions and cooling system performance.",
            "Check for blocked ventilation or failed cooling fans.",
            "Verify ambient temperature is within operational limits.",
            "Allow equipment to cool before resuming if overheated.",
        ],
        "priority": "medium",
        "plant_context": (
            "Line 1, spindle and drive housing: a narrowing gap between process "
            "and ambient temperature means heat is not leaving the machine. "
            "Check coolant flow and extraction before the spindle bearing takes "
            "damage."
        ),
    },
    "speed": {
        "feature_patterns": ["rotational_speed_rpm", "is_low_speed"],
        "condition": "increases risk",
        "title": "Speed Anomaly Alert",
        "icon": "🔄",
        "recommendations": [
            "Investigate power supply stability and motor controller function.",
            "Check for mechanical binding that may reduce speed.",
            "Verify RPM sensor calibration.",
            "Inspect belt tension and coupling condition.",
        ],
        "priority": "medium",
        "plant_context": (
            "Line 1, spindle speed: a drop against the setpoint points at "
            "supply instability or mechanical drag. Sustained low speed with "
            "normal torque leaves a rough cut edge on the tile."
        ),
    },
    "overload": {
        "feature_patterns": ["overload_indicator", "is_high_torque"],
        "condition": "increases risk",
        "title": "Overload / Overstrain Alert",
        "icon": "⚡",
        "recommendations": [
            "Reduce operational load immediately if overstrain is detected.",
            "Schedule maintenance window to inspect machine components.",
            "Check for material defects that may increase process resistance.",
            "Review operating parameters against manufacturer specifications.",
        ],
        "priority": "critical",
        "plant_context": (
            "Line 1, overstrain: high torque at low speed is the signature of a "
            "stalling cut. Stop the line before the tool shatters and damages "
            "the die set."
        ),
    },
    "product_type": {
        "feature_patterns": ["type"],
        "condition": "increases risk",
        "title": "Product Type Risk",
        "icon": "📦",
        "recommendations": [
            "Review quality control procedures for this product variant.",
            "Check if process parameters are optimized for this product type.",
            "Consider additional quality inspection for high-risk product types.",
        ],
        "priority": "low",
        "plant_context": (
            "Board grade drives the cutting parameters. Confirm the programme "
            "loaded matches the grade running - economy mineral fibre and "
            "premium gypsum need different feed and speed."
        ),
    },
}


def generate_recommendations(
    shap_explanation: Dict[str, Any],
    config: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Generate maintenance recommendations based on SHAP feature contributions.

    Maps the top contributing features (from SHAP) to rule-based
    recommendations. Only features that increase risk trigger recommendations.

    Args:
        shap_explanation: SHAP explanation dictionary from SHAPEngine.
        config: Project configuration.
        top_n: Maximum recommendations to return.

    Returns:
        List of recommendation dictionaries.
    """
    recommendations = []
    seen_rules = set()

    top_factors = shap_explanation.get("top_factors", [])

    for factor in top_factors:
        feature_name = factor["feature"]
        impact = factor["impact"]

        # Only recommend action for risk-increasing features
        if impact != "increases risk":
            continue

        # Find matching rule
        for rule_key, rule in RECOMMENDATION_RULES.items():
            if rule_key in seen_rules:
                continue

            # Check if the feature matches any pattern for this rule
            matches = any(
                pattern in feature_name or feature_name in pattern
                for pattern in rule["feature_patterns"]
            )

            if matches:
                seen_rules.add(rule_key)

                # What this sensor reading means on the shop floor.
                station_label = config.get("industry_mapping", {}).get(
                    feature_name, "No station mapping configured."
                )

                recommendations.append({
                    "rule": rule_key,
                    "title": rule["title"],
                    "icon": rule["icon"],
                    "priority": rule["priority"],
                    "feature": feature_name,
                    "feature_value": factor["value"],
                    "impact": impact,
                    "shap_importance": factor["abs_importance"],
                    "recommendations": rule["recommendations"],
                    "plant_context": rule["plant_context"],
                    "industry_mapping": station_label,
                })

                if len(recommendations) >= top_n:
                    break

        if len(recommendations) >= top_n:
            break

    # Sort by priority (critical > high > medium > low)
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

    logger.info(
        "Generated %d recommendations based on %d risk factors.",
        len(recommendations), len(top_factors)
    )

    return recommendations


def format_recommendations_text(
    recommendations: List[Dict[str, Any]],
) -> str:
    """Format recommendations as plain text for display.

    Args:
        recommendations: List of recommendation dictionaries.

    Returns:
        Formatted text string.
    """
    if not recommendations:
        return (
            "✅ No significant risk factors detected. "
            "Machine is operating within normal parameters.\n"
            "Continue routine monitoring as per maintenance schedule."
        )

    lines = ["⚠ AI-GENERATED DECISION-SUPPORT RECOMMENDATIONS", ""]

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{rec['icon']} {i}. {rec['title']} "
                     f"[Priority: {rec['priority'].upper()}]")
        lines.append(f"   Triggered by: {rec['feature']} → {rec['impact']}")
        for action in rec["recommendations"][:2]:
            lines.append(f"   • {action}")
        lines.append(f"   Plant context: {rec['plant_context']}")
        lines.append("")

    lines.append(
        "DISCLAIMER: These are AI-generated decision-support recommendations, "
        "not certified engineering procedures. Always verify with qualified "
        "maintenance personnel before taking action."
    )

    return "\n".join(lines)


def get_recommendation_summary(
    recommendations: List[Dict[str, Any]],
) -> List[str]:
    """Get a concise summary of recommendations.

    Args:
        recommendations: List of recommendation dictionaries.

    Returns:
        List of short recommendation strings.
    """
    if not recommendations:
        return ["No immediate action required. Continue routine monitoring."]

    summary = []
    for rec in recommendations[:3]:
        actions = rec.get("recommendations", [])
        primary_action = actions[0] if isinstance(actions, list) and actions else "Review conditions."
        icon = rec.get("icon", "⚠️")
        summary.append(f"{icon} {primary_action}")

    return summary
