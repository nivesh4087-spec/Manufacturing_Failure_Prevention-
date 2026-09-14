"""
Risk Assessment
===============
Single-asset diagnosis: enter the current sensor readings, get a calibrated
failure probability, the reasoning behind it, and what to do about it.

Inputs are bound to ``st.session_state`` keys so a loaded scenario survives the
rerun that follows any later interaction. Binding them by value instead — the
obvious approach — silently reverts the form the moment the user touches
anything else, because a Streamlit button reads ``True`` for exactly one rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st

from app.components.styles import (
    TOKENS,
    plotly_layout,
    render_footer,
    render_kv_rows,
    render_masthead,
    render_notice,
    render_pill,
    render_risk_pill,
    render_stat_tile,
)

#: session_state key -> the config field each input reads its default from.
INPUT_KEYS = {
    "rp_air_temp": "air_temp_k",
    "rp_process_temp": "process_temp_k",
    "rp_rpm": "rotational_speed_rpm",
    "rp_torque": "torque_nm",
    "rp_tool_wear": "tool_wear_min",
    "rp_type": "type",
}

TYPE_LABELS = {0: "Economy (L)", 1: "Standard (M)", 2: "Premium (H)"}
LABEL_TO_TYPE = {v: k for k, v in TYPE_LABELS.items()}

#: Priority -> the status token used for its pill and left rail.
PRIORITY_STATUS = {
    "critical": "critical",
    "high": "serious",
    "medium": "warning",
    "low": "good",
}


def _seed_inputs(scenario_values: Dict[str, Any]) -> None:
    """Write a scenario's values into session state.

    Called before the widgets are constructed on the next rerun, so the widgets
    pick the values up as their initial state.
    """
    for key, field in INPUT_KEYS.items():
        if field in scenario_values:
            st.session_state[key] = scenario_values[field]


def _ensure_defaults(config: Dict[str, Any]) -> None:
    """Seed the form once per session from the ``normal`` demo scenario."""
    if "rp_initialised" not in st.session_state:
        _seed_inputs(config["demo_scenarios"]["normal"]["values"])
        st.session_state.rp_initialised = True


def render_page(project_root: Path, load_artifacts_fn, load_dataset_fn) -> None:
    """Render the Risk Assessment page."""
    try:
        artifacts, config = load_artifacts_fn()
    except Exception as exc:
        st.markdown(render_notice("Models unavailable", str(exc), "critical"), unsafe_allow_html=True)
        return

    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]
    best_name = artifacts.get("best_model_name", "model")

    _ensure_defaults(config)

    st.markdown(
        render_masthead(
            "Risk Assessment",
            "Score one asset against its current sensor readings and see what drives the result.",
            f"{best_name}<br>calibrated probabilities",
        ),
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------- scenarios
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Starting point</div>'
        '<div class="panel-note">Load a representative operating state, then adjust</div></div>',
        unsafe_allow_html=True,
    )

    scenarios = config["demo_scenarios"]
    scenario_cols = st.columns(len(scenarios))
    for col, (key, scenario) in zip(scenario_cols, scenarios.items()):
        with col:
            if st.button(
                scenario["name"],
                key=f"scenario_{key}",
                use_container_width=True,
                help=scenario["description"],
            ):
                _seed_inputs(scenario["values"])
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------- inputs
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Sensor readings</div>'
        '<div class="panel-note">Line 1 — stamping and milling</div></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.number_input(
            "Air temperature (K)", min_value=290.0, max_value=310.0, step=0.1,
            key="rp_air_temp", help="Ambient temperature in the press hall.",
        )
        st.number_input(
            "Process temperature (K)", min_value=300.0, max_value=320.0, step=0.1,
            key="rp_process_temp", help="Spindle and drive housing temperature.",
        )

    with col2:
        st.number_input(
            "Rotational speed (rpm)", min_value=1000, max_value=3000, step=10,
            key="rp_rpm", help="Cutting spindle speed.",
        )
        st.number_input(
            "Torque (Nm)", min_value=3.0, max_value=80.0, step=0.5,
            key="rp_torque", help="Press drive torque during the cut.",
        )

    with col3:
        st.number_input(
            "Tool wear (min)", min_value=0, max_value=260, step=1,
            key="rp_tool_wear", help="Minutes accumulated on the current punch and die.",
        )
        st.selectbox(
            "Board grade",
            list(TYPE_LABELS.values()),
            index=int(st.session_state.get("rp_type", 1)),
            key="rp_type_label",
        )

    # The thermal margin is the single most diagnostic derived value, so show it
    # live rather than making the user wait for a prediction to see it.
    margin = st.session_state.rp_process_temp - st.session_state.rp_air_temp
    margin_status = "good" if margin >= 9.5 else ("warning" if margin >= 8.0 else "critical")
    st.markdown(
        f'<div style="margin-top:10px; font-size:0.78rem; color:var(--ink-muted);">'
        f"Thermal margin {margin:.1f} K &nbsp; "
        f'{render_pill("healthy" if margin_status == "good" else "narrow", margin_status)}'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    predict = st.button("Assess failure risk", type="primary", use_container_width=True)
    if not predict:
        st.caption("Load a scenario or enter readings, then run the assessment.")
        return

    # ------------------------------------------------------------- prediction
    input_data = {
        "air_temp_k": float(st.session_state.rp_air_temp),
        "process_temp_k": float(st.session_state.rp_process_temp),
        "rotational_speed_rpm": int(st.session_state.rp_rpm),
        "torque_nm": float(st.session_state.rp_torque),
        "tool_wear_min": int(st.session_state.rp_tool_wear),
        "type": LABEL_TO_TYPE[st.session_state.rp_type_label],
    }
    st.session_state.rp_type = input_data["type"]

    from src.preprocessing.pipeline import preprocess_single_input
    from src.risk.scoring import (
        compute_risk_score,
        generate_early_warning,
        generate_risk_assessment,
        get_risk_category,
    )

    try:
        X_input = preprocess_single_input(
            input_data, config, artifacts["scaler"], artifacts["feature_stats"],
            artifacts["feature_names"], artifacts["numerical_cols"],
        )
        failure_prob = float(model.predict_proba(X_input)[:, 1][0])
    except Exception as exc:
        st.markdown(
            render_notice(f"{type(exc).__name__}", str(exc), "critical"), unsafe_allow_html=True
        )
        return

    risk_score = compute_risk_score(failure_prob, config)
    risk_cat = get_risk_category(risk_score, config)

    shap_explanation = _explain(artifacts, config, X_input)
    risk_assessment = generate_risk_assessment(failure_prob, shap_explanation, config)
    early_warning = generate_early_warning(risk_assessment, config)

    from src.recommendations.engine import generate_recommendations, get_recommendation_summary

    recommendations = generate_recommendations(shap_explanation, config, top_n=5)
    st.session_state.prediction_history.add_prediction(
        input_data, risk_assessment, get_recommendation_summary(recommendations)
    )

    # ----------------------------------------------------------------- result
    if early_warning:
        st.markdown(
            render_notice(
                f"{early_warning['type']} — {early_warning['severity']}",
                early_warning["message"],
                "critical" if risk_score > config["risk"]["thresholds"]["high"] else "serious",
            ),
            unsafe_allow_html=True,
        )

    band_status = {"LOW RISK": "good", "MODERATE RISK": "warning",
                   "HIGH RISK": "serious", "CRITICAL RISK": "critical"}.get(
        risk_cat["label"], "neutral")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        failing = failure_prob >= 0.5
        st.markdown(
            render_stat_tile(
                "Call", "Failure" if failing else "No failure",
                "At the 50% decision threshold", "critical" if failing else "good",
            ),
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            render_stat_tile(
                "Failure probability", f"{failure_prob * 100:.1f}%",
                "Calibrated estimate", band_status,
            ),
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            render_stat_tile(
                "Risk score", f"{risk_score:.0f}", render_risk_pill(risk_cat["label"]), band_status,
            ),
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            render_stat_tile(
                "Action window",
                _action_window(risk_score, config),
                "Based on the risk band", band_status,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    gauge_col, shap_col = st.columns([2, 3], gap="medium")

    with gauge_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">Risk score</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _risk_gauge(risk_score, TOKENS[band_status], config), use_container_width=True
        )
        st.markdown(
            f'<div style="font-size:0.8rem; color:var(--ink-secondary); line-height:1.5;">'
            f'{risk_assessment["assessment_summary"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            render_kv_rows(
                {
                    "Thermal margin": f"{margin:.1f} K",
                    "Torque per 1000 rpm": f"{input_data['torque_nm'] / input_data['rotational_speed_rpm'] * 1000:.2f} Nm",
                    "Tool wear": f"{input_data['tool_wear_min']} min",
                    "Board grade": TYPE_LABELS[input_data["type"]],
                }
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with shap_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">What drove this result</div>'
            '<div class="panel-note">SHAP contribution per feature</div></div>',
            unsafe_allow_html=True,
        )

        factors = shap_explanation.get("top_factors", [])[:8]
        if factors:
            st.plotly_chart(_shap_chart(factors, config), use_container_width=True)
            st.caption(
                "Bars to the right pushed the prediction toward failure; bars to "
                "the left pushed it away. Length is how hard."
            )
        elif shap_explanation.get("error"):
            st.markdown(
                render_notice(
                    "Explanation unavailable",
                    f"SHAP could not run for this prediction: "
                    f"<code>{shap_explanation['error']}</code>",
                    "warning",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No feature moved this prediction materially.")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------- actions
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Recommended actions</div>'
        '<div class="panel-note">Highest priority first</div></div>',
        unsafe_allow_html=True,
    )

    if recommendations:
        for rec in recommendations[:4]:
            status = PRIORITY_STATUS.get(rec["priority"], "neutral")
            steps = "".join(f"<li>{step}</li>" for step in rec["recommendations"][:2])
            st.markdown(
                f'<div class="action action-{rec["priority"]}">'
                f'<div class="action-head">'
                f'<span class="action-title">{rec["title"]}</span>'
                f'{render_pill(rec["priority"], status)}'
                f"</div>"
                f'<div class="action-trigger">Triggered by '
                f'<code>{rec["feature"]}</code> = {rec["feature_value"]:.2f} '
                f"&mdash; {rec['industry_mapping']}</div>"
                f"<ul>{steps}</ul>"
                f'<div class="action-context">{rec["plant_context"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            render_notice(
                "No action required",
                "No feature pushed this asset toward failure. Continue on the "
                "normal maintenance schedule.",
                "good",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="disclaimer">These are model-derived suggestions. Confirm the '
        "physical condition of the equipment with a qualified technician before "
        "acting on any of them.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        render_footer(
            f"Assessed with {best_name}",
            f"{len(st.session_state.prediction_history.get_history_df())} assessments this session",
        ),
        unsafe_allow_html=True,
    )


# ============================================================================
# Helpers
# ============================================================================

def _action_window(risk_score: float, config: Dict[str, Any]) -> str:
    """Translate a risk score into the timeframe an engineer should act in."""
    thresholds = config["risk"]["thresholds"]
    if risk_score <= thresholds["low"]:
        return "Routine"
    if risk_score <= thresholds["moderate"]:
        return "Next service"
    if risk_score <= thresholds["high"]:
        return "This shift"
    return "Immediate"


@st.cache_resource(show_spinner=False)
def _build_explainer(_artifacts: Dict[str, Any], _config: Dict[str, Any], cache_key: str):
    # `cache_key` is never read: it exists so Streamlit re-keys the cache when the
    # underlying model changes, since the artifacts themselves are unhashable.
    """Build and cache a SHAP explainer with a fixed background sample.

    Rebuilding the explainer on every prediction dominated the response time.
    Caching it as a resource keeps the background sample fixed too, which also
    makes successive explanations directly comparable.
    """
    from src.explainability.shap_engine import SHAPEngine

    from src.data.loader import load_dataset
    from src.features.engineer import engineer_features
    from src.preprocessing.pipeline import (
        apply_scaler,
        drop_id_columns,
        drop_leakage_columns,
        encode_type_column,
        rename_columns,
    )

    df = load_dataset(_config)
    df = drop_leakage_columns(df.copy(), _config)
    df = drop_id_columns(df, _config)
    df = rename_columns(df, _config)
    df = encode_type_column(df, _config)
    df.pop("machine_failure", None)
    df, _ = engineer_features(df, _config, fit_stats=_artifacts["feature_stats"])
    df = apply_scaler(df, _artifacts["scaler"], _config, feature_cols=_artifacts["numerical_cols"])

    background = df.sample(
        n=min(_config["shap"]["background_samples"], len(df)), random_state=42
    )
    return SHAPEngine(
        model=_artifacts.get("best_model"),
        X_background=background,
        config=_config,
        model_type="tree",
        feature_names=_artifacts["feature_names"],
    )


def _explain(artifacts, config, X_input) -> Dict[str, Any]:
    """Explain one prediction, degrading to an empty explanation on failure."""
    try:
        engine = _build_explainer(
            artifacts, config, cache_key=artifacts.get("best_model_name", "model")
        )
        return engine.explain_single_prediction(X_input, top_n=10)
    except Exception as exc:
        return {"top_factors": [], "shap_values": [], "contributions": [], "error": str(exc)}


def _risk_gauge(risk_score: float, color: str, config: Dict[str, Any]) -> go.Figure:
    """Build the risk gauge, with the band edges taken from config."""
    thresholds = config["risk"]["thresholds"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"font": {"size": 40, "color": color}, "suffix": ""},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": TOKENS["border_strong"],
                    "tickfont": {"size": 10, "color": TOKENS["ink_muted"]},
                },
                "bar": {"color": color, "thickness": 0.7},
                "bgcolor": TOKENS["surface_raised"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, thresholds["low"]], "color": "rgba(12,163,12,0.14)"},
                    {"range": [thresholds["low"], thresholds["moderate"]], "color": "rgba(250,178,25,0.13)"},
                    {"range": [thresholds["moderate"], thresholds["high"]], "color": "rgba(236,131,90,0.13)"},
                    {"range": [thresholds["high"], 100], "color": "rgba(208,59,59,0.15)"},
                ],
                "threshold": {
                    "line": {"color": TOKENS["critical"], "width": 2},
                    "thickness": 0.85,
                    "value": config["early_warning"]["threshold"],
                },
            },
        )
    )
    fig.update_layout(
        **plotly_layout(
            height=220, margin=dict(t=14, b=4, l=24, r=24),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
    )
    return fig


def _shap_chart(factors, config: Dict[str, Any]) -> go.Figure:
    """Build the per-feature SHAP contribution chart.

    Diverging by sign: one warm pole for "toward failure", one cool pole for
    "away from it", with the zero line as the neutral midpoint.
    """
    mapping = config.get("industry_mapping", {})
    labels = [mapping.get(f["feature"], f["feature"]) for f in factors]
    values = [f["shap_value"] for f in factors]
    colors = [TOKENS["critical"] if v > 0 else TOKENS["good"] for v in values]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            marker_line=dict(width=2, color=TOKENS["surface"]),
            customdata=[[f["feature"], f["value"]] for f in factors],
            hovertemplate="%{customdata[0]} = %{customdata[1]:.2f}<br>SHAP %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=TOKENS["border_strong"])
    fig.update_layout(
        **plotly_layout(
            height=44 + 34 * len(factors),
            x_title="Contribution to failure risk",
            margin=dict(t=10, b=42, l=8, r=26),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(size=11, color=TOKENS["ink_secondary"]),
                showgrid=False,
                showline=False,
                automargin=True,
            ),
        )
    )
    return fig
