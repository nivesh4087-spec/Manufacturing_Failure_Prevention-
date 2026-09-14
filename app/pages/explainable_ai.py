"""
Model Diagnostics
=================
What the model has learned, and why it reaches a given conclusion.

Global view: which sensor readings move predictions across the whole fleet.
Local view: the same breakdown for one chosen asset.
Station view: what each feature corresponds to on the shop floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data_access import active_dataset
from app.components.styles import (
    panel,
    SERIES,
    TOKENS,
    plotly_layout,
    render_footer,
    render_masthead,
    render_notice,
    render_risk_pill,
    render_stat_tile,
)


def render_page(project_root: Path, load_artifacts_fn, load_dataset_fn, load_results_fn) -> None:
    """Render the Model Diagnostics page."""
    try:
        artifacts, config = load_artifacts_fn()
    except Exception as exc:
        st.markdown(render_notice("Models unavailable", str(exc), "critical"), unsafe_allow_html=True)
        return

    best_name = artifacts.get("best_model_name", "model")

    st.markdown(
        render_masthead(
            "Model Diagnostics",
            "Feature attribution from SHAP — what the model weighs, and how it "
            "reached a particular answer.",
            f"{best_name}<br>Shapley additive explanations",
        ),
        unsafe_allow_html=True,
    )

    tab_global, tab_local, tab_stations = st.tabs(
        ["Across the fleet", "One asset", "Station mapping"]
    )

    with tab_global:
        _render_global(artifacts, config, load_results_fn)

    with tab_local:
        _render_local(artifacts, config, load_dataset_fn)

    with tab_stations:
        _render_station_mapping(config)

    st.markdown(
        render_footer("Explanations computed with SHAP", f"Model: {best_name}"),
        unsafe_allow_html=True,
    )


# ============================================================================
# Global
# ============================================================================

def _render_global(artifacts: Dict[str, Any], config: Dict[str, Any], load_results_fn) -> None:
    """Render fleet-wide feature importance."""
    shap_data = load_results_fn("shap_analysis.json")

    if not shap_data or "feature_importance" not in shap_data:
        st.markdown(
            render_notice(
                "Global analysis not computed yet",
                "Fleet-wide SHAP importance is written by the full training run. "
                "Run <code>python scripts/train_pipeline.py --mode fast</code> to "
                "generate it. The single-asset view on the next tab works without it.",
                "warning",
            ),
            unsafe_allow_html=True,
        )
        return

    imp_df = pd.DataFrame(shap_data["feature_importance"])
    mapping = config.get("industry_mapping", {})

    top = imp_df.head(12).copy()
    top["label"] = top["feature"].map(lambda f: mapping.get(f, f))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            render_stat_tile("Features analysed", str(len(imp_df)), "Raw plus engineered", "neutral"),
            unsafe_allow_html=True,
        )
    with c2:
        leader = imp_df.iloc[0]
        st.markdown(
            render_stat_tile(
                "Strongest driver",
                mapping.get(leader["feature"], leader["feature"]).split(",")[0],
                f"{leader['contribution_pct']:.1f}% of total attribution",
                "accent",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        top3 = float(imp_df.head(3)["contribution_pct"].sum())
        st.markdown(
            render_stat_tile(
                "Top three share", f"{top3:.0f}%", "Concentration of model attention", "neutral"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with panel('Mean absolute SHAP value', 'Higher means the feature moves predictions more'):

        fig = go.Figure(
            go.Bar(
                y=top["label"],
                x=top["mean_abs_shap"],
                orientation="h",
                marker_color=SERIES[0],
                marker_line_width=0,
                customdata=top[["feature", "contribution_pct"]].values,
                hovertemplate="%{customdata[0]}<br>Mean |SHAP| %{x:.4f}<br>"
                "%{customdata[1]:.1f}% of total<extra></extra>",
            )
        )
        fig.update_layout(
            **plotly_layout(
                height=52 + 30 * len(top),
                x_title="Mean |SHAP value|",
                margin=dict(t=8, b=42, l=8, r=20),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    showgrid=False,
                    showline=False,
                    automargin=True,
                ),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full ranking as a table"):
            table = imp_df[["rank", "feature", "mean_abs_shap", "contribution_pct"]].copy()
            table["station"] = table["feature"].map(lambda f: mapping.get(f, "—"))
            table.columns = ["Rank", "Feature", "Mean |SHAP|", "Share %", "Station"]
            st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="disclaimer">SHAP distributes a prediction among its inputs '
            "using Shapley values from cooperative game theory, so the contributions "
            "sum to the difference between this prediction and the average one. "
            "Importance here is influence on the model, which is not the same as "
            "physical causation.</div>",
            unsafe_allow_html=True,
        )



# ============================================================================
# Local
# ============================================================================

@st.cache_data(show_spinner="Preparing samples...")
def _prepare_samples(_df: pd.DataFrame, _artifacts: Dict[str, Any], _config: Dict[str, Any], cache_key: str):
    """Preprocess the dataset once for the sample explorer.

    Returns the scaled feature frame alongside the target, so the selector can
    offer real failing and passing assets rather than synthetic ones.
    """
    from src.features.engineer import engineer_features
    from src.preprocessing.pipeline import (
        apply_scaler,
        drop_id_columns,
        drop_leakage_columns,
        encode_type_column,
        rename_columns,
    )

    proc = drop_leakage_columns(_df.copy(), _config)
    proc = drop_id_columns(proc, _config)
    proc = rename_columns(proc, _config)
    proc = encode_type_column(proc, _config)
    target = proc.pop("machine_failure")
    proc, _ = engineer_features(proc, _config, fit_stats=_artifacts["feature_stats"])
    scaled = apply_scaler(proc, _artifacts["scaler"], _config, feature_cols=_artifacts["numerical_cols"])
    return scaled, target


def _render_local(artifacts: Dict[str, Any], config: Dict[str, Any], load_dataset_fn) -> None:
    """Render the single-asset explanation."""
    from app.pages.risk_predictor import _build_explainer

    df, _ = active_dataset(load_dataset_fn)

    try:
        scaled, target = _prepare_samples(
            df, artifacts, config, cache_key=artifacts.get("best_model_name", "m")
        )
    except Exception as exc:
        st.markdown(
            render_notice("Samples unavailable", f"{type(exc).__name__}: {exc}", "critical"),
            unsafe_allow_html=True,
        )
        return

    failing = target[target == 1].index.tolist()
    passing = target[target == 0].index.tolist()

    choice_col, idx_col = st.columns([1, 2])
    with choice_col:
        pool_name = st.radio(
            "Sample pool",
            ["Recorded failures", "Normal operation"],
            horizontal=False,
            key="xai_pool",
        )
    pool = failing if pool_name == "Recorded failures" else passing

    if not pool:
        st.markdown(
            render_notice("No samples", f"The dataset contains no {pool_name.lower()}.", "warning"),
            unsafe_allow_html=True,
        )
        return

    with idx_col:
        position = st.slider(
            "Sample", 0, max(len(pool) - 1, 0), 0, key="xai_position",
            help="Step through the assets in the selected pool.",
        )
    idx = pool[position]
    X_single = scaled.loc[[idx]]

    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]
    prob = float(model.predict_proba(X_single)[:, 1][0])

    from src.risk.scoring import compute_risk_score, get_risk_category

    risk_score = compute_risk_score(prob, config)
    risk_cat = get_risk_category(risk_score, config)
    actual = "Failed" if target.loc[idx] == 1 else "Ran normally"
    predicted_fail = prob >= 0.5
    agrees = predicted_fail == (target.loc[idx] == 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_stat_tile("Asset row", f"#{idx}", pool_name, "neutral"), unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            render_stat_tile(
                "Failure probability", f"{prob * 100:.1f}%", render_risk_pill(risk_cat["label"]),
                "critical" if predicted_fail else "good",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_stat_tile("Risk score", f"{risk_score:.0f}", "Out of 100", "neutral"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_stat_tile(
                "Ground truth", actual,
                "Model agrees" if agrees else "Model disagrees",
                "good" if agrees else "warning",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    try:
        engine = _build_explainer(artifacts, config, cache_key=artifacts.get("best_model_name", "m"))
        explanation = engine.explain_single_prediction(X_single, top_n=10)
    except Exception as exc:
        st.markdown(
            render_notice("Explanation failed", f"{type(exc).__name__}: {exc}", "warning"),
            unsafe_allow_html=True,
        )
        return

    factors = explanation.get("top_factors", [])
    if not factors:
        st.caption("No feature moved this prediction materially.")
        return

    mapping = config.get("industry_mapping", {})
    with panel('Contribution breakdown', 'Right pushes toward failure, left pushes away'):

        labels = [mapping.get(f["feature"], f["feature"]) for f in factors]
        values = [f["shap_value"] for f in factors]
        fig = go.Figure(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=[TOKENS["critical"] if v > 0 else TOKENS["good"] for v in values],
                marker_line=dict(width=2, color=TOKENS["surface"]),
                customdata=[[f["feature"], f["value"]] for f in factors],
                hovertemplate="%{customdata[0]} = %{customdata[1]:.3f}<br>SHAP %{x:+.4f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=TOKENS["border_strong"])
        fig.update_layout(
            **plotly_layout(
                height=52 + 32 * len(factors),
                x_title="SHAP contribution",
                margin=dict(t=8, b=42, l=8, r=24),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    showgrid=False,
                    showline=False,
                    automargin=True,
                ),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="panel-title" style="margin-top:8px;">In plain terms</div>',
            unsafe_allow_html=True,
        )
        for factor in factors[:5]:
            direction = "raised" if factor["impact"] == "increases risk" else "lowered"
            station = mapping.get(factor["feature"], factor["feature"])
            st.markdown(
                f'<div style="font-size:0.82rem; color:var(--ink-secondary); padding:4px 0; '
                f'border-bottom:1px solid var(--border);">'
                f'<strong style="color:var(--ink);">{station}</strong> at '
                f'<code>{factor["value"]:.2f}</code> {direction} the failure estimate '
                f'<span style="font-variant-numeric:tabular-nums;">'
                f'({factor["shap_value"]:+.4f})</span></div>',
                unsafe_allow_html=True,
            )



# ============================================================================
# Station mapping
# ============================================================================

def _render_station_mapping(config: Dict[str, Any]) -> None:
    """Render what each model feature corresponds to on the plant floor."""
    st.markdown(
        render_notice(
            "Reading this page against the plant",
            "Line 1 is the CNC stamping and milling machinery the model monitors. "
            "Line 2 is the tile inspection line downstream. Tool condition on "
            "Line 1 is what drives the defect rate on Line 2, which is why a "
            "machinery model is worth running at all.",
            "accent",
        ),
        unsafe_allow_html=True,
    )

    mapping = config.get("industry_mapping", {})
    if not mapping:
        st.caption("No station mapping configured.")
        return

    engineered = set(config.get("features", {}).get("engineered", {}))
    rows = [
        {
            "Feature": feature,
            "Station or reading": label,
            "Source": "Derived" if feature in engineered or feature.startswith("is_")
            or feature.endswith(("_severity", "_indicator")) else "Sensor",
        }
        for feature, label in mapping.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
#### How each reading turns into a failure mode

| Reading drifts | Physical mechanism | Failure mode it precedes |
|:--|:--|:--|
| Thermal margin narrows | Heat is not leaving the spindle housing | Heat dissipation failure |
| Torque rises at constant speed | Board resists the cut; feed or density off spec | Overstrain failure |
| Speed falls below setpoint | Supply instability or mechanical drag | Power failure |
| Tool wear accumulates | Punch and die edge rounds off | Tool wear failure, then Line 2 edge chipping |
| Torque high while speed low | The cut is stalling | Overstrain, with tool breakage risk |
        """
    )
