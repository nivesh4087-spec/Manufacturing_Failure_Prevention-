"""
Model & Cost Analysis
=====================
Which model to deploy, and what that choice is worth in currency.

Curves are recomputed from the held-out test split rather than read from a saved
image, so they stay honest after a retrain. The split itself is cached — it used
to be rebuilt from scratch on every widget interaction, which made the cost
sliders feel broken.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from app.components.data_access import active_dataset, get_test_split
from app.components.styles import (
    SEQUENTIAL,
    SERIES,
    TOKENS,
    plotly_layout,
    render_footer,
    render_masthead,
    render_notice,
    render_stat_tile,
)


def render_page(project_root: Path, load_artifacts_fn, load_results_fn, load_dataset_fn) -> None:
    """Render the Model & Cost Analysis page."""
    try:
        artifacts, config = load_artifacts_fn()
    except Exception as exc:
        st.markdown(render_notice("Models unavailable", str(exc), "critical"), unsafe_allow_html=True)
        return

    test_results: List[Dict[str, Any]] = artifacts.get("test_results", [])
    best_name = artifacts.get("best_model_name", "model")
    all_models = artifacts.get("all_models", {})
    calibrated_model = artifacts.get("best_model_calibrated")

    st.markdown(
        render_masthead(
            "Model & Cost Analysis",
            "How the candidate models compare on the held-out test split, and what "
            "each one is worth once failures and false alarms are priced.",
            f"{len(all_models)} candidates<br>selected on "
            f"{config['selection']['primary_metric'].upper()}",
        ),
        unsafe_allow_html=True,
    )

    df, _ = active_dataset(load_dataset_fn)
    try:
        X_test, y_test = get_test_split(df, config, cache_key=str(len(df)))
    except Exception as exc:
        st.markdown(
            render_notice("Evaluation split unavailable", f"{type(exc).__name__}: {exc}", "critical"),
            unsafe_allow_html=True,
        )
        return

    tabs = st.tabs(
        ["Leaderboard", "ROC & PR", "Confusion", "Calibration", "Feature impact", "Cost"]
    )

    with tabs[0]:
        _render_leaderboard(test_results, best_name, config)
    with tabs[1]:
        _render_curves(all_models, X_test, y_test)
    with tabs[2]:
        _render_confusion(test_results, best_name)
    with tabs[3]:
        _render_calibration(all_models, calibrated_model, best_name, X_test, y_test, load_results_fn)
    with tabs[4]:
        _render_ablation(load_results_fn)
    with tabs[5]:
        _render_cost(all_models, calibrated_model, best_name, X_test, y_test, config)

    st.markdown(
        render_footer(
            f"Evaluated on {len(y_test):,} held-out assets",
            f"Deployed model: {best_name}",
        ),
        unsafe_allow_html=True,
    )


# ============================================================================
# Leaderboard
# ============================================================================

def _render_leaderboard(test_results, best_name: str, config: Dict[str, Any]) -> None:
    """Render headline metrics and the full comparison table."""
    if not test_results:
        st.markdown(
            render_notice("No results", "Re-run the training pipeline.", "warning"),
            unsafe_allow_html=True,
        )
        return

    best = next((r for r in test_results if r["model"] == best_name), test_results[0])

    cols = st.columns(5)
    tiles = [
        ("Selected model", best_name, f"Chosen on {config['selection']['primary_metric'].upper()}", "accent"),
        ("Precision", f"{best['precision']:.3f}", "Of the flagged, how many really fail", "neutral"),
        ("Recall", f"{best['recall']:.3f}", "Of the failures, how many were caught", "neutral"),
        ("F1", f"{best['f1']:.3f}", "Balance of the two above", "neutral"),
        ("PR-AUC", f"{best['pr_auc']:.3f}", "Robust to the class imbalance", "neutral"),
    ]
    for col, (label, value, note, status) in zip(cols, tiles):
        with col:
            st.markdown(render_stat_tile(label, value, note, status), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">All candidates</div>'
        '<div class="panel-note">Test split, best value per column highlighted</div></div>',
        unsafe_allow_html=True,
    )

    table = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
                "PR-AUC": r["pr_auc"],
                "ROC-AUC": r["roc_auc"],
                "Brier": r["brier_score"],
            }
            for r in test_results
        ]
    )

    st.dataframe(
        table.style.highlight_max(
            subset=["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"], color="#16331f"
        )
        .highlight_min(subset=["Brier"], color="#16331f")
        .format({c: "{:.4f}" for c in table.columns if c != "Model"}),
        use_container_width=True,
        hide_index=True,
    )

    # Grouped bars: one slot per model, assigned in fixed order.
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
               ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]
    fig = go.Figure()
    for i, r in enumerate(test_results):
        fig.add_trace(
            go.Bar(
                name=r["model"],
                x=[label for _, label in metrics],
                y=[r[key] for key, _ in metrics],
                marker_color=SERIES[i % len(SERIES)],
                marker_line=dict(width=2, color=TOKENS["surface"]),
                hovertemplate=f"{r['model']}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **plotly_layout(
            height=320,
            show_legend=True,
            y_title="Score",
            barmode="group",
            bargap=0.28,
            yaxis=dict(
                range=[0, 1.05],
                gridcolor=TOKENS["grid"],
                tickfont=dict(size=11, color=TOKENS["ink_muted"]),
                title_font=dict(size=11, color=TOKENS["ink_muted"]),
            ),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="disclaimer">Accuracy is deliberately absent. With roughly '
        "97 percent of assets running normally, a model that predicts "
        '"no failure" every time scores 97 percent accurate and catches nothing. '
        "F1 and PR-AUC both stay honest under that imbalance.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Curves
# ============================================================================

def _render_curves(all_models, X_test, y_test) -> None:
    """Render ROC and precision-recall curves side by side."""
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">ROC</div>'
            '<div class="panel-note">True positives against false positives</div></div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color=TOKENS["border_strong"], width=1),
                name="Chance", hoverinfo="skip", showlegend=False,
            )
        )
        for i, (name, model) in enumerate(all_models.items()):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} ({auc(fpr, tpr):.3f})",
                    line=dict(width=2, color=SERIES[i % len(SERIES)]),
                    hovertemplate=f"{name}<br>FPR %{{x:.3f}} / TPR %{{y:.3f}}<extra></extra>",
                )
            )
        fig.update_layout(
            **plotly_layout(
                height=400, show_legend=True,
                x_title="False positive rate", y_title="True positive rate",
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">Precision-recall</div>'
            '<div class="panel-note">The honest view under imbalance</div></div>',
            unsafe_allow_html=True,
        )
        baseline = float(y_test.mean())
        fig = go.Figure()
        fig.add_hline(
            y=baseline, line_dash="dash", line_color=TOKENS["border_strong"], line_width=1,
            annotation_text=f"Chance ({baseline:.3f})",
            annotation_font=dict(size=10, color=TOKENS["ink_muted"]),
        )
        for i, (name, model) in enumerate(all_models.items()):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                continue
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision, mode="lines",
                    name=f"{name} ({average_precision_score(y_test, y_prob):.3f})",
                    line=dict(width=2, color=SERIES[i % len(SERIES)]),
                    hovertemplate=f"{name}<br>Recall %{{x:.3f}} / Precision %{{y:.3f}}<extra></extra>",
                )
            )
        fig.update_layout(
            **plotly_layout(height=400, show_legend=True, x_title="Recall", y_title="Precision")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer">ROC looks flattering on an imbalanced problem '
        "because the false positive rate is divided by a very large negative "
        "class. The precision-recall curve divides by the flagged set instead, "
        "so it exposes the cost of over-flagging.</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# Confusion
# ============================================================================

def _render_confusion(test_results, best_name: str) -> None:
    """Render the confusion matrix and per-class report for one model."""
    if not test_results:
        st.caption("No results to show.")
        return

    names = [r["model"] for r in test_results]
    chosen = st.selectbox(
        "Model", names, index=names.index(best_name) if best_name in names else 0, key="cm_model"
    )
    result = next(r for r in test_results if r["model"] == chosen)
    cm = result["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_stat_tile("Caught", f"{tp:,}", "Failures correctly flagged", "good"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_stat_tile("Missed", f"{fn:,}", "Failures that slipped through", "critical"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_stat_tile("False alarms", f"{fp:,}", "Healthy assets flagged", "warning"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_stat_tile("Correctly cleared", f"{tn:,}", "Healthy assets passed", "neutral"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="medium")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">Confusion matrix</div></div>',
            unsafe_allow_html=True,
        )
        labels = ["Ran normally", "Failed"]
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                reversescale=True,
                text=[[f"{v:,}" for v in row] for row in cm],
                texttemplate="%{text}",
                textfont={"size": 17, "color": TOKENS["ink"]},
                showscale=False,
                xgap=3,
                ygap=3,
                hovertemplate="Actual %{y}, predicted %{x}: %{z:,}<extra></extra>",
            )
        )
        fig.update_layout(
            **plotly_layout(
                height=300,
                x_title="Predicted",
                y_title="Actual",
                margin=dict(t=8, b=48, l=100, r=8),
                xaxis=dict(showgrid=False, showline=False,
                           tickfont=dict(size=11, color=TOKENS["ink_secondary"])),
                yaxis=dict(showgrid=False, showline=False, autorange="reversed",
                           tickfont=dict(size=11, color=TOKENS["ink_secondary"])),
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">Per-class report</div></div>',
            unsafe_allow_html=True,
        )
        report = result.get("classification_report", {})
        if report:
            st.dataframe(
                pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True
            )
        else:
            st.caption("No classification report saved for this model.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Calibration
# ============================================================================

def _render_calibration(all_models, calibrated_model, best_name, X_test, y_test, load_results_fn) -> None:
    """Render the reliability diagram and Brier scores."""
    cal = load_results_fn("calibration_results.json")

    if cal:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                render_stat_tile("Brier, raw", f"{cal['brier_uncalibrated']:.4f}",
                                 "Before calibration", "neutral"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_stat_tile("Brier, calibrated", f"{cal['brier_calibrated']:.4f}",
                                 f"{cal.get('method', 'isotonic').title()} regression", "good"),
                unsafe_allow_html=True,
            )
        with c3:
            improvement = cal["improvement"]
            st.markdown(
                render_stat_tile(
                    "Improvement", f"{improvement:+.4f}",
                    "Lower Brier is better", "good" if improvement > 0 else "warning",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Reliability diagram</div>'
        '<div class="panel-note">Closer to the diagonal means the probabilities can be trusted</div></div>',
        unsafe_allow_html=True,
    )

    models = dict(all_models)
    if calibrated_model is not None:
        models[f"{best_name}, calibrated"] = calibrated_model

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color=TOKENS["border_strong"], width=1),
            name="Perfect", hoverinfo="skip", showlegend=False,
        )
    )
    for i, (name, model) in enumerate(models.items()):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        except Exception:
            continue
        fig.add_trace(
            go.Scatter(
                x=prob_pred, y=prob_true, mode="lines+markers", name=name,
                line=dict(width=2, color=SERIES[i % len(SERIES)]),
                marker=dict(size=8, line=dict(width=2, color=TOKENS["surface"])),
                hovertemplate=f"{name}<br>Predicted %{{x:.3f}} / observed %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **plotly_layout(
            height=420, show_legend=True,
            x_title="Mean predicted probability", y_title="Observed failure rate",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="disclaimer">A calibrated model that says 80 percent means '
        "roughly 80 of every 100 such assets really do fail. That property is "
        "what makes the cost arithmetic on the next tab meaningful — an "
        "uncalibrated score can rank assets correctly while still being the "
        "wrong number to multiply a cost by.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Ablation
# ============================================================================

def _render_ablation(load_results_fn) -> None:
    """Render the feature-engineering ablation result."""
    ablation = load_results_fn("ablation_results.json")

    if not ablation or "with_feature_engineering" not in ablation:
        st.markdown(
            render_notice(
                "Ablation not computed",
                "The feature-engineering ablation runs as part of the full training "
                "pipeline. Run <code>python scripts/train_pipeline.py --mode fast</code> "
                "to generate it.",
                "warning",
            ),
            unsafe_allow_html=True,
        )
    else:
        without = ablation["without_feature_engineering"]
        with_fe = ablation["with_feature_engineering"]
        delta = with_fe["f1"] - without["f1"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                render_stat_tile("Raw sensors only", f"{without['f1']:.4f}", "F1 score", "neutral"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_stat_tile("With derived features", f"{with_fe['f1']:.4f}", "F1 score", "accent"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_stat_tile(
                    "Difference", f"{delta:+.4f}",
                    "Attributable to feature engineering",
                    "good" if delta > 0 else "warning",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
#### Why these features exist

| Derived feature | Formula | The physics it captures |
|:--|:--|:--|
| Thermal margin | process temp − air temp | How well heat is leaving the machine |
| Mechanical power | torque × speed × 2π / 60 | Actual work being done at the cut |
| Load per unit speed | torque ÷ speed | Whether the drive is straining |
| Accumulated strain | tool wear × torque | Cumulative stress on the tooling |
| Thermal-speed stress | thermal margin × speed | Heat generated at operating speed |

None of these is new information — each is a combination of readings the model
already has. They help because a tree has to spend many splits to approximate a
ratio or a product, and stating it directly spends none.
        """
    )


# ============================================================================
# Cost
# ============================================================================

def _cost_of(tp: int, fp: int, fn: int, failure_cost: float,
             preventive: float, false_alarm: float, licence: float) -> float:
    """Return the annual operating cost of running a model at one threshold."""
    return tp * preventive + fp * false_alarm + fn * failure_cost + licence


def _render_cost(all_models, calibrated_model, best_name, X_test, y_test, config) -> None:
    """Render the cost comparison and the cost-optimal threshold sweep."""
    business = config.get("business", {})

    st.markdown(
        render_notice(
            "What this tab does",
            "Every confusion-matrix cell is priced, so the model choice and the "
            "alert threshold become a currency decision rather than a metric one. "
            "Change the figures below to your own plant's numbers.",
            "accent",
        ),
        unsafe_allow_html=True,
    )

    # Controls live on the page, not the global sidebar, so they stay next to
    # the numbers they change.
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        downtime_cost = st.number_input(
            "Downtime ($/hr)", min_value=0.0, step=500.0,
            value=float(business.get("downtime_cost_per_hour", 10000.0)), key="cost_downtime",
        )
    with c2:
        downtime_hours = st.number_input(
            "Recovery (hrs)", min_value=0.1, step=0.5,
            value=float(business.get("avg_downtime_hours", 4.0)), key="cost_hours",
        )
    with c3:
        preventive = st.number_input(
            "Planned fix ($)", min_value=0.0, step=100.0,
            value=float(business.get("preventive_action_cost", 1500.0)), key="cost_preventive",
        )
    with c4:
        false_alarm = st.number_input(
            "False alarm ($)", min_value=0.0, step=100.0,
            value=float(business.get("false_alarm_cost", 1500.0)), key="cost_false_alarm",
        )
    with c5:
        licence = st.number_input(
            "Platform ($/mo)", min_value=0.0, step=500.0,
            value=float(business.get("license_cost_monthly", 2500.0)), key="cost_licence",
        )

    failure_cost = downtime_cost * downtime_hours

    # ------------------------------------------------------ model comparison
    rows = []
    for name, model in all_models.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            continue
        tn, fp, fn, tp = confusion_matrix(y_test, (y_prob >= 0.5).astype(int)).ravel()
        reactive = (tp + fn) * failure_cost
        predictive = _cost_of(tp, fp, fn, failure_cost, preventive, false_alarm, licence)
        rows.append(
            {
                "Model": name,
                "Missed failures": int(fn),
                "False alarms": int(fp),
                "Reactive cost": reactive,
                "With the model": predictive,
                "Avoided": reactive - predictive,
                "Return": (reactive - predictive) / predictive * 100 if predictive else 0.0,
            }
        )

    if not rows:
        st.caption("No model could be scored.")
        return

    cost_df = pd.DataFrame(rows).sort_values("Avoided", ascending=False)
    top = cost_df.iloc[0]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            render_stat_tile(
                "Cost of one failure", f"${failure_cost:,.0f}",
                f"{downtime_hours:.1f} hrs at ${downtime_cost:,.0f}/hr", "critical",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_stat_tile(
                "Best avoided cost", f"${top['Avoided']:,.0f}",
                f"{top['Model']}, on this test split", "good",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_stat_tile(
                "Return on the spend", f"{top['Return']:,.0f}%",
                "Avoided cost over total cost", "good",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Cost by model</div>'
        f'<div class="panel-note">Over the {len(y_test):,} assets in the test split</div></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        cost_df.style.format(
            {
                "Reactive cost": "${:,.0f}",
                "With the model": "${:,.0f}",
                "Avoided": "${:,.0f}",
                "Return": "{:,.0f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------- threshold sweep
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Where to set the alert threshold</div>'
        '<div class="panel-note">Total cost against the flagging threshold</div></div>',
        unsafe_allow_html=True,
    )

    model = calibrated_model if calibrated_model is not None else all_models.get(best_name)
    if model is None:
        st.caption("No model available for the sweep.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.02, 0.98, 97)
    costs, missed, alarms = [], [], []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_test, (y_prob >= threshold).astype(int)).ravel()
        costs.append(_cost_of(tp, fp, fn, failure_cost, preventive, false_alarm, licence))
        missed.append(int(fn))
        alarms.append(int(fp))

    best_idx = int(np.argmin(costs))
    best_threshold = float(thresholds[best_idx])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds, y=costs, mode="lines",
            line=dict(width=2, color=SERIES[0]), name="Total cost",
            hovertemplate="Threshold %{x:.2f}<br>Cost $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_threshold], y=[costs[best_idx]], mode="markers",
            marker=dict(size=11, color=TOKENS["good"], line=dict(width=2, color=TOKENS["surface"])),
            name=f"Cheapest at {best_threshold:.2f}",
            hovertemplate=f"Cheapest: threshold {best_threshold:.2f}"
                          f"<br>Cost $%{{y:,.0f}}<extra></extra>",
        )
    )
    fig.add_vline(
        x=0.5, line_dash="dash", line_color=TOKENS["border_strong"], line_width=1,
        annotation_text="Default 0.50",
        annotation_font=dict(size=10, color=TOKENS["ink_muted"]),
    )
    fig.update_layout(
        **plotly_layout(
            height=320, show_legend=True,
            x_title="Flag an asset when failure probability exceeds",
            y_title="Total cost ($)",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    saving_vs_default = _cost_of(
        *_counts_at(y_test, y_prob, 0.5), failure_cost, preventive, false_alarm, licence
    ) - costs[best_idx]

    st.markdown(
        render_notice(
            f"Cheapest threshold is {best_threshold:.2f}, not 0.50",
            f"At {best_threshold:.2f} the model misses {missed[best_idx]} failures and "
            f"raises {alarms[best_idx]} false alarms, for a total of "
            f"${costs[best_idx]:,.0f} — "
            + (
                f"${saving_vs_default:,.0f} less than the default 0.50 threshold."
                if saving_vs_default > 0
                else "which the default 0.50 threshold already matches."
            )
            + " A missed failure costs far more than a needless inspection, so the "
            "arithmetic favours flagging early.",
            "good" if saving_vs_default > 0 else "accent",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _counts_at(y_test, y_prob, threshold: float):
    """Return ``(tp, fp, fn)`` at a given decision threshold."""
    tn, fp, fn, tp = confusion_matrix(y_test, (y_prob >= threshold).astype(int)).ravel()
    return int(tp), int(fp), int(fn)
