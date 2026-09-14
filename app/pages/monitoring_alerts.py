"""
Alerts
======
Every assessment made this session, what was flagged, and how risk moved.

The history lives in ``st.session_state`` and resets when the server restarts,
which is honest about what it is: a working log for one operator's session, not
a persisted audit trail.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.components.styles import (
    panel,
    TOKENS,
    plotly_layout,
    render_footer,
    render_masthead,
    render_notice,
    render_pill,
    render_stat_tile,
)

#: Risk band -> status token, in severity order.
BANDS = [
    ("LOW RISK", "good"),
    ("MODERATE RISK", "warning"),
    ("HIGH RISK", "serious"),
    ("CRITICAL RISK", "critical"),
]


def render_page(project_root: Path) -> None:
    """Render the Alerts page."""
    history = st.session_state.get("prediction_history")
    if history is None:
        from src.risk.scoring import PredictionHistory

        history = st.session_state.prediction_history = PredictionHistory()

    history_df = history.get_history_df()
    alert_count = history.get_alert_count()
    risk_dist = history.get_risk_distribution()

    st.markdown(
        render_masthead(
            "Alerts",
            "Assessments made in this session, with anything above the warning "
            "threshold raised to the top.",
            f"Session log<br>{len(history_df)} assessments",
        ),
        unsafe_allow_html=True,
    )

    if history_df.empty:
        st.markdown(
            render_notice(
                "Nothing logged yet",
                "Run an assessment on the <strong>Risk Assessment</strong> page or "
                "score a file on <strong>Batch Analysis</strong>, and every result "
                "will appear here with the reasoning that produced it.",
                "accent",
            ),
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------------ KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_stat_tile("Assessments", str(len(history_df)), "This session", "neutral"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_stat_tile(
                "Raised alerts", str(alert_count),
                "High and critical bands",
                "serious" if alert_count else "good",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        critical = risk_dist.get("CRITICAL RISK", 0)
        st.markdown(
            render_stat_tile(
                "Critical", str(critical),
                "Immediate attention" if critical else "None",
                "critical" if critical else "good",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        mean_risk = float(history_df["risk_score"].mean())
        st.markdown(
            render_stat_tile(
                "Mean risk score", f"{mean_risk:.1f}",
                f"Peak {history_df['risk_score'].max():.0f}", "neutral",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------- band + feed
    left, right = st.columns([2, 3], gap="medium")

    with left:
        with panel('By band'):
            counts = [risk_dist.get(band, 0) for band, _ in BANDS]
            fig = go.Figure(
                go.Bar(
                    x=counts,
                    y=[band.replace(" RISK", "").title() for band, _ in BANDS],
                    orientation="h",
                    marker_color=[TOKENS[status] for _, status in BANDS],
                    marker_line_width=0,
                    text=[str(c) for c in counts],
                    textposition="auto",
                    textfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    hovertemplate="%{y}: %{x} assessments<extra></extra>",
                )
            )
            fig.update_layout(
                **plotly_layout(
                    height=190,
                    margin=dict(t=8, b=8, l=8, r=16),
                    xaxis=dict(visible=False, range=[0, max(counts or [1]) * 1.02]),
                    yaxis=dict(
                        autorange="reversed", showgrid=False, showline=False,
                        tickfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    ),
                )
            )
            st.plotly_chart(fig, use_container_width=True)


    with right:
        with panel('Alert feed', 'Most recent first'):

            alerts = history_df[
                history_df["risk_category"].isin(["HIGH RISK", "CRITICAL RISK"])
            ].sort_values("id", ascending=False).head(8)

            if alerts.empty:
                st.markdown(
                    render_notice(
                        "No alerts raised",
                        "Every assessment this session landed in the routine or "
                        "moderate band.",
                        "good",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                for _, row in alerts.iterrows():
                    status = "critical" if row["risk_category"] == "CRITICAL RISK" else "serious"
                    st.markdown(
                        f'<div class="action action-{"critical" if status == "critical" else "high"}">'
                        f'<div class="action-head">'
                        f'<span class="action-title">Assessment #{int(row["id"])} &mdash; '
                        f'risk {row["risk_score"]:.0f}</span>'
                        f'{render_pill(row["risk_category"].replace(" RISK", ""), status)}</div>'
                        f'<div class="action-trigger">{row["timestamp"]} &middot; '
                        f'failure probability {row["failure_probability"]:.1f}% &middot; '
                        f'driven by <code>{row["top_factor"]}</code></div>'
                        f'<div class="action-context">{row["recommendation"]}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )


    # ------------------------------------------------------------- timeline
    if len(history_df) > 1:
        with panel('Risk over the session', 'One point per assessment'):

            scores = history_df["risk_score"].tolist()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(scores) + 1)),
                    y=scores,
                    mode="lines+markers",
                    line=dict(color=TOKENS["ink_muted"], width=2),
                    marker=dict(
                        size=9,
                        color=[_band_color(s) for s in scores],
                        line=dict(width=2, color=TOKENS["surface"]),
                    ),
                    hovertemplate="Assessment %{x}: risk %{y:.0f}<extra></extra>",
                    name="Risk score",
                )
            )
            fig.add_hline(
                y=60,
                line_dash="dash",
                line_color=TOKENS["serious"],
                line_width=1,
                annotation_text="Warning threshold",
                annotation_font=dict(size=10, color=TOKENS["serious"]),
                annotation_position="top left",
            )
            fig.update_layout(
                **plotly_layout(
                    height=280,
                    x_title="Assessment number",
                    y_title="Risk score",
                    yaxis=dict(
                        range=[0, 105],
                        gridcolor=TOKENS["grid"],
                        tickfont=dict(size=11, color=TOKENS["ink_muted"]),
                        title_font=dict(size=11, color=TOKENS["ink_muted"]),
                    ),
                )
            )
            st.plotly_chart(fig, use_container_width=True)


    # -------------------------------------------------------------- log table
    with panel('Full log'):

        columns = {
            "id": "#",
            "timestamp": "Time",
            "risk_score": "Risk",
            "failure_probability": "Probability %",
            "risk_category": "Band",
            "prediction": "Call",
            "top_factor": "Top factor",
            "recommendation": "Recommendation",
        }
        present = {k: v for k, v in columns.items() if k in history_df.columns}
        table = history_df[list(present)].rename(columns=present).sort_values("#", ascending=False)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=min(420, 40 + 35 * len(table)),
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=100, format="%.0f"
                ),
            },
        )

        st.download_button(
            "Export session log",
            history_df.to_csv(index=False),
            f"assessment_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )


    st.markdown(
        render_footer(
            "Session log — resets when the server restarts",
            f"{len(history_df)} assessments &middot; {alert_count} alerts",
        ),
        unsafe_allow_html=True,
    )


def _band_color(score: float) -> str:
    """Return the status colour for a risk score."""
    if score <= 30:
        return TOKENS["good"]
    if score <= 60:
        return TOKENS["warning"]
    if score <= 80:
        return TOKENS["serious"]
    return TOKENS["critical"]
