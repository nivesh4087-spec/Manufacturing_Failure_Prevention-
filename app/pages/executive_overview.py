"""
Asset Health
============
Fleet-level view: how many assets are at risk right now, where the risk sits in
the operating envelope, and which assets to look at first.

Every figure on this page is derived from the model's own scoring of the active
dataset. Nothing is hardcoded — swap the dataset or retrain and the numbers move.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.data_access import (
    active_dataset,
    get_fleet_summary,
    get_scored_fleet,
    get_type_mix,
)
from app.components.styles import (
    SERIES,
    TOKENS,
    plotly_layout,
    render_footer,
    render_kv_rows,
    render_masthead,
    render_notice,
    render_stat_tile,
)

#: Risk bands in severity order, with the status token each maps to.
BAND_ORDER = [
    ("LOW RISK", "good"),
    ("MODERATE RISK", "warning"),
    ("HIGH RISK", "serious"),
    ("CRITICAL RISK", "critical"),
]


def _band_colors() -> list[str]:
    """Return the status colours for the risk bands, in severity order."""
    return [TOKENS[status] for _, status in BAND_ORDER]


def render_page(project_root: Path, load_artifacts_fn, load_dataset_fn, load_results_fn) -> None:
    """Render the Asset Health page."""
    try:
        artifacts, config = load_artifacts_fn()
    except Exception as exc:
        st.markdown(
            render_notice("Models unavailable", str(exc), "critical"), unsafe_allow_html=True
        )
        return

    df, source_label = active_dataset(load_dataset_fn)
    target_col = config["data"]["target_column"]
    best_name = artifacts.get("best_model_name", "unknown")

    st.markdown(
        render_masthead(
            "Asset Health",
            "Failure risk across the monitored fleet, scored by the calibrated model.",
            f"{source_label} &middot; {len(df):,} assets<br>"
            f"Scored {datetime.now().strftime('%d %b %Y, %H:%M')}",
        ),
        unsafe_allow_html=True,
    )

    # --- score the fleet ----------------------------------------------------
    try:
        scored = get_scored_fleet(df, artifacts, config, cache_key=f"{best_name}:{len(df)}")
    except Exception as exc:
        st.markdown(
            render_notice(
                "Fleet could not be scored",
                f"{type(exc).__name__}: {exc}. The active dataset may be missing "
                "one of the required sensor columns.",
                "critical",
            ),
            unsafe_allow_html=True,
        )
        return

    summary = get_fleet_summary(scored, config, target_col, cache_key=f"{best_name}:{len(df)}")

    # ------------------------------------------------------------------ KPIs
    cols = st.columns(5, gap="small")

    with cols[0]:
        st.markdown(
            render_stat_tile("Monitored assets", f"{summary['n_assets']:,}", source_label, "neutral"),
            unsafe_allow_html=True,
        )

    with cols[1]:
        critical = summary["n_critical"]
        st.markdown(
            render_stat_tile(
                "Critical risk",
                f"{critical:,}",
                "Stop and inspect this shift" if critical else "None outstanding",
                "critical" if critical else "good",
            ),
            unsafe_allow_html=True,
        )

    with cols[2]:
        elevated = summary["n_elevated"]
        st.markdown(
            render_stat_tile(
                "Above routine",
                f"{elevated:,}",
                f"Risk score over {config['risk']['thresholds']['low']}",
                "warning" if elevated else "good",
            ),
            unsafe_allow_html=True,
        )

    with cols[3]:
        health = summary["health_index"]
        st.markdown(
            render_stat_tile(
                "Fleet health",
                f"{health:.1f}%",
                "Share of assets in the routine band",
                "good" if health >= 90 else ("warning" if health >= 75 else "serious"),
            ),
            unsafe_allow_html=True,
        )

    with cols[4]:
        if "catch_rate" in summary:
            st.markdown(
                render_stat_tile(
                    "Failures caught",
                    f"{summary['catch_rate']:.0f}%",
                    f"{summary['caught_failures']} of {summary['actual_failures']} "
                    "labelled failures flagged",
                    "good" if summary["catch_rate"] >= 80 else "warning",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                render_stat_tile(
                    "Mean risk score",
                    f"{summary['mean_risk']:.1f}",
                    f"95th percentile {summary['p95_risk']:.0f}",
                    "neutral",
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # --------------------------------------------------- risk band + envelope
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">Operating envelope</div>'
            '<div class="panel-note">Each point is one asset</div></div>',
            unsafe_allow_html=True,
        )

        tab_torque, tab_wear, tab_thermal = st.tabs(
            ["Speed against torque", "Tool wear", "Thermal margin"]
        )

        with tab_torque:
            # Two-slot categorical split (flagged / not) keeps the all-pairs
            # colour distance well clear of the floor.
            fig = px.scatter(
                scored,
                x="Rotational speed [rpm]",
                y="Torque [Nm]",
                color="prediction",
                color_discrete_map={"NO FAILURE": SERIES[0], "FAILURE": TOKENS["critical"]},
                opacity=0.55,
                custom_data=["risk_score", "risk_category"],
            )
            fig.update_traces(
                marker=dict(size=5, line=dict(width=0)),
                hovertemplate=(
                    "%{x:,.0f} rpm &middot; %{y:.1f} Nm<br>"
                    "Risk %{customdata[0]:.0f} — %{customdata[1]}<extra></extra>"
                ),
            )
            fig.update_layout(
                **plotly_layout(
                    height=330,
                    show_legend=True,
                    x_title="Rotational speed (rpm)",
                    y_title="Torque (Nm)",
                    legend_title_text="",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Failures concentrate in two corners: high torque at low speed "
                "(overstrain) and low torque at high speed (power loss)."
            )

        with tab_wear:
            fig = px.histogram(
                scored,
                x="Tool wear [min]",
                color="risk_category",
                category_orders={"risk_category": [b for b, _ in BAND_ORDER]},
                color_discrete_sequence=_band_colors(),
                nbins=40,
            )
            fig.update_traces(marker_line_width=0)
            fig.update_layout(
                **plotly_layout(
                    height=330,
                    show_legend=True,
                    x_title="Tool wear (minutes)",
                    y_title="Assets",
                    barmode="stack",
                    bargap=0.06,
                    legend_title_text="",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Risk accumulates with wear rather than appearing suddenly, "
                "which is what makes scheduled replacement effective."
            )

        with tab_thermal:
            envelope = scored.assign(
                thermal_margin=scored["Process temperature [K]"] - scored["Air temperature [K]"]
            )
            fig = px.histogram(
                envelope,
                x="thermal_margin",
                color="risk_category",
                category_orders={"risk_category": [b for b, _ in BAND_ORDER]},
                color_discrete_sequence=_band_colors(),
                nbins=40,
            )
            fig.update_traces(marker_line_width=0)
            fig.update_layout(
                **plotly_layout(
                    height=330,
                    show_legend=True,
                    x_title="Process minus air temperature (K)",
                    y_title="Assets",
                    barmode="stack",
                    bargap=0.06,
                    legend_title_text="",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "A narrow margin means heat is not leaving the process — the "
                "signature that precedes a heat dissipation failure."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="panel-title">Risk distribution</div></div>',
            unsafe_allow_html=True,
        )

        counts = [summary["band_counts"].get(band, 0) for band, _ in BAND_ORDER]
        labels = [band.replace(" RISK", "").title() for band, _ in BAND_ORDER]

        fig = go.Figure(
            go.Bar(
                x=counts,
                y=labels,
                orientation="h",
                marker_color=_band_colors(),
                marker_line_width=0,
                text=[f"{c:,}" for c in counts],
                textposition="outside",
                textfont=dict(color=TOKENS["ink_secondary"], size=11),
                hovertemplate="%{y}: %{x:,} assets<extra></extra>",
            )
        )
        fig.update_layout(
            **plotly_layout(
                height=190,
                x_title=None,
                y_title=None,
                margin=dict(t=8, b=8, l=8, r=52),
                xaxis=dict(visible=False),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    showgrid=False,
                    showline=False,
                ),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        type_mix = get_type_mix(df)
        tier_names = {"L": "Economy", "M": "Standard", "H": "Premium"}
        rows = {
            "Mean risk score": f"{summary['mean_risk']:.1f}",
            "95th percentile": f"{summary['p95_risk']:.0f}",
        }
        if "actual_failure_rate" in summary:
            rows["Recorded failure rate"] = f"{summary['actual_failure_rate']:.2f}%"
            rows["Flagged by model"] = f"{summary['flagged']:,}"
        for tier, share in sorted(type_mix.items()):
            rows[f"{tier_names.get(tier, tier)} tier"] = f"{share:.1f}%"

        st.markdown(render_kv_rows(rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------- watchlist
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-head"><div class="panel-title">Watchlist</div>'
        '<div class="panel-note">Highest risk first</div></div>',
        unsafe_allow_html=True,
    )

    watch_size = st.slider(
        "Assets to list", min_value=5, max_value=50, value=15, step=5,
        key="watchlist_size", label_visibility="collapsed",
    )

    watchlist = scored.nlargest(watch_size, "risk_score")

    if watchlist["risk_score"].max() <= config["risk"]["thresholds"]["low"]:
        st.markdown(
            render_notice(
                "Nothing above routine",
                "Every asset in the active dataset scores within the routine band. "
                "The watchlist still shows the highest scores so a trend is visible "
                "before it crosses a threshold.",
                "good",
            ),
            unsafe_allow_html=True,
        )

    display_cols = {
        "UDI": "Asset",
        "Product ID": "Serial",
        "Type": "Tier",
        "Air temperature [K]": "Air T (K)",
        "Process temperature [K]": "Process T (K)",
        "Rotational speed [rpm]": "Speed (rpm)",
        "Torque [Nm]": "Torque (Nm)",
        "Tool wear [min]": "Wear (min)",
        "risk_score": "Risk",
        "risk_category": "Band",
    }
    present = {k: v for k, v in display_cols.items() if k in watchlist.columns}
    table = watchlist[list(present)].rename(columns=present)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=min(420, 38 + 35 * len(table)),
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%.0f"
            ),
        },
    )

    csv = watchlist[list(present)].rename(columns=present).to_csv(index=False)
    st.download_button(
        "Export watchlist",
        csv,
        f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        render_footer(
            f"Scored with {best_name} &middot; calibrated probabilities",
            f"{summary['n_assets']:,} assets &middot; {summary['n_critical']} critical",
        ),
        unsafe_allow_html=True,
    )
