"""
Data Explorer Module
====================
Interactive dataset exploration with filtering and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.components.data_access import active_dataset
from app.components.styles import (
    panel,
    SERIES,
    TOKENS,
    plotly_layout,
    render_masthead,
    render_notice,
    render_stat_tile,
)


def render_page(project_root, load_dataset_fn):
    """Render the Data Explorer page."""

    try:
        df, source_label = active_dataset(load_dataset_fn)
    except Exception as exc:
        st.markdown(
            render_notice("Dataset unavailable", str(exc), "critical"), unsafe_allow_html=True
        )
        return

    st.markdown(
        render_masthead(
            "Data Explorer",
            "The raw sensor record behind every prediction — distributions, "
            "relationships and the failure modes recorded alongside them.",
            f"{source_label}<br>{len(df):,} rows &times; {len(df.columns)} columns",
        ),
        unsafe_allow_html=True,
    )

    # Detect available columns dynamically
    all_columns = list(df.columns)
    numerical_cols_detected = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    target_col_default = "Machine failure"

    # Check if standard target column exists
    target_col = target_col_default if target_col_default in df.columns else None

    tabs = st.tabs(["Overview", "Distributions", "Correlations", "Filter"])


    # ========================================================================
    # TAB 1 — Dataset Overview
    # ========================================================================

    with tabs[0]:
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                render_stat_tile("Records", f"{len(df):,}", source_label, "neutral"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_stat_tile("Columns", str(len(df.columns)), "Including labels", "neutral"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_stat_tile(
                    "Missing values", f"{missing:,}",
                    "Complete record" if not missing else "Needs imputation",
                    "good" if not missing else "warning",
                ),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                render_stat_tile(
                    "Duplicate rows", f"{duplicates:,}",
                    "None found" if not duplicates else "Review before training",
                    "good" if not duplicates else "warning",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown("#### Sample rows")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        st.markdown("#### Statistical summary")
        st.dataframe(
            df.describe().round(3).T,
            use_container_width=True,
        )

        st.markdown("#### Column inventory")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
            "Null": [int(df[c].isna().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ========================================================================
    # TAB 2 — Feature Analysis
    # ========================================================================

    with tabs[1]:
        target_col = "Machine failure"

        st.caption(
            "Each reading split by whether the asset went on to fail. Where the "
            "two distributions separate, the sensor carries signal."
        )

        numerical_cols = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
        ]

        available = [c for c in numerical_cols if c in df.columns]
        chosen = st.multiselect(
            "Readings to show", available, default=available[:2], key="de_dist_cols"
        )

        for col in chosen:
            with panel('{col}'):
                fig = go.Figure()
                for label, color, name in [
                    (0, SERIES[0], "Ran normally"),
                    (1, TOKENS["critical"], "Failed"),
                ]:
                    fig.add_trace(go.Histogram(
                        x=df[df[target_col] == label][col],
                        name=name,
                        marker_color=color,
                        marker_line_width=0,
                        opacity=0.65,
                        nbinsx=40,
                        histnorm="probability density",
                    ))
                fig.update_layout(
                    **plotly_layout(
                        height=260,
                        show_legend=True,
                        x_title=col,
                        y_title="Density",
                        barmode="overlay",
                        bargap=0.04,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)


        with panel('Spread comparison', 'Median, quartiles and outliers'):
            selected_feat = st.selectbox("Reading", available, key="de_box_feat")
            if selected_feat in df.columns:
                labelled = df.assign(
                    _outcome=df[target_col].map({0: "Ran normally", 1: "Failed"})
                )
                fig = px.box(
                    labelled, x="_outcome", y=selected_feat, color="_outcome",
                    color_discrete_map={
                        "Ran normally": SERIES[0], "Failed": TOKENS["critical"]
                    },
                    category_orders={"_outcome": ["Ran normally", "Failed"]},
                )
                fig.update_layout(
                    **plotly_layout(height=320, x_title=None, y_title=selected_feat)
                )
                st.plotly_chart(fig, use_container_width=True)


        # Failure type breakdown
        st.markdown("#### Recorded failure modes")
        failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        failure_counts = {
            "Tool Wear Failure (TWF)": int(df["TWF"].sum()) if "TWF" in df.columns else 0,
            "Heat Dissipation (HDF)": int(df["HDF"].sum()) if "HDF" in df.columns else 0,
            "Power Failure (PWF)": int(df["PWF"].sum()) if "PWF" in df.columns else 0,
            "Overstrain (OSF)": int(df["OSF"].sum()) if "OSF" in df.columns else 0,
            "Random Failure (RNF)": int(df["RNF"].sum()) if "RNF" in df.columns else 0,
        }

        fig = go.Figure(go.Bar(
            x=list(failure_counts.values()),
            y=list(failure_counts.keys()),
            orientation="h",
            # One hue: these are five slices of the same quantity, not five
            # independent series, so magnitude carries the meaning.
            marker_color=SERIES[0],
            marker_line_width=0,
            text=[str(v) for v in failure_counts.values()],
            textposition="outside",
            textfont=dict(size=11, color=TOKENS["ink_secondary"]),
            hovertemplate="%{y}: %{x} events<extra></extra>",
        ))
        fig.update_layout(
            **plotly_layout(
                height=250,
                x_title="Recorded events",
                margin=dict(t=8, b=42, l=8, r=46),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    showgrid=False, showline=False, automargin=True,
                ),
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "These five labels are the recorded root cause of each failure. They "
            "are held out of training — a model given them would simply read the "
            "answer instead of learning the sensor signature that precedes it."
        )

    # ========================================================================
    # TAB 3 — Correlations
    # ========================================================================

    with tabs[2]:
        with panel('Correlation matrix', 'Pearson, pairwise'):

            corr = df[available + [target_col]].corr()
            short = [c.split(" [")[0] for c in corr.columns]

            # Diverging: blue for negative, red for positive, neutral grey at zero.
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=short,
                y=short,
                colorscale=[
                    [0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, TOKENS["surface_raised"]],
                    [0.75, "#e07a6a"], [1.0, "#a32a2a"],
                ],
                zmid=0, zmin=-1, zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10, "color": TOKENS["ink"]},
                hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
                xgap=2, ygap=2,
                colorbar=dict(
                    thickness=10, len=0.7,
                    tickfont=dict(size=10, color=TOKENS["ink_muted"]),
                    outlinewidth=0,
                ),
            ))
            fig.update_layout(
                **plotly_layout(
                    height=430, margin=dict(t=8, b=8, l=8, r=8),
                    xaxis=dict(showgrid=False, showline=False, tickangle=-30,
                               tickfont=dict(size=10, color=TOKENS["ink_muted"])),
                    yaxis=dict(showgrid=False, showline=False, autorange="reversed",
                               tickfont=dict(size=10, color=TOKENS["ink_muted"])),
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Torque and rotational speed are strongly negatively correlated — "
                "the drive trades one against the other to hold power roughly "
                "constant. That relationship is why their ratio is an engineered feature."
            )


        with panel('Pairwise view', 'Pick any two readings'):
            sc1, sc2 = st.columns(2)
            with sc1:
                x_feat = st.selectbox("Horizontal", available, index=min(3, len(available) - 1),
                                      key="de_x")
            with sc2:
                y_feat = st.selectbox("Vertical", available, index=min(2, len(available) - 1),
                                      key="de_y")

            labelled = df.assign(_outcome=df[target_col].map({0: "Ran normally", 1: "Failed"}))
            fig = px.scatter(
                labelled, x=x_feat, y=y_feat, color="_outcome",
                color_discrete_map={"Ran normally": SERIES[0], "Failed": TOKENS["critical"]},
                category_orders={"_outcome": ["Ran normally", "Failed"]},
                opacity=0.5,
            )
            fig.update_traces(marker=dict(size=5, line=dict(width=0)))
            fig.update_layout(
                **plotly_layout(
                    height=390, show_legend=True, x_title=x_feat, y_title=y_feat,
                    legend_title_text="",
                )
            )
            st.plotly_chart(fig, use_container_width=True)


    # ========================================================================
    # TAB 4 — Interactive Filter
    # ========================================================================

    with tabs[3]:
        st.caption("Narrow the record down, then export what you selected.")

        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            type_filter = st.multiselect(
                "Product Type",
                df["Type"].unique() if "Type" in df.columns else [],
                default=list(df["Type"].unique()) if "Type" in df.columns else [],
            )

        with fcol2:
            failure_filter = st.multiselect(
                "Failure Status",
                [0, 1], default=[0, 1],
                format_func=lambda x: "No Failure" if x == 0 else "Failure",
            )

        with fcol3:
            torque_range = st.slider(
                "Torque Range [Nm]",
                float(df["Torque [Nm]"].min()),
                float(df["Torque [Nm]"].max()),
                (float(df["Torque [Nm]"].min()), float(df["Torque [Nm]"].max())),
            )

        # Apply filters
        filtered = df.copy()
        if "Type" in filtered.columns and type_filter:
            filtered = filtered[filtered["Type"].isin(type_filter)]
        if failure_filter is not None:
            filtered = filtered[filtered[target_col].isin(failure_filter)]
        filtered = filtered[
            (filtered["Torque [Nm]"] >= torque_range[0]) &
            (filtered["Torque [Nm]"] <= torque_range[1])
        ]

        st.metric("Filtered Records", f"{len(filtered):,}")
        st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
