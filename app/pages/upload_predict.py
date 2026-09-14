"""
Batch Analysis
==============
Score a whole file, table or database query at once.

Column mapping is attempted automatically against a list of known aliases, and
anything it cannot resolve is offered for manual mapping rather than failing.
The scoring itself is delegated to :mod:`src.inference.batch`, which is the same
code path the fleet overview uses — there is no second implementation to drift.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.styles import (
    panel,
    SERIES,
    TOKENS,
    plotly_layout,
    render_footer,
    render_masthead,
    render_notice,
    render_stat_tile,
    render_validation_item,
)

#: Internal feature name -> the label shown to the user.
REQUIRED_COLUMNS = {
    "air_temp_k": "Air temperature (K)",
    "process_temp_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
}

OPTIONAL_COLUMNS = {"type": "Board grade (L/M/H or 0/1/2)"}

#: Header spellings seen in the wild, per internal feature name. Matching is
#: case-insensitive and whitespace-trimmed, so only genuinely different wordings
#: need listing here.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "air_temp_k": [
        "Air temperature [K]", "Air temperature", "air_temp", "air_temperature",
        "AirTemp", "ambient_temp", "ambient_temperature",
    ],
    "process_temp_k": [
        "Process temperature [K]", "Process temperature", "process_temp",
        "process_temperature", "ProcessTemp", "machine_temp", "spindle_temp",
    ],
    "rotational_speed_rpm": [
        "Rotational speed [rpm]", "Rotational speed", "rotational_speed", "rpm",
        "RPM", "speed", "motor_speed", "spindle_speed",
    ],
    "torque_nm": ["Torque [Nm]", "Torque", "torque", "torque_nm", "motor_torque", "drive_torque"],
    "tool_wear_min": ["Tool wear [min]", "Tool wear", "tool_wear", "wear", "tool_wear_min"],
    "type": ["Type", "type", "product_type", "ProductType", "quality", "tier", "grade"],
}

BANDS = [
    ("LOW RISK", "good"),
    ("MODERATE RISK", "warning"),
    ("HIGH RISK", "serious"),
    ("CRITICAL RISK", "critical"),
]


def auto_map_columns(upload_columns) -> Dict[str, str]:
    """Match uploaded headers to internal feature names.

    Tries an exact match first, then a case-insensitive one, then the internal
    name itself — so a file already using internal names needs no aliases.

    Args:
        upload_columns: The uploaded file's column names.

    Returns:
        Internal feature name -> matched source column.
    """
    mapping: Dict[str, str] = {}
    lowered = {str(c).lower().strip(): c for c in upload_columns}

    for internal, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in upload_columns:
                mapping[internal] = alias
                break
        if internal not in mapping:
            for alias in aliases:
                key = alias.lower().strip()
                if key in lowered:
                    mapping[internal] = lowered[key]
                    break
        if internal not in mapping and internal.lower() in lowered:
            mapping[internal] = lowered[internal.lower()]

    return mapping


def _template_csv(config) -> str:
    """Build a CSV template from the configured demo scenarios.

    Deriving it from config rather than hardcoding rows means the template stays
    correct if the expected operating ranges ever change.
    """
    rows = []
    for i, (_, scenario) in enumerate(config["demo_scenarios"].items(), start=1):
        values = scenario["values"]
        rows.append(
            {
                "UDI": i,
                "Product ID": f"M{14860 + i}",
                "Type": {0: "L", 1: "M", 2: "H"}.get(values.get("type", 1), "M"),
                "Air temperature [K]": values["air_temp_k"],
                "Process temperature [K]": values["process_temp_k"],
                "Rotational speed [rpm]": values["rotational_speed_rpm"],
                "Torque [Nm]": values["torque_nm"],
                "Tool wear [min]": values["tool_wear_min"],
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def render_page(project_root: Path, load_artifacts_fn, load_raw_dataset_fn=None) -> None:
    """Render the Batch Analysis page."""
    try:
        artifacts, config = load_artifacts_fn()
    except Exception as exc:
        st.markdown(render_notice("Models unavailable", str(exc), "critical"), unsafe_allow_html=True)
        return

    best_name = artifacts.get("best_model_name", "model")

    st.markdown(
        render_masthead(
            "Batch Analysis",
            "Score a file, a database query or the bundled telemetry, then export "
            "the assets that need attention.",
            f"{best_name}<br>{datetime.now().strftime('%d %b %Y')}",
        ),
        unsafe_allow_html=True,
    )

    _render_source_picker(config, load_raw_dataset_fn)

    if "uploaded_dataset" not in st.session_state:
        _render_schema_help(config)
        return

    df = st.session_state["uploaded_dataset"]
    filename = st.session_state.get("uploaded_filename", "batch")

    _render_file_summary(df, filename)
    mapping = _render_mapping(df)

    if not all(key in mapping for key in REQUIRED_COLUMNS):
        st.markdown(
            render_notice(
                "Mapping incomplete",
                "Map every required reading before scoring. Anything left "
                "unmapped is a column the model cannot do without.",
                "warning",
            ),
            unsafe_allow_html=True,
        )
        return

    if st.button("Score this batch", type="primary", use_container_width=True):
        _score(df, mapping, artifacts, config)

    if "batch_results" in st.session_state:
        _render_results(df, st.session_state["batch_results"], config, best_name)


# ============================================================================
# Source
# ============================================================================

def _render_source_picker(config, load_raw_dataset_fn) -> None:
    """Render the file / database / sample source tabs."""
    with panel('Data source'):

        tab_file, tab_db, tab_sample = st.tabs(["File", "Database", "Bundled sample"])

        with tab_file:
            st.caption("CSV or Excel. Headers are matched automatically where possible.")
            st.download_button(
                "Download a template",
                _template_csv(config),
                "telemetry_template.csv",
                "text/csv",
                help="A correctly shaped file you can fill in.",
            )
            uploaded = st.file_uploader(
                "Telemetry file", type=["csv", "xlsx", "xls"], key="batch_uploader",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    frame = (
                        pd.read_excel(uploaded)
                        if uploaded.name.endswith((".xlsx", ".xls"))
                        else pd.read_csv(uploaded)
                    )
                    st.session_state["uploaded_dataset"] = frame
                    st.session_state["uploaded_filename"] = uploaded.name
                    st.session_state.pop("batch_results", None)
                    st.success(f"Loaded {len(frame):,} rows from {uploaded.name}.")
                except Exception as exc:
                    st.error(f"Could not read the file — {type(exc).__name__}: {exc}")

        with tab_db:
            st.caption("Pull telemetry straight from a plant historian or warehouse.")
            c1, c2 = st.columns([1, 2])
            with c1:
                db_type = st.selectbox(
                    "Engine",
                    ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Snowflake", "Oracle",
                     "Microsoft SQL Server"],
                    key="batch_db_type",
                )
            with c2:
                conn_str = st.text_input(
                    "Connection URI",
                    placeholder="postgresql://user:password@host:5432/plant",
                    key="batch_conn_str",
                )
            query_str = st.text_input(
                "Table or query", value="SELECT * FROM equipment_telemetry LIMIT 1000",
                key="batch_query_str",
            )
            if st.button("Connect and import", key="batch_db_btn"):
                if not conn_str.strip():
                    st.warning("Enter a connection URI first.")
                else:
                    try:
                        from src.data.loader import load_from_database

                        frame = load_from_database(db_type, conn_str, query_str)
                        st.session_state["uploaded_dataset"] = frame
                        st.session_state["uploaded_filename"] = f"{db_type} query"
                        st.session_state.pop("batch_results", None)
                        st.success(f"Imported {len(frame):,} rows.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")

        with tab_sample:
            st.caption("Load a slice of the bundled plant telemetry to try the workflow.")
            size = st.select_slider(
                "Rows", options=[100, 500, 1000, 2500, 5000], value=1000, key="batch_sample_size"
            )
            if st.button("Load sample", key="batch_sample_btn"):
                if load_raw_dataset_fn is None:
                    st.warning("No bundled dataset is available.")
                else:
                    frame = load_raw_dataset_fn().head(size).copy()
                    st.session_state["uploaded_dataset"] = frame
                    st.session_state["uploaded_filename"] = f"bundled sample ({size:,} rows)"
                    st.session_state.pop("batch_results", None)
                    st.rerun()




def _render_schema_help(_config) -> None:
    """Explain the expected schema when nothing has been loaded yet."""
    with panel('What the file needs'):
        left, right = st.columns(2)
        with left:
            st.markdown("**Required**")
            for internal, label in REQUIRED_COLUMNS.items():
                st.markdown(render_validation_item(f"{label} — <code>{internal}</code>", True),
                            unsafe_allow_html=True)
        with right:
            st.markdown("**Optional**")
            for internal, label in OPTIONAL_COLUMNS.items():
                st.markdown(render_validation_item(f"{label} — <code>{internal}</code>", False),
                            unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.78rem; color:var(--ink-muted); margin-top:10px;">'
                "Headers carrying units, such as <code>Air temperature [K]</code>, are "
                "recognised automatically. Anything unmatched can be mapped by hand "
                "after loading.</div>",
                unsafe_allow_html=True,
            )



# ============================================================================
# Summary & mapping
# ============================================================================

def _render_file_summary(df: pd.DataFrame, filename: str) -> None:
    """Render four tiles describing the loaded file."""
    missing = int(df.isnull().sum().sum())
    affected = int(df.isnull().any(axis=1).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_stat_tile("Source", filename[:26], "Active batch", "accent"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_stat_tile("Rows", f"{len(df):,}", "Assets to score", "neutral"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_stat_tile("Columns", str(len(df.columns)), "Detected in the file", "neutral"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_stat_tile(
                "Missing values", f"{missing:,}",
                "Nothing missing" if not missing else f"{affected:,} rows affected",
                "good" if not missing else "warning",
            ),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


def _render_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Render auto-detected mapping plus manual fallbacks. Returns the mapping."""
    with panel('Column mapping', 'Matched against known header spellings'):

        auto = auto_map_columns(list(df.columns))
        mapping = dict(auto)

        left, right = st.columns([3, 2], gap="medium")
        with left:
            for internal, label in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items():
                if internal in auto:
                    st.markdown(
                        render_validation_item(f"{label} &rarr; <code>{auto[internal]}</code>", True),
                        unsafe_allow_html=True,
                    )
                else:
                    required = internal in REQUIRED_COLUMNS
                    st.markdown(
                        render_validation_item(
                            f"{label} — not found"
                            + ("" if required else " (a default will be used)"),
                            not required,
                        ),
                        unsafe_allow_html=True,
                    )
        with right:
            st.dataframe(df.head(5), use_container_width=True, hide_index=True, height=190)

        unmapped = [k for k in REQUIRED_COLUMNS if k not in mapping]
        if unmapped:
            st.markdown(
                '<div style="margin-top:12px;" class="panel-title">Map the rest by hand</div>',
                unsafe_allow_html=True,
            )
            options = ["— not available —"] + list(df.columns)
            cols = st.columns(min(3, len(unmapped)))
            for i, internal in enumerate(unmapped):
                with cols[i % len(cols)]:
                    chosen = st.selectbox(
                        REQUIRED_COLUMNS[internal], options, key=f"map_{internal}"
                    )
                    if chosen != options[0]:
                        mapping[internal] = chosen

        if "type" not in mapping:
            options = ["— not available —"] + list(df.columns)
            chosen = st.selectbox(OPTIONAL_COLUMNS["type"], options, key="map_type")
            if chosen != options[0]:
                mapping["type"] = chosen


    return mapping


# ============================================================================
# Scoring
# ============================================================================

def _score(df: pd.DataFrame, mapping: Dict[str, str], artifacts, config) -> None:
    """Score the batch and stash the result in session state."""
    from src.inference.batch import score_frame

    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]

    with st.spinner(f"Scoring {len(df):,} assets..."):
        try:
            results = score_frame(
                df, mapping, model, artifacts["scaler"], config,
                artifacts["feature_stats"], artifacts["feature_names"],
                artifacts["numerical_cols"],
            )
            st.session_state["batch_results"] = results
        except Exception as exc:
            st.markdown(
                render_notice("Scoring failed", f"{type(exc).__name__}: {exc}", "critical"),
                unsafe_allow_html=True,
            )


def _render_results(df: pd.DataFrame, results: pd.DataFrame, config, best_name: str) -> None:
    """Render the scored-batch dashboard and exports."""
    valid = results[results["failure_probability"].notna()]
    if valid.empty:
        st.markdown(
            render_notice("No scores produced", "Every row failed to score.", "critical"),
            unsafe_allow_html=True,
        )
        return

    n_total = len(valid)
    n_flagged = int((valid["prediction"] == "FAILURE").sum())
    elevated = int(valid["risk_category"].isin(["HIGH RISK", "CRITICAL RISK"]).sum())
    mean_risk = float(valid["risk_score"].mean())

    business = config.get("business", {})
    failure_cost = business.get("downtime_cost_per_hour", 10000.0) * business.get(
        "avg_downtime_hours", 4.0
    )
    preventive = business.get("preventive_action_cost", 1500.0)

    # Expected cost if nothing is done: each asset's probability times the cost
    # of a failure. Expected cost if the flagged assets are serviced: a planned
    # fix for those, residual risk for the rest.
    probabilities = valid["failure_probability"]
    do_nothing = float((probabilities * failure_cost).sum())
    intervene = float(
        sum(
            preventive if pred == "FAILURE" else p * failure_cost
            for p, pred in zip(probabilities, valid["prediction"])
        )
    )
    avoided = do_nothing - intervene

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            render_stat_tile("Scored", f"{n_total:,}", "Assets in this batch", "neutral"),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_stat_tile(
                "Flagged", f"{n_flagged:,}",
                f"{n_flagged / n_total * 100:.1f}% of the batch",
                "critical" if n_flagged else "good",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_stat_tile(
                "Needs attention", f"{elevated:,}", "High and critical bands",
                "serious" if elevated else "good",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            render_stat_tile("Mean risk", f"{mean_risk:.1f}", "Out of 100", "neutral"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            render_stat_tile(
                "Expected cost, no action", f"${do_nothing:,.0f}",
                "Probability-weighted failure cost", "critical",
            ),
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            render_stat_tile(
                "Expected cost, acting on flags", f"${intervene:,.0f}",
                "Planned fixes plus residual risk", "warning",
            ),
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            render_stat_tile(
                "Difference", f"${avoided:,.0f}",
                "Estimate over this batch only",
                "good" if avoided > 0 else "warning",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")

    with left:
        with panel('Risk bands'):
            counts = valid["risk_category"].value_counts()
            values = [int(counts.get(band, 0)) for band, _ in BANDS]
            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=[band.replace(" RISK", "").title() for band, _ in BANDS],
                    orientation="h",
                    marker_color=[TOKENS[status] for _, status in BANDS],
                    marker_line_width=0,
                    text=[f"{v:,}" for v in values],
                    textposition="auto",
                    textfont=dict(size=11, color=TOKENS["ink_secondary"]),
                    hovertemplate="%{y}: %{x:,} assets<extra></extra>",
                )
            )
            fig.update_layout(
                **plotly_layout(
                    height=230,
                    margin=dict(t=8, b=8, l=8, r=16),
                    xaxis=dict(visible=False, range=[0, max(values or [1]) * 1.02]),
                    yaxis=dict(autorange="reversed", showgrid=False, showline=False,
                               tickfont=dict(size=11, color=TOKENS["ink_secondary"])),
                )
            )
            st.plotly_chart(fig, use_container_width=True)


    with right:
        with panel('Score distribution'):
            fig = go.Figure(
                go.Histogram(
                    x=valid["risk_score"], nbinsx=25,
                    marker_color=SERIES[0], marker_line_width=0,
                    hovertemplate="Risk %{x}<br>%{y} assets<extra></extra>",
                )
            )
            fig.add_vline(
                x=config["early_warning"]["threshold"], line_dash="dash",
                line_color=TOKENS["serious"], line_width=1,
                annotation_text="Alert threshold",
                annotation_font=dict(size=10, color=TOKENS["serious"]),
            )
            fig.update_layout(
                **plotly_layout(height=230, x_title="Risk score", y_title="Assets", bargap=0.04)
            )
            st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------------ table
    with panel('Results', 'Highest risk first'):

        table = results.copy()
        table.insert(0, "Row", table.index)
        table["failure_probability"] = (table["failure_probability"] * 100).round(2)
        table = table.rename(
            columns={
                "prediction": "Call",
                "failure_probability": "Probability %",
                "risk_score": "Risk",
                "risk_category": "Band",
            }
        ).sort_values("Risk", ascending=False)

        st.dataframe(
            table, use_container_width=True, hide_index=True, height=380,
            column_config={
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f")
            },
        )


    # ------------------------------------------------------------ export
    export = pd.concat(
        [df.reset_index(drop=True), results.reset_index(drop=True)], axis=1
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Export every row", export.to_csv(index=False),
            f"batch_scores_{stamp}.csv", "text/csv", use_container_width=True,
        )
    with e2:
        attention = export[export["risk_category"].isin(["HIGH RISK", "CRITICAL RISK"])]
        if attention.empty:
            st.markdown(
                render_notice("Nothing elevated", "No asset in this batch needs attention.", "good"),
                unsafe_allow_html=True,
            )
        else:
            st.download_button(
                f"Export the {len(attention):,} needing attention",
                attention.to_csv(index=False),
                f"attention_{stamp}.csv", "text/csv", use_container_width=True,
            )

    st.markdown(
        render_footer(
            f"Scored with {best_name}",
            "Verify equipment condition before scheduling work",
        ),
        unsafe_allow_html=True,
    )
