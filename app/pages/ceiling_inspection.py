"""
Defect Inspection
=================
Line 2: the optical inspection cell that checks finished tiles, and the link
between its reject rate and the condition of the Line 1 machinery.

The inspection cell is described by ``config.inspection`` rather than by literals
here. The tool-wear relationship is computed from the dataset at runtime, so the
figures on this page are measurements rather than claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data_access import active_dataset
from app.components.styles import (
    SERIES,
    TOKENS,
    panel,
    plotly_layout,
    render_footer,
    render_kv_rows,
    render_masthead,
    render_notice,
    render_pill,
    render_stat_tile,
)

#: BGR colours for the synthetic frame, drawn from the shared status tokens so
#: the overlay matches the rest of the interface.
_OVERLAY_BGR = {
    "critical": (59, 59, 208),
    "serious": (90, 131, 236),
    "warning": (25, 178, 250),
    "good": (12, 163, 12),
}


def draw_inspection_frame(defect_name: str, severity: str, confidence: float) -> np.ndarray:
    """Render a synthetic inspection frame with the detection overlay.

    This stands in for a live camera feed so the page is demonstrable without
    plant hardware attached. The geometry is illustrative; the detection methods
    it represents are listed in ``config.inspection.defects``.

    Args:
        defect_name: Which defect to draw, or ``"None"`` for a passing tile.
        severity: Status token driving the overlay colour.
        confidence: Detector confidence, shown in the caption.

    Returns:
        An RGB image array ready for ``st.image``.
    """
    import cv2

    h, w = 420, 640
    img = np.full((h, w, 3), 232, dtype=np.uint8)

    # Acoustic perforation texture, seeded so the frame is stable across reruns.
    rng = np.random.default_rng(42)
    for _ in range(340):
        cx, cy = int(rng.integers(24, w - 24)), int(rng.integers(24, h - 24))
        cv2.circle(img, (cx, cy), int(rng.integers(1, 3)), (186, 186, 186), -1)

    cv2.rectangle(img, (12, 12), (w - 12, h - 12), (84, 84, 84), 5)
    cv2.rectangle(img, (18, 18), (w - 18, h - 18), (148, 148, 148), 2)

    colour = _OVERLAY_BGR.get(severity, _OVERLAY_BGR["critical"])
    label = f"{defect_name.upper()} {confidence:.0%}"

    if defect_name == "Sagging or warp":
        cv2.ellipse(img, (w // 2, h // 2), (190, 104), 0, 0, 360, (142, 142, 142), 3)
        cv2.ellipse(img, (w // 2, h // 2), (126, 62), 0, 0, 360, (112, 112, 112), 2)
        cv2.rectangle(img, (126, 94), (514, 326), colour, 3)
        cv2.putText(img, label, (130, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)

    elif defect_name == "Surface water stain":
        cv2.circle(img, (408, 168), 78, (212, 192, 162), -1)
        cv2.circle(img, (438, 198), 56, (202, 178, 142), -1)
        cv2.rectangle(img, (310, 74), (512, 262), colour, 3)
        cv2.putText(img, label, (314, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)

    elif defect_name == "Edge chipping":
        cv2.fillPoly(img, [np.array([[12, 12], [92, 12], [12, 78]], np.int32)], (44, 44, 44))
        cv2.rectangle(img, (8, 8), (122, 98), colour, 3)
        cv2.putText(img, label, (12, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)

    elif defect_name == "T-grid misalignment":
        cv2.line(img, (w // 2, 12), (w // 2 + 38, h - 12), colour, 4)
        cv2.rectangle(img, (w // 2 - 24, 22), (w // 2 + 66, h - 22), colour, 3)
        cv2.putText(img, label, (w // 2 - 96, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)

    else:
        good = _OVERLAY_BGR["good"]
        cv2.rectangle(img, (24, 24), (w - 24, h - 24), good, 2)
        cv2.putText(img, f"PASS {confidence:.0%}", (28, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, good, 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def tool_wear_failure_profile(
    _df: pd.DataFrame, target_col: str, bin_width: int = 20
) -> pd.DataFrame:
    """Measure how the failure rate rises with accumulated tool wear.

    This replaces what used to be a randomly generated curve. Binning the real
    dataset gives a defensible basis for a replacement interval instead of an
    asserted one.

    Args:
        _df: The raw dataset.
        target_col: Name of the failure column.
        bin_width: Width of each tool-wear bucket, in minutes.

    Returns:
        One row per bucket with ``assets``, ``failures``, ``failure_rate`` and,
        where the column exists, the tool-wear-specific ``twf`` count.
    """
    wear_col = "Tool wear [min]"
    frame = _df[[wear_col, target_col]].copy()
    if "TWF" in _df.columns:
        frame["TWF"] = _df["TWF"]

    upper = int(np.ceil(frame[wear_col].max() / bin_width) * bin_width)
    edges = list(range(0, upper + bin_width, bin_width))
    frame["bucket"] = pd.cut(frame[wear_col], bins=edges, right=True)

    agg = {"assets": (target_col, "size"), "failures": (target_col, "sum")}
    if "TWF" in frame.columns:
        agg["twf"] = ("TWF", "sum")

    grouped = frame.groupby("bucket", observed=True).agg(**agg).reset_index()
    # .map over a categorical returns a categorical; cast so the midpoints
    # can be compared and plotted numerically.
    grouped["wear_midpoint"] = grouped["bucket"].map(
        lambda b: (b.left + b.right) / 2
    ).astype(float)
    grouped["failure_rate"] = grouped["failures"] / grouped["assets"] * 100
    if "twf" in grouped.columns:
        grouped["twf_rate"] = grouped["twf"] / grouped["assets"] * 100
    return grouped


def recommended_change_point(profile: pd.DataFrame) -> Dict[str, Any]:
    """Derive a tool-change threshold from the measured wear profile.

    Picks the first bucket whose failure rate exceeds twice the rate of the
    healthy region, then reports what changing the tooling at that point would
    have avoided across the dataset.

    Args:
        profile: Output of :func:`tool_wear_failure_profile`.

    Returns:
        A dict with the threshold and the avoidable-failure arithmetic, or an
        empty dict if no clear change point exists.
    """
    if profile.empty:
        return {}

    baseline = float(profile["failure_rate"].head(len(profile) // 2).mean())
    elevated = profile[profile["failure_rate"] > baseline * 2]
    if elevated.empty:
        return {}

    first = elevated.iloc[0]
    threshold = int(first["bucket"].left)

    after = profile[profile["wear_midpoint"] > threshold]
    if "twf" in profile.columns:
        avoidable = int(after["twf"].sum())
        total = int(profile["twf"].sum())
        kind = "tool-wear failures"
    else:
        avoidable = int(after["failures"].sum())
        total = int(profile["failures"].sum())
        kind = "failures"

    return {
        "threshold": threshold,
        "baseline_rate": baseline,
        "elevated_rate": float(first["failure_rate"]),
        "avoidable": avoidable,
        "total": total,
        "share": (avoidable / total * 100) if total else 0.0,
        "kind": kind,
    }


def render_page(project_root: Path, load_artifacts_fn, load_raw_dataset_fn) -> None:
    """Render the Defect Inspection page."""
    try:
        _, config = load_artifacts_fn()
    except Exception as exc:
        st.markdown(render_notice("Configuration unavailable", str(exc), "critical"),
                    unsafe_allow_html=True)
        return

    setup = config.get("inspection", {})
    capture = setup.get("capture", {})
    defects: List[Dict[str, str]] = setup.get("defects", [])

    st.markdown(
        render_masthead(
            "Defect Inspection",
            f"{setup.get('line_name', 'Inspection line')} — optical inspection of "
            "finished tiles, and how its reject rate tracks machine condition upstream.",
            f"{capture.get('resolution', '—')} at {capture.get('frame_rate_fps', '—')} fps"
            f"<br>{capture.get('inference_latency_ms', '—')} ms per frame",
        ),
        unsafe_allow_html=True,
    )

    tab_cell, tab_link, tab_ingest = st.tabs(
        ["Inspection cell", "Link to machine health", "How data arrives"]
    )

    with tab_cell:
        _render_cell(setup, capture, defects)

    with tab_link:
        _render_link(config, load_raw_dataset_fn)

    with tab_ingest:
        _render_ingestion(capture)

    st.markdown(
        render_footer(
            setup.get("line_name", "Inspection line"),
            f"{len(defects)} defect classes configured",
        ),
        unsafe_allow_html=True,
    )


# ============================================================================
# Inspection cell
# ============================================================================

def _render_cell(setup: Dict[str, Any], capture: Dict[str, Any], defects) -> None:
    """Render the simulated camera view and its controls."""
    frame_col, control_col = st.columns([2, 1], gap="medium")

    with control_col:
        with panel("Cell settings"):
            sources = capture.get("sources", [])
            st.selectbox(
                "Capture source",
                [s["label"] for s in sources] or ["Not configured"],
                key="insp_source",
            )
            st.selectbox(
                "Product", setup.get("products", ["Not configured"]), key="insp_product"
            )
            defect_names = ["None — passing tile"] + [d["name"] for d in defects]
            chosen = st.selectbox("Simulate condition", defect_names, key="insp_defect")
            confidence = st.slider(
                "Detector confidence", 0.50, 0.99, 0.92, 0.01, key="insp_conf"
            )

        with panel("Frame diagnostics"):
            st.markdown(
                render_kv_rows(
                    {
                        "Resolution": capture.get("resolution", "—"),
                        "Frame rate": f"{capture.get('frame_rate_fps', '—')} fps",
                        "Inference": f"{capture.get('inference_latency_ms', '—')} ms",
                        "Edge device": capture.get("edge_device", "—"),
                    }
                ),
                unsafe_allow_html=True,
            )

    selected = next((d for d in defects if d["name"] == chosen), None)
    severity = selected["severity"] if selected else "good"
    defect_name = selected["name"] if selected else "None"

    with frame_col:
        with panel("Conveyor camera", "Detection overlay"):
            try:
                frame = draw_inspection_frame(defect_name, severity, confidence)
                st.image(frame, use_container_width=True)
            except ImportError:
                st.markdown(
                    render_notice(
                        "OpenCV not installed",
                        "The frame preview needs <code>opencv-python-headless</code>. "
                        "Install the requirements to enable it.",
                        "warning",
                    ),
                    unsafe_allow_html=True,
                )

            if selected:
                st.markdown(
                    render_notice(
                        f"Reject — {selected['name']}",
                        f"Detected by {selected['method'].lower()} at "
                        f"{confidence:.0%} confidence. Tile diverted to rework.<br>"
                        f"<strong>Likely upstream cause:</strong> {selected['cause']}",
                        severity,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    render_notice(
                        "Pass",
                        f"Tile is within geometric and surface tolerance at "
                        f"{confidence:.0%} confidence.",
                        "good",
                    ),
                    unsafe_allow_html=True,
                )

    with panel("Defect catalogue", "What the cell looks for, and why it happens"):
        if defects:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Defect": d["name"],
                            "Detection method": d["method"],
                            "Upstream cause": d["cause"],
                            "Severity": d["severity"].title(),
                        }
                        for d in defects
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No defect classes configured.")


# ============================================================================
# Link to machine health
# ============================================================================

def _render_link(config: Dict[str, Any], load_raw_dataset_fn) -> None:
    """Render the measured relationship between tool wear and failures."""
    df, source_label = active_dataset(load_raw_dataset_fn)
    target_col = config["data"]["target_column"]

    if "Tool wear [min]" not in df.columns or target_col not in df.columns:
        st.markdown(
            render_notice(
                "Not available for this dataset",
                "This view needs the tool wear and failure columns, which the "
                "active dataset does not carry.",
                "warning",
            ),
            unsafe_allow_html=True,
        )
        return

    profile = tool_wear_failure_profile(df, target_col)
    finding = recommended_change_point(profile)

    st.markdown(
        render_notice(
            "Why a machine model belongs on a quality page",
            "Edge chipping is caused by worn tooling, not by the tile. If the "
            "punch and die are replaced before they degrade, the defect never "
            "reaches Line 2. Everything below is measured from "
            f"{len(df):,} assets in the active dataset ({source_label.lower()}), "
            "not assumed.",
            "accent",
        ),
        unsafe_allow_html=True,
    )

    if finding:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                render_stat_tile(
                    "Change tooling at", f"{finding['threshold']} min",
                    "Where the failure rate first doubles", "accent",
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_stat_tile(
                    "Failure rate after that point",
                    f"{finding['elevated_rate']:.1f}%",
                    f"Against {finding['baseline_rate']:.1f}% while healthy",
                    "critical",
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_stat_tile(
                    "Avoidable at that interval",
                    f"{finding['share']:.0f}%",
                    f"{finding['avoidable']} of {finding['total']} {finding['kind']}",
                    "good",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    with panel("Failure rate against accumulated tool wear",
               "Each bar is a 20-minute bucket of the real dataset"):
        colours = [
            TOKENS["critical"]
            if finding and mid > finding["threshold"]
            else SERIES[0]
            for mid in profile["wear_midpoint"]
        ]
        fig = go.Figure(
            go.Bar(
                x=profile["wear_midpoint"],
                y=profile["failure_rate"],
                marker_color=colours,
                marker_line_width=0,
                width=18,
                customdata=profile[["assets", "failures"]].values,
                hovertemplate="Tool wear around %{x:.0f} min<br>"
                "Failure rate %{y:.2f}%<br>"
                "%{customdata[1]} of %{customdata[0]} assets<extra></extra>",
            )
        )
        if finding:
            fig.add_vline(
                x=finding["threshold"],
                line_dash="dash",
                line_color=TOKENS["warning"],
                line_width=1,
                annotation_text=f"Change at {finding['threshold']} min",
                annotation_font=dict(size=10, color=TOKENS["warning"]),
            )
        fig.update_layout(
            **plotly_layout(
                height=320,
                x_title="Accumulated tool wear (minutes)",
                y_title="Failure rate (%)",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        if finding:
            st.markdown(
                render_notice(
                    f"The curve is flat until about {finding['threshold']} minutes",
                    f"Below that point the failure rate sits near "
                    f"{finding['baseline_rate']:.1f}% and barely moves — wear alone "
                    f"is not causing failures. Past it the rate climbs steeply. "
                    f"Replacing tooling at {finding['threshold']} minutes would have "
                    f"put {finding['avoidable']} of the {finding['total']} "
                    f"{finding['kind']} ({finding['share']:.0f}%) on the safe side of "
                    "the line, which is the entire argument for a wear-based "
                    "replacement interval over a fixed calendar one.",
                    "good",
                ),
                unsafe_allow_html=True,
            )

    with st.expander("The numbers behind the chart"):
        table = profile.copy()
        table["Tool wear"] = table["bucket"].astype(str)
        cols = {"Tool wear": "Tool wear (min)", "assets": "Assets",
                "failures": "Failures", "failure_rate": "Failure rate %"}
        if "twf" in table.columns:
            cols["twf"] = "Tool-wear failures"
        st.dataframe(
            table[list(cols)].rename(columns=cols).round(2),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================================
# Ingestion
# ============================================================================

def _render_ingestion(capture: Dict[str, Any]) -> None:
    """Describe how telemetry and frames reach the platform."""
    st.markdown(
        "Nothing on this platform assumes a person uploads a file. In a running "
        "plant, four paths feed it, and the dashboard is the read-only end of them."
    )

    for i, source in enumerate(capture.get("sources", []), start=1):
        st.markdown(
            f'<div class="action action-low">'
            f'<div class="action-head"><span class="action-title">{i}. {source["label"]}</span>'
            f'{render_pill("capture", "accent")}</div>'
            f'<div class="action-trigger">{source["detail"]}</div></div>',
            unsafe_allow_html=True,
        )

    with panel("Sensor and control paths"):
        st.markdown(
            """
| Path | Protocol | What it carries |
|:--|:--|:--|
| Machine telemetry | MQTT, OPC-UA, Modbus | Thermocouples, torque encoders, spindle tachometers |
| Cycle events | HTTP POST from the PLC | One record per completed cut |
| Batch logs | Watched network share | CSV drops from the AOI scanner |
| Historian backfill | SQL query | Replaying past shifts for model retraining |

Everything lands in the same schema the **Batch Analysis** page accepts, so a
file drop and a live feed follow the identical code path — which is what makes
the file-upload workflow a genuine test of the production one.
            """
        )
