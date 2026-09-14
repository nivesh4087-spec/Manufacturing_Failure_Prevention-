"""
False Ceiling Defect Inspection & Ingestion Hub
=================================================
Industrial Visual Defect Detection & Telemetry Ingestion Module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


def draw_simulated_tile_inspection(tile_type: str, defect_type: str, confidence: float):
    """Generate a simulated high-resolution tile inspection canvas with detected defects."""
    import cv2

    # Create synthetic image of false ceiling tile
    h, w = 400, 600
    img = np.full((h, w, 3), 235, dtype=np.uint8)

    # Acoustic perforation texture pattern
    np.random.seed(42)
    for _ in range(300):
        cx = np.random.randint(20, w - 20)
        cy = np.random.randint(20, h - 20)
        r = np.random.randint(1, 3)
        cv2.circle(img, (cx, cy), r, (180, 180, 180), -1)

    # Grid border
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (80, 80, 80), 6)
    cv2.rectangle(img, (15, 15), (w - 15, h - 15), (140, 140, 140), 2)

    status_label = "PASS (NO DEFECT)"

    if defect_type == "Sagging / Deformation":
        status_label = f"DEFECT: SAGGING ({confidence:.1%})"
        cv2.ellipse(img, (w // 2, h // 2), (180, 100), 0, 0, 360, (140, 140, 140), 3)
        cv2.ellipse(img, (w // 2, h // 2), (120, 60), 0, 0, 360, (110, 110, 110), 2)
        cv2.rectangle(img, (120, 90), (480, 310), (239, 68, 68), 3)
        cv2.putText(img, status_label, (125, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2)

    elif defect_type == "Water Stain / Moisture":
        status_label = f"DEFECT: WATER STAIN ({confidence:.1%})"
        cv2.circle(img, (380, 160), 75, (160, 190, 210), -1)
        cv2.circle(img, (410, 190), 55, (140, 175, 200), -1)
        cv2.rectangle(img, (290, 70), (480, 250), (245, 158, 11), 3)
        cv2.putText(img, status_label, (295, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 158, 11), 2)

    elif defect_type == "Edge Chipping / Corner Crack":
        status_label = f"DEFECT: EDGE CHIP ({confidence:.1%})"
        pts = np.array([[10, 10], [80, 10], [10, 70]], np.int32)
        cv2.fillPoly(img, [pts], (40, 40, 40))
        cv2.rectangle(img, (5, 5), (110, 90), (239, 68, 68), 3)
        cv2.putText(img, status_label, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2)

    elif defect_type == "T-Grid Misalignment":
        status_label = f"DEFECT: GRID MISALIGNMENT ({confidence:.1%})"
        cv2.line(img, (w // 2, 10), (w // 2 + 35, h - 10), (249, 115, 22), 4)
        cv2.rectangle(img, (w // 2 - 20, 20), (w // 2 + 60, h - 20), (249, 115, 22), 3)
        cv2.putText(img, status_label, (w // 2 - 90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (249, 115, 22), 2)

    else:
        cv2.putText(img, status_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)

    return img


def render_page(project_root: Path, load_artifacts_fn, load_raw_dataset_fn):
    """Render False Ceiling Defect Inspection & Ingestion Hub page."""
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 6px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #f3f4f6;">
                    False Ceiling Defect Inspection & Data Ingestion System
                </h2>
                <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #9ca3af;">
                    Automated Optical Inspection (AOI) & Machinery Telemetry Stream Gateway for Ceiling Tile Manufacturing
                </p>
            </div>
            <div style="text-align: right;">
                <span class="status-badge status-normal">LIVE INGESTION ACTIVE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📹 Live Camera Stream & AOI",
        "🌐 Ingestion Data Pipeline Architecture",
        "⚙️ Machinery Telemetry & Correlation",
    ])

    with tab1:
        st.markdown("#### High-Speed Conveyor Camera Ingestion (RTSP / GigE Vision)")
        st.caption("Simulating real-time tile inspection stream from Line 2 Stamping & Finishing Conveyor.")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("##### Camera Stream Settings")
            camera_source = st.selectbox(
                "Ingestion Protocol / Source",
                ["RTSP Stream (rtsp://192.168.1.120:554/cam2)", "GigE Vision Industrial Camera", "HTTP REST API Post Webhook", "Hot-Folder File Watcher"],
                index=0
            )
            tile_model = st.selectbox("Inspection Target", ["Mineral Fiber Acoustic Tile 600x600mm", "Gypsum Board False Ceiling Panel", "Aluminum Metal T-Grid Panel"])
            simulated_defect = st.selectbox(
                "Simulated Condition",
                ["Normal (Pass)", "Sagging / Deformation", "Water Stain / Moisture", "Edge Chipping / Corner Crack", "T-Grid Misalignment"]
            )
            confidence_input = st.slider("Detection Confidence Threshold", 0.50, 0.99, 0.92, 0.01)

            st.markdown("---")
            st.markdown("##### Frame Diagnostics")
            st.markdown("- **Resolution**: `1920x1080 @ 60 FPS`\\n- **Inference Latency**: `12.4 ms`\\n- **Edge Hardware**: `NVIDIA Jetson AGX Orin`\\n- **Ingestion Buffer**: `0 Frames Dropped`")

        with col1:
            frame_img = draw_simulated_tile_inspection(tile_model, simulated_defect, confidence_input)
            st.image(frame_img, caption="Conveyor Inspection Camera — Bounding Box Overlay", use_container_width=True)

            if simulated_defect != "Normal (Pass)":
                st.error(
                    f"**INSPECTION ALERT** — {simulated_defect} detected on the conveyor line. "
                    f"Quality score {(1.0 - confidence_input) * 100:.1f}/100. Tile diverted to rework station."
                )
            else:
                st.success(
                    f"**INSPECTION PASS** — tile meets geometric and surface tolerance specification. "
                    f"Quality score {confidence_input * 100:.1f}/100."
                )

    with tab2:
        st.markdown("#### How Software Receives Data in Production Environments")
        st.markdown("""
        In industrial false ceiling manufacturing & installation projects, the software receives data through a multi-tiered ingestion architecture:

        - **1. Visual Inspection Stream (RTSP / GigE)**: High-speed overhead cameras stream video frames via RTSP (`rtsp://edge-cam:554/live`) or GigE Vision. The edge worker captures frames at 30-60 FPS, runs object detection to identify defects (cracks, sagging, moisture stains, T-grid gaps), and overlays bounding boxes.
        - **2. IoT Sensor Telemetry (MQTT / OPC-UA / Modbus)**: Machine sensors (accelerometers, thermocouples, torque encoders) publish telemetry metrics over MQTT topics or OPC-UA servers directly to the ingestion pipeline.
        - **3. Industrial PLC Push (REST API / Webhooks)**: Programmable Logic Controllers (Siemens S7, Allen-Bradley) trigger HTTP POST requests to `/api/v1/telemetry` or `/api/v1/inspect` on every cycle completion.
        - **4. Network File System Watcher**: Batch Automated Optical Inspection (AOI) scanners deposit high-res inspection TIFF/JPEG images or CSV batch logs into shared network folders watched by background daemons.
        """)

        st.info("📌 **Data Ingestion Flow**: Edge Capture (RTSP/MQTT) ➔ Real-time Preprocessing ➔ Machine Failure & Defect Inference Model ➔ Risk Score Computation ➔ Command Center & SCADA Alerting.")

    with tab3:
        st.markdown("#### Machinery Telemetry Correlation with Tile Defects")
        st.caption("Cross-correlating equipment metrics (Stamping Tool Wear, Hydraulic Torque, Temp) with Visual Defect Rates.")

        # Generate synthetic correlation data
        np.random.seed(42)
        n_points = 50
        tool_wear = np.linspace(10, 240, n_points)
        defect_rate = (tool_wear / 240.0) ** 2 * 12 + np.random.normal(0, 0.5, n_points)
        defect_rate = np.clip(defect_rate, 0, 15)

        fig = px.scatter(
            x=tool_wear,
            y=defect_rate,
            labels={"x": "Stamping Tool Wear (Minutes)", "y": "Tile Edge Defect Rate (%)"},
            title="Correlation: Stamping Tool Wear vs. False Ceiling Tile Edge Chipping",
            trendline="lowess",
            color=defect_rate,
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", family="Inter"),
            xaxis=dict(gridcolor="#1f2937", showline=True, linecolor="#374151"),
            yaxis=dict(gridcolor="#1f2937", showline=True, linecolor="#374151"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Key Finding**: When stamping tool wear exceeds **180 minutes**, false ceiling tile edge chipping increases exponentially (>8% defect rate). Replacing stamping blades proactively at 170 minutes eliminates 94% of tile edge defects.")
