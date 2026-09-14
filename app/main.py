"""
Predictive Maintenance & Inspection Platform
============================================
Streamlit entry point: theme injection, sidebar, cold-start handling and routing.

Usage:
    streamlit run app/main.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Resolve the project root and drop `app/` from sys.path, or `app.pages` shadows
# the stdlib-adjacent name `pages` that Streamlit itself looks for.
project_root = Path(__file__).resolve().parent.parent
app_dir = str(Path(__file__).resolve().parent)
if app_dir in sys.path:
    sys.path.remove(app_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Predictive Maintenance & Inspection Platform",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.styles import (  # noqa: E402  (must follow set_page_config)
    get_custom_css,
    render_footer,
    render_kv_rows,
    render_notice,
    render_pill,
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# ============================================================================
# Cached loaders
# ============================================================================

@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """Load model artifacts and configuration. Cached for the server's lifetime."""
    from src.data.loader import load_config
    from src.models.trainer import load_model_artifacts

    config = load_config(str(project_root / "config" / "config.yaml"))
    artifacts = load_model_artifacts(config, project_root)
    return artifacts, config


@st.cache_data(show_spinner="Loading telemetry...")
def load_raw_dataset():
    """Load the raw sensor dataset."""
    from src.data.loader import load_config, load_dataset

    config = load_config(str(project_root / "config" / "config.yaml"))
    return load_dataset(config)


@st.cache_data(show_spinner=False)
def load_results_json(filename: str):
    """Load a JSON results file written by the training pipeline, if present."""
    filepath = project_root / "reports" / "results" / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


@st.cache_data(show_spinner=False)
def load_config_only():
    """Load configuration without touching model artifacts.

    The cold-start screen needs the config before any model exists.
    """
    from src.data.loader import load_config

    return load_config(str(project_root / "config" / "config.yaml"))


def _package_version(name: str) -> str:
    """Return an installed package's version, or ``"not installed"``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


# ============================================================================
# Session state
# ============================================================================

#: Page label -> (module path, extra loader arguments). The sidebar renders these
#: in order and the router dispatches on the label.
PAGES = {
    "Asset Health": "app.pages.executive_overview",
    "Risk Assessment": "app.pages.risk_predictor",
    "Data Explorer": "app.pages.data_explorer",
    "Model Diagnostics": "app.pages.explainable_ai",
    "Defect Inspection": "app.pages.ceiling_inspection",
    "Model & Cost Analysis": "app.pages.model_comparison",
    "Batch Analysis": "app.pages.upload_predict",
    "Alerts": "app.pages.monitoring_alerts",
}


def init_session_state() -> None:
    """Seed the session keys the pages rely on."""
    if "prediction_history" not in st.session_state:
        from src.risk.scoring import PredictionHistory

        st.session_state.prediction_history = PredictionHistory(max_entries=500)

    if "current_page" not in st.session_state:
        st.session_state.current_page = next(iter(PAGES))

    # Show the guide once per browser session, not on every rerun.
    if "guide_seen" not in st.session_state:
        st.session_state.guide_seen = True
        st.session_state.show_guide = True


init_session_state()


# ============================================================================
# Cold start
# ============================================================================

def render_cold_start() -> None:
    """Explain the missing artifacts and offer a one-click baseline training run.

    Without this, a fresh clone lands on a raw stack trace. The button trains the
    same preprocessing pipeline with fixed hyperparameters (~20-40s) instead of
    making the user wait out the full randomised search.
    """
    st.markdown(
        '<div class="masthead"><div><h1>Model artifacts not found</h1>'
        '<div class="masthead-sub">The dashboard needs a trained model before it '
        "can score telemetry. Train a fast baseline now, or run the full pipeline "
        "for the tuned, calibrated model set.</div></div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Option A — baseline, about 30 seconds</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Trains Random Forest and HistGradientBoosting with fixed "
            "hyperparameters, calibrates the better of the two, and writes the "
            "same artifacts the full pipeline produces. Good enough to explore "
            "every page."
        )

        if st.button("Train baseline model", type="primary", use_container_width=True):
            status = st.status("Starting...", expanded=True)
            try:
                from src.models.baseline import train_baseline

                config = load_config_only()
                summary = train_baseline(
                    config,
                    project_root,
                    progress=lambda message: status.write(message),
                )
                status.update(
                    label=f"Done in {summary['elapsed_seconds']}s — "
                    f"best model: {summary['best_model_name']}",
                    state="complete",
                )
                load_artifacts.clear()
                st.rerun()
            except Exception as exc:  # surfaced to the user, not swallowed
                status.update(label="Training failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Option B — full pipeline, 10 to 20 minutes</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Randomised hyperparameter search across every enabled model family, "
            "probability calibration, SHAP analysis, ablation study and the full "
            "figure set. Run it in a terminal:"
        )
        st.code("python scripts/train_pipeline.py --mode full", language="bash")
        st.markdown("Or the quicker searched profile:")
        st.code("python scripts/train_pipeline.py --mode fast", language="bash")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar(artifacts_ready: bool) -> None:
    """Render brand, navigation and the live system panel."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 2px 2px 12px 2px;">
                <div style="font-size: 0.92rem; font-weight: 600; color: var(--ink);
                            letter-spacing: -0.01em;">
                    Predictive Maintenance
                </div>
                <div style="font-size: 0.66rem; color: var(--ink-muted); font-weight: 500;
                            letter-spacing: 0.09em; text-transform: uppercase; margin-top: 2px;">
                    Ceiling Systems Plant
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bound to session state by key, and seeded with the current page's
        # index, so the selection survives a rerun and can be driven
        # programmatically (the guide's links, and the headless tests).
        page_names = list(PAGES)
        current = st.session_state.get("current_page", page_names[0])
        st.radio(
            "Navigation",
            page_names,
            index=page_names.index(current) if current in page_names else 0,
            label_visibility="collapsed",
            key="nav_choice",
        )
        st.session_state.current_page = st.session_state.nav_choice

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- live system panel, read from the artifacts rather than hardcoded
        model_name = "—"
        profile = "—"
        if artifacts_ready:
            try:
                artifacts, _ = load_artifacts()
                model_name = artifacts.get("best_model_name", "—")
                profile = artifacts.get("training_profile", "full pipeline")
            except Exception:
                artifacts_ready = False

        if artifacts_ready:
            engine_pill = render_pill("Online", "good")
        else:
            engine_pill = render_pill("No model", "critical")

        source_pill = (
            render_pill("Uploaded batch", "accent")
            if "uploaded_dataset" in st.session_state
            else render_pill("Plant telemetry", "neutral")
        )

        st.markdown(
            '<div style="font-size: 0.66rem; font-weight: 600; color: var(--ink-muted); '
            'text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">'
            "System</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            render_kv_rows(
                {
                    "Engine": engine_pill,
                    "Data source": source_pill,
                    "Model": f'<span style="font-size:0.74rem;">{model_name}</span>',
                    "Profile": f'<span style="font-size:0.74rem;">{profile}</span>',
                    "SHAP": f'<span style="font-size:0.74rem;">{_package_version("shap")}</span>',
                }
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.button("How to use this platform", use_container_width=True):
            st.session_state.show_guide = True

        with st.expander("Connect a database"):
            st.caption("Stream telemetry from an existing plant historian or warehouse.")
            db_type = st.selectbox(
                "Engine",
                ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Snowflake", "Oracle"],
                key="sidebar_db_type",
            )
            conn_str = st.text_input(
                "Connection URI",
                placeholder="postgresql://user:pass@host:5432/plant",
                key="sidebar_conn_str",
            )
            query_str = st.text_input(
                "Table or query",
                value="SELECT * FROM telemetry LIMIT 1000",
                key="sidebar_query_str",
            )
            if st.button("Connect and sync", key="sidebar_db_btn", use_container_width=True):
                if not conn_str.strip():
                    st.warning("Enter a connection URI first.")
                else:
                    try:
                        from src.data.loader import load_from_database

                        df_db = load_from_database(db_type, conn_str, query_str)
                        st.session_state.uploaded_dataset = df_db
                        st.success(f"Loaded {len(df_db):,} rows from {db_type}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")

        st.markdown(
            '<div style="margin-top: 18px; font-size: 0.68rem; color: var(--ink-muted); '
            "border: 1px solid var(--border); border-radius: 4px; padding: 9px 11px; "
            'line-height: 1.5;">'
            "Predictions come from a calibrated statistical model. Confirm the "
            "physical condition of equipment before acting on a recommendation."
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================================
# Guide dialog
# ============================================================================

@st.dialog("How to use this platform", width="large")
def render_guide_dialog() -> None:
    """Short orientation covering the workflow, not a feature list."""
    st.markdown(
        """
**The normal workflow is: look at the fleet, then drill into one asset, then act.**

| Page | What it answers |
|:--|:--|
| **Asset Health** | How is the fleet doing right now, and which assets need attention? |
| **Risk Assessment** | What is this specific machine's failure probability, and why? |
| **Data Explorer** | What does the underlying sensor record actually look like? |
| **Model Diagnostics** | Which sensor readings drive the model's decisions overall? |
| **Defect Inspection** | Is the product coming off the line within tolerance? |
| **Model & Cost Analysis** | Which model is best, and what is it worth in currency? |
| **Batch Analysis** | Score a whole CSV or database table at once. |
| **Alerts** | What has been flagged this session, and what was recommended? |

**Reading a risk score.** The score is a 0-100 transform of the calibrated
failure probability. Below 30 is routine, 30-60 warrants a scheduled check,
60-80 needs intervention this shift, and above 80 means stop and inspect.

**Reading a SHAP value.** A positive value pushed the prediction toward failure,
a negative one away from it. The magnitude is how hard it pushed. Every single
prediction carries its own breakdown, so a recommendation always has a stated
reason behind it.
        """
    )
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.show_guide = False
        st.rerun()


# ============================================================================
# Router
# ============================================================================

def main() -> None:
    """Route to the selected page, handling the cold start first."""
    from src.models.baseline import artifacts_present

    try:
        artifacts_ready = artifacts_present(load_config_only(), project_root)
    except Exception as exc:
        st.error(f"Configuration could not be loaded: {exc}")
        return

    render_sidebar(artifacts_ready)

    if not artifacts_ready:
        render_cold_start()
        return

    if st.session_state.get("show_guide", False):
        render_guide_dialog()

    page = st.session_state.current_page

    try:
        if page == "Asset Health":
            from app.pages.executive_overview import render_page

            render_page(project_root, load_artifacts, load_raw_dataset, load_results_json)

        elif page == "Risk Assessment":
            from app.pages.risk_predictor import render_page

            render_page(project_root, load_artifacts, load_raw_dataset)

        elif page == "Data Explorer":
            from app.pages.data_explorer import render_page

            render_page(project_root, load_raw_dataset)

        elif page == "Model Diagnostics":
            from app.pages.explainable_ai import render_page

            render_page(project_root, load_artifacts, load_raw_dataset, load_results_json)

        elif page == "Defect Inspection":
            from app.pages.ceiling_inspection import render_page

            render_page(project_root, load_artifacts, load_raw_dataset)

        elif page == "Model & Cost Analysis":
            from app.pages.model_comparison import render_page

            render_page(project_root, load_artifacts, load_results_json, load_raw_dataset)

        elif page == "Batch Analysis":
            from app.pages.upload_predict import render_page

            render_page(project_root, load_artifacts, load_raw_dataset)

        elif page == "Alerts":
            from app.pages.monitoring_alerts import render_page

            render_page(project_root)

    except FileNotFoundError as exc:
        st.markdown(
            render_notice(
                "Artifacts missing",
                f"A required file could not be found: <code>{exc}</code>. "
                "Re-run <code>python scripts/train_pipeline.py</code>.",
                "critical",
            ),
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.markdown(
            render_notice(
                f"{type(exc).__name__} on the {page} page",
                str(exc),
                "critical",
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Technical detail"):
            import traceback

            st.code(traceback.format_exc(), language="python")

    st.markdown(
        render_footer(
            "Predictive maintenance and inspection platform",
            f"Session started {datetime.now().strftime('%d %b %Y, %H:%M')}",
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
