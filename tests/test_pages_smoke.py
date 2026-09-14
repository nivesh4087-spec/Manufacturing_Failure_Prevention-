"""
Page smoke tests.

Sprint 1 shipped a page carrying a syntax error that made it impossible to open,
because "done" meant "written" rather than "run". These tests close that gap: the
whole app is executed headlessly and every page is rendered, so any page that
cannot open fails the build.

They need trained artifacts. Where none exist the module skips rather than
failing, so a fresh clone's first test run is not a wall of red.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP = PROJECT_ROOT / "app" / "main.py"

pytest.importorskip("streamlit.testing.v1", reason="Streamlit testing API unavailable")

from streamlit.testing.v1 import AppTest  # noqa: E402

#: Every page the router serves, by the label the sidebar uses.
PAGES = [
    "Asset Health",
    "Risk Assessment",
    "Data Explorer",
    "Model Diagnostics",
    "Defect Inspection",
    "Model & Cost Analysis",
    "Batch Analysis",
    "Alerts",
]


def _artifacts_available() -> bool:
    """Return whether a trained model exists to render against."""
    from src.data.loader import load_config
    from src.models.baseline import artifacts_present

    try:
        return artifacts_present(load_config(str(PROJECT_ROOT / "config" / "config.yaml")),
                                 PROJECT_ROOT)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _artifacts_available(),
    reason="No trained artifacts. Run `python scripts/train_pipeline.py --mode baseline`.",
)


def run_page(page: str) -> AppTest:
    """Render one page headlessly and return the resulting app state."""
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.session_state["current_page"] = page
    # The sidebar radio owns the current page, so drive it too — setting only
    # current_page is overwritten by the widget's own default on first render.
    at.session_state["nav_choice"] = page
    at.session_state["show_guide"] = False
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    """Every page opens. This is the test Sprint 1 did not have."""
    at = run_page(page)
    assert not at.exception, (
        f"{page} raised: "
        + "; ".join(str(e.value).splitlines()[0] for e in at.exception)
    )


@pytest.mark.parametrize("page", PAGES)
def test_page_reports_no_error_banner(page):
    """A page that renders but shows only an error banner is still broken."""
    at = run_page(page)
    assert not at.error, f"{page} surfaced: {[e.value[:160] for e in at.error]}"


@pytest.mark.parametrize("page", PAGES)
def test_page_produces_content(page):
    """Guards against a page that silently renders nothing at all."""
    at = run_page(page)
    assert len(at.markdown) > 3, f"{page} rendered almost nothing"


class TestNavigation:
    """The router must actually switch pages."""

    def test_pages_render_differently(self):
        # Every page previously rendered an identical element count because the
        # sidebar radio ignored the requested page. If that regresses, the smoke
        # tests above would pass while testing one page eight times.
        counts = {page: len(run_page(page).markdown) for page in PAGES}
        assert len(set(counts.values())) > 1, (
            f"Every page rendered the same element count ({counts}); "
            "the router is not switching pages."
        )

    def test_router_covers_every_advertised_page(self):
        import app.main as main

        assert set(main.PAGES) == set(PAGES)


class TestPredictionFlow:
    """The core workflow, end to end."""

    def test_scenario_load_survives_the_next_rerun(self):
        # The bug this pins: scenario buttons wrote to a local variable, so the
        # values vanished on the rerun that followed any other interaction.
        at = run_page("Risk Assessment")
        before = at.session_state["rp_tool_wear"]
        at.button(key="scenario_critical_risk").click().run()
        after = at.session_state["rp_tool_wear"]

        assert after != before, "Scenario did not load"

        assess = [b for b in at.button if "Assess" in str(b.label)]
        assert assess, "No assessment button found"
        assess[0].click().run()

        assert not at.exception
        assert at.session_state["rp_tool_wear"] == after, (
            "Scenario values were lost on rerun"
        )

    def test_assessment_produces_a_risk_result(self):
        at = run_page("Risk Assessment")
        assess = [b for b in at.button if "Assess" in str(b.label)]
        assess[0].click().run()

        rendered = " ".join(m.value for m in at.markdown)
        assert "Failure probability" in rendered
        assert "Risk score" in rendered

    def test_batch_scoring_produces_banded_results(self):
        at = run_page("Batch Analysis")
        at.button(key="batch_sample_btn").click().run()
        assert not at.exception

        score = [b for b in at.button if "Score this batch" in str(b.label)]
        assert score, "No scoring button after loading the sample"
        score[0].click().run()
        assert not at.exception

        results = at.session_state["batch_results"]
        assert len(results) > 0
        assert set(results.columns) >= {
            "failure_probability", "risk_score", "risk_category", "prediction"
        }
        assert results["failure_probability"].between(0, 1).all()
        assert results["risk_score"].between(0, 100).all()
