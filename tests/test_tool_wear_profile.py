"""
Tests for the tool-wear failure profile behind the Defect Inspection page.

The recommended replacement interval shown to users is derived here, so it must
come from the data rather than from an assumption baked into the page.
"""

import numpy as np
import pandas as pd
import pytest

from app.pages.ceiling_inspection import recommended_change_point, tool_wear_failure_profile

# The Streamlit cache decorator wraps the function; call the plain one in tests.
profile_fn = tool_wear_failure_profile.__wrapped__


def synthetic_dataset(
    n_per_bucket: int = 200,
    breakpoint_min: int = 200,
    low_rate: float = 0.02,
    high_rate: float = 0.25,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a dataset whose failure rate jumps at a known tool-wear point."""
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, 260, 20):
        rate = high_rate if start >= breakpoint_min else low_rate
        wear = rng.integers(start + 1, start + 21, size=n_per_bucket)
        failed = rng.random(n_per_bucket) < rate
        for w, f in zip(wear, failed):
            rows.append({"Tool wear [min]": int(w), "Machine failure": int(f),
                         "TWF": int(f and w >= breakpoint_min)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset()


@pytest.fixture(scope="module")
def profile(dataset):
    return profile_fn(dataset, "Machine failure")


class TestProfile:
    """Bucketing and rate computation."""

    def test_one_row_per_populated_bucket(self, profile):
        assert len(profile) == 13

    def test_asset_counts_sum_to_the_dataset(self, profile, dataset):
        assert int(profile["assets"].sum()) == len(dataset)

    def test_failure_counts_sum_to_the_dataset(self, profile, dataset):
        assert int(profile["failures"].sum()) == int(dataset["Machine failure"].sum())

    def test_rate_is_a_percentage(self, profile):
        assert profile["failure_rate"].between(0, 100).all()

    def test_midpoints_are_numeric_and_ascending(self, profile):
        midpoints = profile["wear_midpoint"]
        assert midpoints.dtype.kind == "f"
        assert midpoints.is_monotonic_increasing

    def test_twf_column_is_carried_when_present(self, profile):
        assert "twf" in profile.columns

    def test_twf_column_is_omitted_when_absent(self, dataset):
        without = profile_fn(dataset.drop(columns=["TWF"]), "Machine failure")
        assert "twf" not in without.columns

    def test_bin_width_is_configurable(self, dataset):
        wide = profile_fn(dataset, "Machine failure", bin_width=50)
        assert len(wide) < 13


class TestChangePoint:
    """Deriving the replacement interval."""

    def test_finds_the_planted_breakpoint(self, profile):
        assert recommended_change_point(profile)["threshold"] == 200

    def test_reports_the_elevated_rate_above_baseline(self, profile):
        finding = recommended_change_point(profile)
        assert finding["elevated_rate"] > finding["baseline_rate"] * 2

    def test_share_is_a_percentage(self, profile):
        assert 0 <= recommended_change_point(profile)["share"] <= 100

    def test_avoidable_never_exceeds_total(self, profile):
        finding = recommended_change_point(profile)
        assert finding["avoidable"] <= finding["total"]

    def test_attributes_tool_wear_failures_when_available(self, profile):
        assert recommended_change_point(profile)["kind"] == "tool-wear failures"

    def test_falls_back_to_all_failures_without_twf(self, dataset):
        without = profile_fn(dataset.drop(columns=["TWF"]), "Machine failure")
        assert recommended_change_point(without)["kind"] == "failures"

    def test_returns_empty_for_a_flat_profile(self):
        # No wear-related rise means there is no defensible change point, and
        # the page must not invent one.
        flat = synthetic_dataset(breakpoint_min=10_000, low_rate=0.05)
        assert recommended_change_point(profile_fn(flat, "Machine failure")) == {}

    def test_returns_empty_for_an_empty_profile(self):
        assert recommended_change_point(pd.DataFrame()) == {}


class TestAgainstRealDataset:
    """The figures the page actually shows for the bundled dataset."""

    @pytest.fixture(scope="class")
    @classmethod
    def real_finding(cls):
        from src.data.loader import load_config, load_dataset

        config = load_config("config/config.yaml")
        try:
            df = load_dataset(config)
        except Exception:
            pytest.skip("AI4I dataset not available locally")
        return recommended_change_point(profile_fn(df, config["data"]["target_column"]))

    def test_recommends_a_threshold_in_the_documented_range(self, real_finding):
        # The README and the project report both quote this interval; if the
        # data ever moves it, this test is what catches the stale claim.
        assert real_finding["threshold"] == 200

    def test_most_tool_wear_failures_are_avoidable(self, real_finding):
        assert real_finding["share"] > 90
        assert real_finding["avoidable"] == 44
        assert real_finding["total"] == 46
