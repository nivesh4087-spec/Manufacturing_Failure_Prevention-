"""
Tests for the training pipeline's mode presets.

The presets decide how long a training run takes and how much memory it asks
for, so their behaviour is worth pinning down without paying for a real run.
"""

import copy
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    """Import train_pipeline.py by path, since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "train_pipeline", PROJECT_ROOT / "scripts" / "train_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    """A configuration shaped like the real one, but independent of it."""
    return {
        "hpo": {"n_iter": 50, "cv_folds": 5, "n_jobs": -1},
        "calibration": {"cv_folds": 5, "method": "isotonic"},
        "models": {
            "random_forest": {
                "params": {"n_jobs": -1},
                "hyperparam_grid": {"n_estimators": [100, 200, 300, 500],
                                    "max_depth": [5, 10, None]},
            },
            "hist_gradient_boosting": {
                "params": {},
                "hyperparam_grid": {"max_iter": [100, 200, 300, 500]},
            },
            "logistic_regression": {
                "params": {"max_iter": 1000},
                "hyperparam_grid": {"C": [0.01, 1.0]},
            },
        },
    }


class TestParseArgs:
    """Command-line surface."""

    def test_defaults_to_full(self, pipeline):
        assert pipeline.parse_args([]).mode == "full"

    @pytest.mark.parametrize("mode", ["baseline", "fast", "full"])
    def test_accepts_each_mode(self, pipeline, mode):
        assert pipeline.parse_args(["--mode", mode]).mode == mode

    def test_rejects_an_unknown_mode(self, pipeline):
        with pytest.raises(SystemExit):
            pipeline.parse_args(["--mode", "turbo"])

    def test_skip_flags_default_off(self, pipeline):
        args = pipeline.parse_args([])
        assert not args.skip_plots
        assert not args.skip_shap

    def test_skip_flags_can_be_set(self, pipeline):
        args = pipeline.parse_args(["--skip-plots", "--skip-shap"])
        assert args.skip_plots and args.skip_shap


class TestFullMode:
    """Full mode must not touch the configured search."""

    def test_leaves_configuration_untouched(self, pipeline, config):
        before = copy.deepcopy(config)
        assert pipeline.apply_mode(config, "full") == before


class TestFastMode:
    """Fast mode must actually be fast."""

    @pytest.fixture
    def fast(self, pipeline, config):
        return pipeline.apply_mode(config, "fast")

    def test_reduces_search_iterations(self, fast):
        assert fast["hpo"]["n_iter"] == 6

    def test_reduces_cross_validation_folds(self, fast):
        assert fast["hpo"]["cv_folds"] == 3

    def test_reduces_calibration_folds(self, fast):
        assert fast["calibration"]["cv_folds"] == 3

    def test_caps_forest_size(self, fast):
        # A single 500-tree forest dominates the runtime of the whole profile.
        assert max(fast["models"]["random_forest"]["hyperparam_grid"]["n_estimators"]) <= 200

    def test_caps_boosting_iterations(self, fast):
        assert max(fast["models"]["hist_gradient_boosting"]["hyperparam_grid"]["max_iter"]) <= 200

    def test_limits_parallel_workers(self, fast):
        # n_jobs=-1 spawns one process per core, each loading its own copy of
        # the scientific stack — several gigabytes before any fitting starts.
        assert fast["hpo"]["n_jobs"] == 2

    def test_limits_workers_on_the_estimators_too(self, fast):
        assert fast["models"]["random_forest"]["params"]["n_jobs"] == 2

    def test_leaves_unrelated_grids_alone(self, fast):
        assert fast["models"]["logistic_regression"]["hyperparam_grid"]["C"] == [0.01, 1.0]

    def test_leaves_non_capped_parameters_alone(self, fast):
        assert fast["models"]["random_forest"]["hyperparam_grid"]["max_depth"] == [5, 10, None]

    def test_never_empties_a_grid(self, pipeline):
        # If every configured value exceeds the cap, fall back to the cap
        # itself rather than handing the search an empty list.
        config = {
            "hpo": {"n_iter": 50, "cv_folds": 5, "n_jobs": -1},
            "calibration": {"cv_folds": 5},
            "models": {"rf": {"params": {}, "hyperparam_grid": {"n_estimators": [800, 1000]}}},
        }
        out = pipeline.apply_mode(config, "fast")
        assert out["models"]["rf"]["hyperparam_grid"]["n_estimators"] == [200]


class TestPresetShape:
    """Guard rails on the preset table itself."""

    def test_full_preset_is_empty(self, pipeline):
        assert pipeline.MODE_PRESETS["full"] == {}

    def test_fast_preset_is_cheaper_than_the_default_config(self, pipeline):
        from src.data.loader import load_config

        real = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
        assert pipeline.MODE_PRESETS["fast"]["n_iter"] < real["hpo"]["n_iter"]
        assert pipeline.MODE_PRESETS["fast"]["cv_folds"] < real["hpo"]["cv_folds"]
