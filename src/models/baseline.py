"""
Baseline Trainer
================
A fast, dependency-light training path used for cold starts and smoke tests.

``scripts/train_pipeline.py`` runs a randomised hyperparameter search across four
model families and takes 10-20 minutes. That is the right thing to run before a
demo, but it is the wrong thing to make someone sit through on their first
launch. This module trains a usable calibrated model from the same preprocessing
pipeline in roughly 20-40 seconds, using fixed sensible hyperparameters and only
scikit-learn estimators — so it also works on an install without XGBoost.

The artifacts written here are schema-identical to the full pipeline's, so the
dashboard cannot tell the difference apart from ``training_profile``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

logger = logging.getLogger(__name__)

#: Hand-tuned defaults that land close to a searched model on this dataset
#: without paying for the search. Kept small so a cold start stays responsive.
BASELINE_ESTIMATORS: Dict[str, Callable[[int], Any]] = {
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    ),
    "HistGradientBoosting": lambda seed: HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=seed,
    ),
}


def train_baseline(
    config: Dict[str, Any],
    project_root: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Train, calibrate and persist a baseline model set.

    Args:
        config: Project configuration dictionary.
        project_root: Repository root. Defaults to the package's grandparent.
        progress: Optional callback invoked with human-readable status strings,
            so a caller (the dashboard) can surface progress live.
        generate_plots: Write the evaluation figures too. On by default so the
            figure set on disk always describes the model on disk — a stale
            figure beside a fresh model is how a report ends up misleading.

    Returns:
        A summary dict with ``best_model_name``, ``test_results``,
        ``elapsed_seconds`` and the artifact directory.

    Raises:
        FileNotFoundError: If the dataset cannot be located or downloaded.
    """
    from src.data.loader import load_dataset
    from src.evaluation.evaluator import evaluate_all_models
    from src.models.trainer import save_model_artifacts
    from src.preprocessing.pipeline import apply_scaler, run_preprocessing_pipeline

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    def say(message: str) -> None:
        logger.info(message)
        if progress is not None:
            progress(message)

    started = time.perf_counter()
    seed = config["project"]["random_seed"]

    say("Loading dataset...")
    df_raw = load_dataset(config)

    say("Preprocessing and engineering features...")
    processed = run_preprocessing_pipeline(df_raw, config)

    X_train, y_train = processed["X_train"], processed["y_train"]
    X_test, y_test = processed["X_test"], processed["y_test"]

    trained: Dict[str, Any] = {}
    for name, build in BASELINE_ESTIMATORS.items():
        say(f"Training {name}...")
        estimator = build(seed)
        estimator.fit(X_train, y_train)
        trained[name] = estimator

    say("Scoring on the held-out test set...")
    test_results = evaluate_all_models(trained, X_test, y_test)

    primary = config["selection"]["primary_metric"]
    test_results.sort(key=lambda r: r[primary], reverse=True)
    best_name = test_results[0]["model"]
    best_model = trained[best_name]

    say(f"Calibrating {best_name} probabilities...")
    # Calibrate on the original, un-resampled training split so the calibration
    # map reflects the true class prior rather than the rebalanced one.
    X_calib = apply_scaler(
        processed["X_train_original"],
        processed["scaler"],
        config,
        feature_cols=processed["numerical_cols"],
    )
    calibrated = CalibratedClassifierCV(
        best_model,
        method=config["calibration"]["method"],
        cv=3,
    )
    calibrated.fit(X_calib, processed["y_train_original"])

    say("Saving artifacts...")
    artifacts = {
        "best_model": best_model,
        "best_model_calibrated": calibrated,
        "best_model_name": best_name,
        "all_models": trained,
        "scaler": processed["scaler"],
        "feature_stats": processed["feature_stats"],
        "feature_names": processed["feature_names"],
        "numerical_cols": processed["numerical_cols"],
        "config": config,
        "test_results": test_results,
        "training_profile": "baseline",
        "selection_results": {
            "best_model_name": best_name,
            "comparison_table": test_results,
        },
    }
    model_dir = save_model_artifacts(artifacts, config, project_root)

    if generate_plots:
        say("Generating evaluation figures...")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from src.evaluation.evaluator import (
                generate_all_plots,
                plot_class_distribution,
                plot_correlation_heatmap,
                plot_feature_distributions,
            )

            figures_dir = project_root / config["artifacts"]["figures_dir"]
            figures_dir.mkdir(parents=True, exist_ok=True)

            plot_models = dict(trained)
            plot_models[f"{best_name} (Calibrated)"] = calibrated
            generate_all_plots(plot_models, X_test, y_test, test_results, config, project_root)
            plot_class_distribution(
                df_raw[config["data"]["target_column"]], "Class distribution", figures_dir
            )
            plt.close("all")

            # Data-level figures: about the dataset rather than the model, but
            # regenerated together so nothing on disk is older than the rest.
            eda = processed["X_train_unscaled"].copy()
            eda["machine_failure"] = processed["y_train_original"].values[: len(eda)]
            plot_feature_distributions(eda, "machine_failure", figures_dir)
            plt.close("all")
            plot_correlation_heatmap(eda, figures_dir)
            plt.close("all")
        except Exception as exc:  # figures are useful, not essential
            logger.warning("Figure generation failed (%s); continuing.", exc)

    elapsed = time.perf_counter() - started
    say(f"Baseline ready in {elapsed:.0f}s — best model: {best_name}")

    return {
        "best_model_name": best_name,
        "test_results": test_results,
        "elapsed_seconds": round(elapsed, 1),
        "model_dir": str(model_dir),
    }


def artifacts_present(config: Dict[str, Any], project_root: Optional[Path] = None) -> bool:
    """Return whether a usable set of model artifacts exists on disk.

    Checks for the specific files the dashboard needs rather than just the
    directory, so a half-written or manually emptied ``models/`` reads as absent.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    model_dir = project_root / config["artifacts"]["model_dir"]
    if not model_dir.is_dir():
        return False

    required = {"best_model", "scaler", "feature_stats", "feature_names", "numerical_cols"}
    present = {p.stem for p in model_dir.glob("*.joblib")}
    return required.issubset(present)


__all__ = ["train_baseline", "artifacts_present", "BASELINE_ESTIMATORS"]
