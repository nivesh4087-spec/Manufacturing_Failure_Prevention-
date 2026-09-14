"""
End-to-End Training Pipeline
==============================
Orchestrates the complete ML pipeline from data loading to model artifacts.

Pipeline:
1. Load & validate dataset
2. Preprocess & engineer features
3. Train all models with HPO
4. Evaluate & compare models
5. Select best model
6. Calibrate probabilities
7. Generate SHAP explanations
8. Save all artifacts
9. Generate evaluation plots

Usage:
    python scripts/train_pipeline.py --mode fast      # minutes, for iterating
    python scripts/train_pipeline.py --mode full      # the full search, for a release
    python scripts/train_pipeline.py --mode baseline  # seconds, fixed hyperparameters
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data.loader import load_config, load_dataset, get_dataset_summary
from src.data.validator import run_full_validation
from src.preprocessing.pipeline import run_preprocessing_pipeline
from src.models.trainer import (
    train_all_models,
    evaluate_and_select_best,
    calibrate_model,
    save_model_artifacts,
)
from src.evaluation.evaluator import (
    evaluate_all_models,
    generate_all_plots,
    save_results,
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
)
from src.features.engineer import generate_feature_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainPipeline")


#: How much search each profile buys. The full profile is the published
#: configuration; fast trades breadth for turnaround so a code change can be
#: checked without waiting out a thousand model fits.
MODE_PRESETS = {
    "fast": {"n_iter": 6, "cv_folds": 3, "calibration_folds": 3},
    "full": {},  # use the configuration as written
}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate and persist the failure-prediction models.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "fast", "full"],
        default="full",
        help=(
            "baseline: fixed hyperparameters, about 30 seconds. "
            "fast: a small randomised search, a few minutes. "
            "full: the configured search (default)."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation. Useful in CI, where nobody looks at them.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis, which dominates runtime on large models.",
    )
    return parser.parse_args(argv)


def apply_mode(config: dict, mode: str) -> dict:
    """Fold a mode preset into the loaded configuration.

    Args:
        config: The configuration as loaded from YAML.
        mode: One of the keys in :data:`MODE_PRESETS`.

    Returns:
        The same dict, mutated, so nothing downstream needs mode awareness.
    """
    preset = MODE_PRESETS.get(mode, {})
    if not preset:
        return config

    if "n_iter" in preset:
        config["hpo"]["n_iter"] = preset["n_iter"]
    if "cv_folds" in preset:
        config["hpo"]["cv_folds"] = preset["cv_folds"]
    if "calibration_folds" in preset:
        config["calibration"]["cv_folds"] = preset["calibration_folds"]

    logger.info(
        "Mode %s: %d search iterations x %d folds",
        mode, config["hpo"]["n_iter"], config["hpo"]["cv_folds"],
    )
    return config


def run_baseline_profile() -> None:
    """Hand off to the short fixed-hyperparameter path."""
    from src.models.baseline import train_baseline

    config = load_config(str(project_root / "config" / "config.yaml"))
    summary = train_baseline(config, project_root, progress=logger.info)
    logger.info("Baseline complete in %.0fs. Best model: %s",
                summary["elapsed_seconds"], summary["best_model_name"])
    for r in summary["test_results"]:
        logger.info("  %-24s F1 %.4f  PR-AUC %.4f  ROC-AUC %.4f",
                    r["model"], r["f1"], r["pr_auc"], r["roc_auc"])


def main(argv=None):
    """Execute the complete training pipeline."""
    args = parse_args(argv)

    logger.info("=" * 70)
    logger.info(" PREDICTIVE MAINTENANCE - TRAINING PIPELINE (%s)", args.mode.upper())
    logger.info("=" * 70)

    # The baseline profile is a much shorter pipeline of its own; hand over to
    # it rather than threading conditionals through every step below.
    if args.mode == "baseline":
        run_baseline_profile()
        return

    # ========================================================================
    # 1. LOAD CONFIGURATION
    # ========================================================================
    logger.info("\n[STEP 1] Loading configuration...")
    config = load_config(str(project_root / "config" / "config.yaml"))
    config = apply_mode(config, args.mode)
    seed = config["project"]["random_seed"]
    np.random.seed(seed)

    # ========================================================================
    # 2. LOAD DATASET
    # ========================================================================
    logger.info("\n[STEP 2] Loading AI4I 2020 dataset...")
    df_raw = load_dataset(config)
    summary = get_dataset_summary(df_raw)
    logger.info("Dataset: %d rows × %d columns", summary["n_rows"], summary["n_columns"])

    # Save dataset summary
    results_dir = project_root / config["artifacts"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ========================================================================
    # 3. VALIDATE DATASET
    # ========================================================================
    logger.info("\n[STEP 3] Validating dataset...")
    validation = run_full_validation(df_raw, config)
    with open(results_dir / "validation_report.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    # ========================================================================
    # 4. EDA PLOTS
    # ========================================================================
    figures_dir = project_root / config["artifacts"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    target_col = config["data"]["target_column"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.skip_plots:
        logger.info("\n[STEP 4] Skipping EDA plots (--skip-plots).")
    else:
        logger.info("\n[STEP 4] Generating EDA plots...")
        plot_class_distribution(df_raw[target_col], "AI4I 2020 Class Distribution", figures_dir)
        plt.close("all")

    # ========================================================================
    # 5. PREPROCESSING PIPELINE
    # ========================================================================
    logger.info("\n[STEP 5] Running preprocessing pipeline...")
    processed = run_preprocessing_pipeline(df_raw, config)

    X_train = processed["X_train"]
    X_val = processed["X_val"]
    X_test = processed["X_test"]
    y_train = processed["y_train"]
    y_val = processed["y_val"]
    y_test = processed["y_test"]

    logger.info("Features: %s", processed["feature_names"])
    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # Save feature engineering report
    feat_report = generate_feature_report(
        processed["X_train_unscaled"], processed["feature_stats"]
    )
    with open(results_dir / "feature_engineering_report.json", "w") as f:
        json.dump(feat_report, f, indent=2, default=str)

    # Generate feature distribution plot using unscaled data
    df_for_eda = processed["X_train_unscaled"].copy()
    df_for_eda["machine_failure"] = processed["y_train_original"].values[:len(df_for_eda)] if len(processed["y_train_original"]) >= len(df_for_eda) else y_train.values[:len(df_for_eda)]
    if not args.skip_plots:
        plot_feature_distributions(df_for_eda, "machine_failure", figures_dir)
        plt.close("all")
        plot_correlation_heatmap(df_for_eda, figures_dir)
        plt.close("all")

    # ========================================================================
    # 6. MODEL TRAINING
    # ========================================================================
    logger.info("\n[STEP 6] Training models with hyperparameter optimization...")
    trained_models = train_all_models(X_train, y_train, config)

    # ========================================================================
    # 7. MODEL EVALUATION & SELECTION
    # ========================================================================
    logger.info("\n[STEP 7] Evaluating and selecting best model...")
    selection = evaluate_and_select_best(trained_models, X_val, y_val, config)

    best_model = selection["best_model"]
    best_name = selection["best_model_name"]
    all_models = selection["all_trained_models"]

    logger.info("BEST MODEL: %s", best_name)

    # ========================================================================
    # 8. TEST SET EVALUATION
    # ========================================================================
    logger.info("\n[STEP 8] Final evaluation on test set...")
    test_results = evaluate_all_models(all_models, X_test, y_test)
    save_results(test_results, config, project_root, "model_comparison.json")

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON — TEST SET RESULTS")
    logger.info("=" * 80)
    header = f"{'Model':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'PR-AUC':>8} {'ROC-AUC':>9} {'Brier':>8}"
    logger.info(header)
    logger.info("-" * 80)
    for r in test_results:
        line = (f"{r['model']:<25} {r['precision']:>10.4f} {r['recall']:>8.4f} "
                f"{r['f1']:>8.4f} {r['pr_auc']:>8.4f} {r['roc_auc']:>9.4f} "
                f"{r['brier_score']:>8.4f}")
        logger.info(line)
    logger.info("=" * 80)

    # ========================================================================
    # 9. PROBABILITY CALIBRATION
    # ========================================================================
    logger.info("\n[STEP 9] Calibrating best model probabilities...")

    # Use the original (un-resampled) training data for calibration
    X_train_orig = processed["X_train_original"]
    y_train_orig = processed["y_train_original"]

    # Scale the original training data
    from src.preprocessing.pipeline import apply_scaler
    X_train_orig_scaled = apply_scaler(
        X_train_orig, processed["scaler"], config,
        feature_cols=processed["numerical_cols"]
    )

    calibrated_model = calibrate_model(
        best_model, X_train_orig_scaled, y_train_orig,
        config, model_name=best_name
    )

    # Evaluate calibrated model
    from sklearn.metrics import f1_score, brier_score_loss
    y_prob_uncal = best_model.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

    brier_uncal = brier_score_loss(y_test, y_prob_uncal)
    brier_cal = brier_score_loss(y_test, y_prob_cal)

    logger.info("Brier Score — Uncalibrated: %.4f, Calibrated: %.4f", brier_uncal, brier_cal)

    calibration_results = {
        "brier_uncalibrated": round(brier_uncal, 4),
        "brier_calibrated": round(brier_cal, 4),
        "improvement": round(brier_uncal - brier_cal, 4),
        "method": config["calibration"]["method"],
    }
    with open(results_dir / "calibration_results.json", "w") as f:
        json.dump(calibration_results, f, indent=2)

    # ========================================================================
    # 10. GENERATE EVALUATION PLOTS
    # ========================================================================
    if args.skip_plots:
        logger.info("\n[STEP 10] Skipping evaluation plots (--skip-plots).")
    else:
        logger.info("\n[STEP 10] Generating evaluation plots...")
        # Include both the calibrated and uncalibrated best model.
        plot_models = dict(all_models)
        plot_models[f"{best_name} (Calibrated)"] = calibrated_model
        generate_all_plots(
            plot_models, X_test, y_test, test_results, config, project_root
        )
        plt.close("all")

    # ========================================================================
    # 11. SHAP ANALYSIS
    # ========================================================================
    logger.info("\n[STEP 11] Computing SHAP explanations...")

    # Determine model type for SHAP
    model_type = "tree"
    for result in trained_models:
        if result["name"] == best_name:
            model_type = result["model_type"]
            break

    try:
        if args.skip_shap:
            raise RuntimeError("skipped by --skip-shap")

        from src.explainability.shap_engine import SHAPEngine

        shap_engine = SHAPEngine(
            model=best_model,  # Use uncalibrated for SHAP (tree structure)
            X_background=X_train,
            config=config,
            model_type=model_type,
            feature_names=processed["feature_names"],
        )

        # Global SHAP analysis
        shap_engine.generate_all_plots(X_test, figures_dir)
        plt.close("all")

        # Get feature importance
        importance_df = shap_engine.get_global_importance()
        importance_df.to_csv(results_dir / "shap_feature_importance.csv", index=False)

        logger.info("\nSHAP Feature Importance:")
        logger.info(importance_df.head(10).to_string())

        # Save SHAP engine reference data for dashboard
        shap_data = {
            "background_samples": len(shap_engine.X_background),
            "model_type": model_type,
            "feature_importance": importance_df.to_dict(orient="records"),
        }
        with open(results_dir / "shap_analysis.json", "w") as f:
            json.dump(shap_data, f, indent=2, default=str)

    except Exception as e:
        logger.error("SHAP analysis failed: %s", e)
        logger.info("Dashboard will still work but without pre-computed SHAP plots.")

    # ========================================================================
    # 12. ABLATION STUDY
    # ========================================================================
    logger.info("\n[STEP 12] Running ablation studies...")

    ablation_results = {}

    # Ablation 1: With vs without feature engineering
    try:
        from src.preprocessing.pipeline import (
            drop_leakage_columns, drop_id_columns, rename_columns,
            encode_type_column, split_data, fit_scaler, apply_scaler
        )
        from src.features.engineer import get_all_feature_names

        df_no_feat = drop_leakage_columns(df_raw.copy(), config)
        df_no_feat = drop_id_columns(df_no_feat, config)
        df_no_feat = rename_columns(df_no_feat, config)
        df_no_feat = encode_type_column(df_no_feat, config)

        X_tr_nf, X_v_nf, X_te_nf, y_tr_nf, y_v_nf, y_te_nf = split_data(df_no_feat, config)

        # No feature engineering — just base features
        base_feats = get_all_feature_names(include_engineered=False)
        X_tr_nf = X_tr_nf[[c for c in base_feats if c in X_tr_nf.columns]]
        X_te_nf = X_te_nf[[c for c in base_feats if c in X_te_nf.columns]]

        num_cols_nf = X_tr_nf.select_dtypes(include=[np.number]).columns.tolist()
        scaler_nf = fit_scaler(X_tr_nf, config, feature_cols=num_cols_nf)
        X_tr_nf = apply_scaler(X_tr_nf, scaler_nf, config, feature_cols=num_cols_nf)
        X_te_nf = apply_scaler(X_te_nf, scaler_nf, config, feature_cols=num_cols_nf)

        from sklearn.ensemble import RandomForestClassifier
        rf_nf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1
        )
        rf_nf.fit(X_tr_nf, y_tr_nf)

        from sklearn.metrics import f1_score as f1_fn, average_precision_score as ap_fn
        y_pred_nf = rf_nf.predict(X_te_nf)
        y_prob_nf = rf_nf.predict_proba(X_te_nf)[:, 1]

        ablation_results["without_feature_engineering"] = {
            "f1": round(float(f1_fn(y_te_nf, y_pred_nf)), 4),
            "pr_auc": round(float(ap_fn(y_te_nf, y_prob_nf)), 4),
        }

        # With feature engineering (use test results of RF)
        rf_test = next((r for r in test_results if "Random Forest" in r["model"]), None)
        if rf_test:
            ablation_results["with_feature_engineering"] = {
                "f1": rf_test["f1"],
                "pr_auc": rf_test["pr_auc"],
            }

        logger.info("Ablation — Feature Engineering: %s", ablation_results)

    except Exception as e:
        logger.warning("Feature engineering ablation failed: %s", e)

    # Save ablation results
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)

    # ========================================================================
    # 13. SAVE ALL ARTIFACTS
    # ========================================================================
    logger.info("\n[STEP 13] Saving model artifacts...")

    artifacts = {
        "training_profile": args.mode,
        "best_model": best_model,
        "best_model_calibrated": calibrated_model,
        "best_model_name": best_name,
        "all_models": all_models,
        "scaler": processed["scaler"],
        "feature_stats": processed["feature_stats"],
        "feature_names": processed["feature_names"],
        "numerical_cols": processed["numerical_cols"],
        "config": config,
        "test_results": test_results,
        "selection_results": {
            "best_model_name": best_name,
            "comparison_table": selection["comparison_table"],
        },
    }

    save_model_artifacts(artifacts, config, project_root)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info(" TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Best Model: %s", best_name)
    logger.info("Best Model F1 (test): %.4f",
                next((r["f1"] for r in test_results if r["model"] == best_name), 0))
    logger.info("Best Model PR-AUC (test): %.4f",
                next((r["pr_auc"] for r in test_results if r["model"] == best_name), 0))
    logger.info("Calibration improvement (Brier): %.4f", calibration_results["improvement"])
    logger.info("Artifacts saved to: %s", project_root / config["artifacts"]["model_dir"])
    logger.info("Figures saved to: %s", figures_dir)
    logger.info("Results saved to: %s", results_dir)
    logger.info("\nRun the dashboard: streamlit run app/main.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
