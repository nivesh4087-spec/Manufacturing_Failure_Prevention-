"""
Model Trainer Module
====================
Trains, tunes, calibrates, and selects ML models for failure prediction.

Models:
1. Logistic Regression — interpretable baseline
2. Random Forest — ensemble with feature importance
3. XGBoost — state-of-the-art gradient boosting
4. HistGradientBoosting — fast sklearn-native boosting

Training includes:
- Hyperparameter optimization via RandomizedSearchCV
- Cross-validation with stratified folds
- Probability calibration (isotonic/sigmoid)
- Model selection based on F1 + PR-AUC
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    brier_score_loss,
)

# XGBoost is an optional heavy dependency. Import it lazily so a minimal install
# (or an environment where the wheel failed to build) can still train, evaluate
# and serve the scikit-learn models instead of failing at import time.
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install profile
    xgb = None
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Model Registry
# ============================================================================

def get_model_registry(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the model registry with configured models.

    Args:
        config: Project configuration.

    Returns:
        Dictionary mapping model names to their estimators and param grids.
    """
    models = {}

    if config["models"]["logistic_regression"]["enabled"]:
        models["Logistic Regression"] = {
            "estimator": LogisticRegression(
                **config["models"]["logistic_regression"]["params"],
                random_state=config["project"]["random_seed"],
            ),
            "param_grid": {
                f"C": config["models"]["logistic_regression"]["hyperparam_grid"]["C"],
                f"penalty": config["models"]["logistic_regression"]["hyperparam_grid"]["penalty"],
                f"class_weight": config["models"]["logistic_regression"]["hyperparam_grid"]["class_weight"],
            },
            "type": "linear",
        }

    if config["models"]["random_forest"]["enabled"]:
        models["Random Forest"] = {
            "estimator": RandomForestClassifier(
                **config["models"]["random_forest"]["params"],
                random_state=config["project"]["random_seed"],
            ),
            "param_grid": config["models"]["random_forest"]["hyperparam_grid"],
            "type": "tree",
        }

    if config["models"]["xgboost"]["enabled"] and not XGBOOST_AVAILABLE:
        logger.warning(
            "XGBoost is enabled in config but not installed — skipping it. "
            "Run `pip install xgboost` to include it in the comparison."
        )

    if config["models"]["xgboost"]["enabled"] and XGBOOST_AVAILABLE:
        models["XGBoost"] = {
            "estimator": xgb.XGBClassifier(
                **config["models"]["xgboost"]["params"],
                random_state=config["project"]["random_seed"],
            ),
            "param_grid": config["models"]["xgboost"]["hyperparam_grid"],
            "type": "tree",
        }

    if config["models"]["hist_gradient_boosting"]["enabled"]:
        models["HistGradientBoosting"] = {
            "estimator": HistGradientBoostingClassifier(
                **config["models"]["hist_gradient_boosting"]["params"],
                random_state=config["project"]["random_seed"],
            ),
            "param_grid": config["models"]["hist_gradient_boosting"]["hyperparam_grid"],
            "type": "tree",
        }

    return models


# ============================================================================
# Hyperparameter Optimization
# ============================================================================

def train_single_model(
    name: str,
    model_info: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train a single model with hyperparameter optimization.

    Uses RandomizedSearchCV with StratifiedKFold to tune hyperparameters
    while preserving class ratios in each fold.

    Args:
        name: Model name.
        model_info: Model estimator and param grid.
        X_train: Training features.
        y_train: Training labels.
        config: Project configuration.

    Returns:
        Training result dictionary.
    """
    logger.info("-" * 50)
    logger.info("Training: %s", name)
    logger.info("-" * 50)

    hpo_config = config["hpo"]
    seed = config["project"]["random_seed"]

    estimator = model_info["estimator"]
    param_grid = model_info["param_grid"]

    # Stratified cross-validation
    cv = StratifiedKFold(
        n_splits=hpo_config["cv_folds"],
        shuffle=True,
        random_state=seed,
    )

    # Determine n_iter (limit to total combinations if grid is small)
    from itertools import product as iter_product
    try:
        total_combinations = 1
        for v in param_grid.values():
            if isinstance(v, list):
                total_combinations *= len(v)
        n_iter = min(hpo_config["n_iter"], total_combinations)
    except Exception:
        n_iter = hpo_config["n_iter"]

    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=hpo_config["scoring"],
        n_jobs=hpo_config["n_jobs"],
        random_state=seed,
        verbose=0,
        refit=True,
        return_train_score=True,
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV %s: %.4f", hpo_config["scoring"], search.best_score_)
    logger.info("Training time: %.2f seconds", train_time)

    return {
        "name": name,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_results": search.cv_results_,
        "train_time_seconds": round(train_time, 2),
        "model_type": model_info["type"],
    }


# ============================================================================
# Train All Models
# ============================================================================

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Train all enabled models with HPO.

    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Project configuration.

    Returns:
        List of training result dictionaries.
    """
    registry = get_model_registry(config)
    results = []

    logger.info("=" * 60)
    logger.info("Training %d models with hyperparameter optimization", len(registry))
    logger.info("=" * 60)

    for name, model_info in registry.items():
        try:
            result = train_single_model(name, model_info, X_train, y_train, config)
            results.append(result)
        except Exception as e:
            logger.error("Failed to train %s: %s", name, e)
            continue

    logger.info("Training complete. %d/%d models trained successfully.",
                len(results), len(registry))
    return results


# ============================================================================
# Probability Calibration
# ============================================================================

def calibrate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    model_name: str = "model",
) -> Any:
    """Calibrate model probabilities using CalibratedClassifierCV.

    Probability calibration improves the reliability of probability estimates.
    This is critical for the risk scoring system.

    Args:
        model: Trained model (best estimator from HPO).
        X_train: Training features.
        y_train: Training labels.
        config: Project configuration.
        model_name: Name for logging.

    Returns:
        Calibrated model.
    """
    method = config["calibration"]["method"]
    cv_folds = config["calibration"]["cv_folds"]

    logger.info("Calibrating %s with %s method (cv=%d)", model_name, method, cv_folds)

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv_folds,
    )
    calibrated.fit(X_train, y_train)

    logger.info("Calibration complete for %s.", model_name)
    return calibrated


# ============================================================================
# Model Selection
# ============================================================================

def evaluate_and_select_best(
    trained_models: List[Dict[str, Any]],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate all trained models and select the best one.

    Selection criteria:
    - Primary: F1-score (balances precision and recall for failure detection)
    - Secondary: PR-AUC (better than ROC-AUC for imbalanced data)

    Args:
        trained_models: List of training results.
        X_val: Validation features.
        y_val: Validation labels.
        config: Project configuration.

    Returns:
        Dictionary with best model info and comparison table.
    """
    primary_metric = config["selection"]["primary_metric"]
    results_table = []

    for result in trained_models:
        model = result["best_estimator"]
        name = result["name"]

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "model": name,
            "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, y_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_val, y_prob)), 4),
            "brier_score": round(float(brier_score_loss(y_val, y_prob)), 4),
            "best_params": result["best_params"],
            "best_cv_score": result["best_cv_score"],
            "train_time": result["train_time_seconds"],
            "model_type": result["model_type"],
        }
        results_table.append(metrics)

        logger.info(
            "%s — F1: %.4f, Precision: %.4f, Recall: %.4f, PR-AUC: %.4f, ROC-AUC: %.4f",
            name, metrics["f1"], metrics["precision"], metrics["recall"],
            metrics["pr_auc"], metrics["roc_auc"]
        )

    # Sort by primary metric
    results_table.sort(key=lambda x: x[primary_metric], reverse=True)
    best_model_name = results_table[0]["model"]

    # Find the corresponding trained model
    best_model_result = next(r for r in trained_models if r["name"] == best_model_name)

    logger.info("=" * 60)
    logger.info("BEST MODEL: %s (F1: %.4f, PR-AUC: %.4f)",
                best_model_name,
                results_table[0]["f1"],
                results_table[0]["pr_auc"])
    logger.info("=" * 60)

    return {
        "best_model_name": best_model_name,
        "best_model": best_model_result["best_estimator"],
        "best_metrics": results_table[0],
        "comparison_table": results_table,
        "all_trained_models": {r["name"]: r["best_estimator"] for r in trained_models},
    }


# ============================================================================
# Save / Load Model Artifacts
# ============================================================================

def save_model_artifacts(
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    project_root: Path = None,
) -> Path:
    """Save all model artifacts to disk.

    Saves:
    - Best model (calibrated and uncalibrated)
    - All trained models
    - Preprocessing pipeline components
    - Feature metadata
    - Model metrics

    Args:
        artifacts: Dictionary of all artifacts to save.
        config: Project configuration.
        project_root: Project root directory.

    Returns:
        Path to saved artifacts directory.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    model_dir = project_root / config["artifacts"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save each artifact
    for name, obj in artifacts.items():
        filepath = model_dir / f"{name}.joblib"
        joblib.dump(obj, filepath)
        logger.info("Saved artifact: %s → %s", name, filepath)

    logger.info("All model artifacts saved to %s", model_dir)
    return model_dir


def load_model_artifacts(
    config: Dict[str, Any],
    project_root: Path = None,
) -> Dict[str, Any]:
    """Load all model artifacts from disk.

    Args:
        config: Project configuration.
        project_root: Project root directory.

    Returns:
        Dictionary of loaded artifacts.

    Raises:
        FileNotFoundError: If artifacts directory does not exist.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    model_dir = project_root / config["artifacts"]["model_dir"]

    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifacts directory not found: {model_dir}")

    artifacts = {}
    for filepath in model_dir.glob("*.joblib"):
        name = filepath.stem
        artifacts[name] = joblib.load(filepath)
        logger.info("Loaded artifact: %s", name)

    logger.info("Loaded %d artifacts from %s", len(artifacts), model_dir)
    return artifacts
