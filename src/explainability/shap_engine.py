"""
SHAP Explainability Engine
===========================
Provides global and local explanations using SHAP (SHapley Additive exPlanations).

Global Explainability:
- Feature importance ranking (mean |SHAP|)
- Summary/beeswarm plots
- Bar plots

Local Explainability:
- Individual prediction explanations
- Waterfall plots
- Feature contribution breakdowns

Uses TreeExplainer for tree-based models (exact, fast)
and LinearExplainer for logistic regression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SHAP is an optional heavy dependency. Import it lazily so the rest of the
# platform -- training, scoring, the dashboard -- still runs without it; only
# the explainability features degrade.
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install profile
    shap = None
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Professional dark theme constants
BG_DARK = "#0a0e17"
BG_CARD = "#1a2332"
TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#64748b"
GRID_COLOR = "#2a3a4e"
BORDER_COLOR = "#2a3a4e"
FIG_DPI = 200

# Feature name mapping for clean display labels
FEATURE_DISPLAY_NAMES = {
    "air_temp_k": "Air Temperature",
    "process_temp_k": "Process Temperature",
    "rotational_speed_rpm": "Rotational Speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool Wear",
    "type": "Product Type",
    "temp_diff": "Temperature Differential",
    "power": "Mechanical Power",
    "torque_per_rpm": "Load Efficiency",
    "strain": "Mechanical Strain",
    "power_factor": "Power Factor",
    "temp_rpm_interaction": "Thermal-Speed Stress",
    "tool_wear_severity": "Wear Severity",
    "is_high_torque": "High Torque Flag",
    "is_low_speed": "Low Speed Flag",
    "overload_indicator": "Overload Indicator",
}


def _apply_dark_theme():
    """Apply professional dark industrial theme to matplotlib."""
    plt.rcParams.update({
        'figure.facecolor': BG_DARK,
        'axes.facecolor': BG_CARD,
        'axes.edgecolor': BORDER_COLOR,
        'axes.labelcolor': TEXT_SECONDARY,
        'axes.titlecolor': TEXT_PRIMARY,
        'xtick.color': TEXT_MUTED,
        'ytick.color': TEXT_MUTED,
        'text.color': TEXT_PRIMARY,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
        'savefig.facecolor': BG_DARK,
        'savefig.edgecolor': BG_DARK,
    })


def _clean_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to clean display names."""
    rename_map = {col: FEATURE_DISPLAY_NAMES.get(col, col.replace('_', ' ').title())
                  for col in X.columns}
    return X.rename(columns=rename_map)


class SHAPEngine:
    """SHAP-based explainability engine for predictive maintenance models.

    Provides both global and local explanations of model predictions.
    Caches SHAP values for dashboard performance.
    """

    def __init__(
        self,
        model: Any,
        X_background: pd.DataFrame,
        config: Dict[str, Any],
        model_type: str = "tree",
        feature_names: List[str] = None,
    ):
        """Initialize SHAP engine.

        Args:
            model: Trained model (best estimator or calibrated model).
            X_background: Background dataset for SHAP (subsample of training data).
            config: Project configuration.
            model_type: Type of model ('tree' or 'linear').
            feature_names: Feature column names for display.
        """
        self.model = model
        self.config = config
        self.model_type = model_type
        self.feature_names = feature_names or list(X_background.columns)

        n_background = min(
            config["shap"]["background_samples"],
            len(X_background)
        )
        self.X_background = X_background.sample(
            n=n_background,
            random_state=config["project"]["random_seed"]
        )

        # Initialize appropriate explainer
        self.explainer = self._create_explainer()

        # Cache for computed SHAP values
        self._shap_values_cache = None
        self._shap_X_cache = None

        logger.info(
            "SHAP engine initialized — model_type: %s, background_samples: %d",
            model_type, n_background
        )

    def _create_explainer(self) -> shap.Explainer:
        """Create the appropriate SHAP explainer based on model type.

        Returns:
            SHAP explainer object.
        """
        try:
            if self.model_type == "tree":
                # For tree-based models (RF, XGBoost, HGBM)
                # Try to get the underlying estimator if calibrated
                base_model = self._get_base_model()
                explainer = shap.TreeExplainer(
                    base_model,
                    data=self.X_background,
                    feature_perturbation="interventional",
                )
                logger.info("Created TreeExplainer.")
            elif self.model_type == "linear":
                base_model = self._get_base_model()
                explainer = shap.LinearExplainer(
                    base_model,
                    self.X_background,
                )
                logger.info("Created LinearExplainer.")
            else:
                # Fallback to KernelExplainer (model-agnostic, slower)
                explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self.X_background,
                )
                logger.info("Created KernelExplainer (fallback).")

            return explainer

        except Exception as e:
            logger.warning(
                "Failed to create specialized explainer: %s. "
                "Falling back to KernelExplainer.",
                e
            )
            return shap.KernelExplainer(
                self.model.predict_proba,
                self.X_background,
            )

    def _get_base_model(self) -> Any:
        """Extract the base model from a potentially calibrated wrapper.

        Returns:
            Base estimator.
        """
        model = self.model

        # If CalibratedClassifierCV, get the base estimator
        if hasattr(model, "calibrated_classifiers_"):
            # Get the first calibrated classifier's base estimator
            model = model.calibrated_classifiers_[0].estimator
            logger.info("Extracted base model from CalibratedClassifierCV.")

        # If Pipeline, get the last step
        if hasattr(model, "steps"):
            model = model.steps[-1][1]

        return model

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = None,
    ) -> shap.Explanation:
        """Compute SHAP values for a dataset.

        Args:
            X: Feature DataFrame to explain.
            max_samples: Maximum samples to compute SHAP for.

        Returns:
            SHAP Explanation object.
        """
        if max_samples and len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=self.config["project"]["random_seed"])

        logger.info("Computing SHAP values for %d samples...", len(X))

        try:
            shap_values = self.explainer.shap_values(X)

            # Handle multi-output (binary classification returns 2 arrays or 3D array)
            if isinstance(shap_values, list):
                # Take the positive class (failure) SHAP values
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            # Create Explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value if not isinstance(
                    self.explainer.expected_value, list
                ) else self.explainer.expected_value[1],
                data=X.values,
                feature_names=list(X.columns),
            )

            self._shap_values_cache = shap_values
            self._shap_X_cache = X

            logger.info("SHAP values computed successfully.")
            return explanation

        except Exception as e:
            logger.error("SHAP computation failed: %s", e)
            raise

    def get_global_importance(
        self,
        X: pd.DataFrame = None,
        shap_values: np.ndarray = None,
    ) -> pd.DataFrame:
        """Get global feature importance based on mean |SHAP| values.

        Args:
            X: Feature DataFrame (uses cached if None).
            shap_values: Pre-computed SHAP values (uses cached if None).

        Returns:
            DataFrame with feature importance ranking.
        """
        if shap_values is None:
            shap_values = self._shap_values_cache
        if X is None:
            X = self._shap_X_cache

        if shap_values is None:
            raise ValueError("No SHAP values available. Call compute_shap_values first.")

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": list(X.columns) if X is not None else self.feature_names,
            "mean_abs_shap": mean_abs_shap,
            "rank": range(1, len(mean_abs_shap) + 1),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["contribution_pct"] = (
            importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum() * 100
        ).round(2)

        return importance_df

    def explain_single_prediction(
        self,
        X_single: pd.DataFrame,
        top_n: int = None,
    ) -> Dict[str, Any]:
        """Generate explanation for a single prediction.

        Args:
            X_single: Single-row DataFrame with features.
            top_n: Number of top contributing features to return.

        Returns:
            Dictionary with prediction explanation.
        """
        if top_n is None:
            top_n = self.config["shap"]["max_display_features"]

        # Compute SHAP values for this single instance
        try:
            shap_vals = self.explainer.shap_values(X_single)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]

            shap_vals = shap_vals.flatten()

            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]

        except Exception as e:
            logger.error("Single prediction SHAP failed: %s", e)
            # Return empty explanation
            return {
                "shap_values": [],
                "base_value": 0,
                "contributions": [],
                "top_factors": [],
                "error": str(e),
            }

        # Build contribution breakdown
        feature_names = list(X_single.columns)
        contributions = []
        for i, feat in enumerate(feature_names):
            contributions.append({
                "feature": feat,
                "value": float(X_single.iloc[0, i]),
                "shap_value": float(shap_vals[i]),
                "impact": "increases risk" if shap_vals[i] > 0 else "decreases risk",
                "abs_importance": abs(float(shap_vals[i])),
            })

        # Sort by absolute importance
        contributions.sort(key=lambda x: x["abs_importance"], reverse=True)
        top_factors = contributions[:top_n]

        explanation = {
            "shap_values": shap_vals.tolist(),
            "base_value": float(base_value),
            "contributions": contributions,
            "top_factors": top_factors,
            "feature_names": feature_names,
            "input_values": X_single.iloc[0].to_dict(),
        }

        return explanation

    # ========================================================================
    # Plot Generation
    # ========================================================================

    def plot_summary(
        self,
        X: pd.DataFrame = None,
        shap_values: np.ndarray = None,
        save_dir: Path = None,
        max_display: int = None,
    ) -> plt.Figure:
        """Generate SHAP summary/beeswarm plot.

        Args:
            X: Feature DataFrame.
            shap_values: Pre-computed SHAP values.
            save_dir: Directory to save figure.
            max_display: Max features to display.

        Returns:
            Matplotlib Figure.
        """
        _apply_dark_theme()
        if shap_values is None:
            shap_values = self._shap_values_cache
        if X is None:
            X = self._shap_X_cache
        if max_display is None:
            max_display = self.config["shap"]["max_display_features"]

        # Use clean feature names
        X_display = _clean_columns(X.copy()) if X is not None else X

        fig, ax = plt.subplots(figsize=(12, 9))
        shap.summary_plot(
            shap_values, X_display,
            max_display=max_display,
            show=False,
        )
        plt.title("Feature Impact on Failure Prediction — SHAP Analysis",
                   fontsize=15, fontweight="bold", color=TEXT_PRIMARY, pad=15)
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "shap_summary.png", dpi=FIG_DPI,
                        bbox_inches="tight", facecolor=BG_DARK)
            logger.info("Saved SHAP summary plot.")

        fig = plt.gcf()
        return fig

    def plot_bar(
        self,
        X: pd.DataFrame = None,
        shap_values: np.ndarray = None,
        save_dir: Path = None,
        max_display: int = None,
    ) -> plt.Figure:
        """Generate SHAP bar plot (mean |SHAP|).

        Args:
            X: Feature DataFrame.
            shap_values: Pre-computed SHAP values.
            save_dir: Directory to save figure.
            max_display: Max features to display.

        Returns:
            Matplotlib Figure.
        """
        _apply_dark_theme()
        if shap_values is None:
            shap_values = self._shap_values_cache
        if X is None:
            X = self._shap_X_cache
        if max_display is None:
            max_display = self.config["shap"]["max_display_features"]

        # Use clean feature names
        X_display = _clean_columns(X.copy()) if X is not None else X

        fig, ax = plt.subplots(figsize=(12, 7))
        shap.summary_plot(
            shap_values, X_display,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title("Feature Importance — Mean Absolute SHAP Value",
                   fontsize=15, fontweight="bold", color=TEXT_PRIMARY, pad=15)
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "shap_bar.png", dpi=FIG_DPI,
                        bbox_inches="tight", facecolor=BG_DARK)
            logger.info("Saved SHAP bar plot.")

        fig = plt.gcf()
        return fig

    def plot_waterfall_single(
        self,
        explanation: Dict[str, Any],
        save_dir: Path = None,
    ) -> plt.Figure:
        """Generate waterfall plot for a single prediction explanation.

        Args:
            explanation: Output from explain_single_prediction.
            save_dir: Directory to save figure.

        Returns:
            Matplotlib Figure.
        """
        max_display = self.config["shap"]["max_display_features"]

        try:
            shap_explanation = shap.Explanation(
                values=np.array(explanation["shap_values"]),
                base_values=explanation["base_value"],
                data=np.array([explanation["input_values"][f]
                               for f in explanation["feature_names"]]),
                feature_names=explanation["feature_names"],
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_explanation, max_display=max_display, show=False)
            plt.title("SHAP Waterfall — Individual Prediction Explanation",
                       fontsize=13, fontweight="bold", pad=15)
            plt.tight_layout()

            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")

            fig = plt.gcf()
            return fig

        except Exception as e:
            logger.error("Waterfall plot failed: %s", e)
            return None

    def generate_all_plots(
        self,
        X: pd.DataFrame,
        save_dir: Path,
    ) -> None:
        """Generate and save all SHAP plots.

        Args:
            X: Feature DataFrame.
            save_dir: Directory to save figures.
        """
        logger.info("Generating SHAP plots...")

        explanation = self.compute_shap_values(X, max_samples=500)
        self.plot_summary(X=self._shap_X_cache, save_dir=save_dir)
        plt.close("all")
        self.plot_bar(X=self._shap_X_cache, save_dir=save_dir)
        plt.close("all")

        logger.info("All SHAP plots saved.")
