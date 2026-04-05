from __future__ import annotations

"""Utilities for the final XGBoost model and its saved artifact bundle."""

import argparse
import json
import joblib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XGB_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "xgboost"

BEST_PARAMS = {
    "colsample_bytree": 0.8,
    "gamma": 0,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "n_estimators": 200,
    "subsample": 1.0,
}

BEST_CALIBRATION_METHOD = "sigmoid"
BEST_THRESHOLD = 0.07


def load_saved_xgb_artifact(
    artifacts_dir: str | Path = DEFAULT_XGB_ARTIFACTS_DIR,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Load the saved best XGBoost artifact and its feature schema."""
    artifacts_dir = Path(artifacts_dir)

    model_path = artifacts_dir / "xgb_model.joblib"
    metadata_path = artifacts_dir / "metadata.json"
    feature_names_path = artifacts_dir / "feature_names.csv"

    missing_paths = [path for path in [model_path, metadata_path, feature_names_path] if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required XGBoost artifact files: {missing_text}")

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names_df = pd.read_csv(feature_names_path)
    if "feature_names" not in feature_names_df.columns:
        raise ValueError(f"Expected 'feature_names' column in {feature_names_path}")

    feature_names = feature_names_df["feature_names"].astype(str).tolist()
    return model, metadata, feature_names


def load_model_ready_train_data(train_full_csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the model-ready training split and return features plus target."""
    df = pd.read_csv(train_full_csv_path)

    target_col = "target"
    drop_cols = ["subject_id", "hadm_id", "admittime", "_original_order"]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=[target_col] + existing_drop_cols, errors="ignore").copy()
    y = df[target_col].copy()
    return X, y


def summarize_saved_artifact(artifacts_dir: str | Path = DEFAULT_XGB_ARTIFACTS_DIR) -> dict[str, Any]:
    """Return a concise summary of the saved XGBoost artifact bundle."""
    model, metadata, feature_names = load_saved_xgb_artifact(artifacts_dir)
    return {
        "artifacts_dir": str(Path(artifacts_dir)),
        "model_class": type(model).__name__,
        "model_name": metadata.get("model_name"),
        "threshold": metadata.get("best_threshold", metadata.get("threshold", BEST_THRESHOLD)),
        "calibration_method": metadata.get(
            "best_calibration_method",
            metadata.get("calibration_method", BEST_CALIBRATION_METHOD),
        ),
        "feature_count": len(feature_names),
    }


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute the XGBoost positive-class weight from a binary target series."""
    n_negative = int((y == 0).sum())
    n_positive = int((y == 1).sum())

    if n_positive == 0:
        raise ValueError("No positive samples found in y.")

    return n_negative / n_positive


def fit_calibrated_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    calibration_method: str = BEST_CALIBRATION_METHOD,
    random_state: int = 42,
):
    """Fit the final calibrated XGBoost model from model-ready training data."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=random_state,
    )

    scale_pos_weight = compute_scale_pos_weight(y_fit)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **BEST_PARAMS,
    )
    base_model.fit(X_fit, y_fit)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method=calibration_method,
    )
    calibrated_model.fit(X_cal, y_cal)

    return calibrated_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for artifact inspection."""
    parser = argparse.ArgumentParser(description="Inspect the saved best XGBoost artifact bundle.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_XGB_ARTIFACTS_DIR,
        help="Directory containing xgb_model.joblib, metadata.json, and feature_names.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Print a short summary of the saved best XGBoost artifact."""
    args = parse_args()
    summary = summarize_saved_artifact(args.artifacts_dir)

    print("Best XGBoost artifact summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
