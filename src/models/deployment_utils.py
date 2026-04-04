from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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


def load_model_ready_train_data(train_full_csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(train_full_csv_path)

    target_col = "target"
    drop_cols = ["subject_id", "hadm_id", "admittime", "_original_order"]

    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=[target_col] + existing_drop_cols, errors="ignore").copy()
    y = df[target_col].copy()

    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
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


def save_packaged_model(
    model: Any,
    output_dir: str | Path,
    *,
    input_feature_columns: list[str],
    threshold: float = BEST_THRESHOLD,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgb_end_to_end_batch_model.joblib"
    schema_path = output_dir / "expected_model_input_columns.json"
    metadata_path = output_dir / "deployment_metadata.json"

    joblib.dump(model, model_path)

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(input_feature_columns, f, indent=4)

    metadata = {
        "model_file": model_path.name,
        "threshold": threshold,
        "best_params": BEST_PARAMS,
        "calibration_method": BEST_CALIBRATION_METHOD,
        "expected_model_input_columns_file": schema_path.name,
        "prediction_probability_column": "predicted_probability",
        "prediction_label_column": "predicted_label",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)