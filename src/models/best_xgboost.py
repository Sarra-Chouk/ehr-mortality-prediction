from __future__ import annotations

"""Utilities for the final XGBoost model and its saved artifact bundle."""

import argparse
import json
import joblib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data import feature_builder, preprocessing
from src.deployment.schema import DEFAULT_SCHEMA_VERSION, ModelInputSchema

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
PIPELINE_VERSION = "3.0.0"

NUMERIC_MODEL_INPUT_FEATURE_COLUMNS = [
    col for col in preprocessing.MODEL_INPUT_FEATURE_COLUMNS
    if col not in preprocessing.CATEGORICAL_COLS
]
CATEGORICAL_MODEL_INPUT_FEATURE_COLUMNS = list(preprocessing.CATEGORICAL_COLS)


class ModelInputValidator(BaseEstimator, TransformerMixin):
    """Validate and align pre-encoded model-input features before sklearn preprocessing."""

    def __init__(self, schema: ModelInputSchema) -> None:
        """Store the packaged schema used to validate inference inputs."""
        self.schema = schema

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ModelInputValidator":
        """Mark the validator as fitted without changing the frozen schema state."""
        self.feature_names_in_ = pd.Index(self.schema.feature_columns, dtype="object")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply schema validation and feature alignment before categorical encoding."""
        return self.schema.prepare_features_for_scoring(X)


class EncodedFeatureAligner(BaseEstimator, TransformerMixin):
    """Align encoded sklearn features to the exact feature order expected by the saved model."""

    def __init__(self, expected_feature_columns: list[str] | tuple[str, ...]) -> None:
        """Store the trained model feature order from the saved XGBoost artifact."""
        self.expected_feature_columns = [str(col) for col in expected_feature_columns]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "EncodedFeatureAligner":
        """Record the observed encoded feature columns for debugging and compatibility."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EncodedFeatureAligner expects a pandas DataFrame from the sklearn preprocessor.")

        self.observed_feature_columns_ = tuple(str(col) for col in X.columns)
        self.feature_names_in_ = pd.Index(self.observed_feature_columns_, dtype="object")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop unexpected encoded columns, add missing trained columns, and preserve model order."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EncodedFeatureAligner expects a pandas DataFrame from the sklearn preprocessor.")

        aligned_df = X.reindex(columns=self.expected_feature_columns, fill_value=0.0).copy()
        for col in aligned_df.columns:
            aligned_df[col] = pd.to_numeric(aligned_df[col], errors="coerce")
        return aligned_df


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


def load_model_input_fit_data(
    model_input_path: str | Path = preprocessing.MODEL_READY_CSV,
) -> pd.DataFrame:
    """Load or rebuild the pre-encoded model-input table used to fit the frozen sklearn preprocessor."""
    model_input_path = Path(model_input_path)

    needs_rebuild = True
    if model_input_path.exists():
        existing_columns = pd.read_csv(model_input_path, nrows=0).columns.tolist()
        needs_rebuild = any(col not in existing_columns for col in preprocessing.MODEL_INPUT_COLUMNS)

    if needs_rebuild:
        master_input_path = (
            feature_builder.MASTER_CSV if feature_builder.MASTER_CSV.exists() else feature_builder.MASTER_PARQUET
        )
        if not master_input_path.exists():
            raise FileNotFoundError(
                "Could not rebuild model-input data because no processed master table was found at "
                f"{feature_builder.MASTER_CSV} or {feature_builder.MASTER_PARQUET}."
            )
        preprocessing.preprocess_master_table(master_input_path)

    model_input_df = pd.read_csv(model_input_path)
    missing_columns = [col for col in preprocessing.MODEL_INPUT_COLUMNS if col not in model_input_df.columns]
    if missing_columns:
        raise ValueError(
            "The model-input table is missing required pre-encoded columns even after preprocessing: "
            f"{missing_columns}"
        )

    drop_columns = [preprocessing.TARGET_COL, *preprocessing.REFERENCE_COLS]
    return model_input_df.drop(columns=drop_columns, errors="ignore").copy()


def build_categorical_preprocessor() -> ColumnTransformer:
    """Create the frozen sklearn preprocessing layer for categorical model-input features."""
    category_order = [preprocessing.ALLOWED_CATEGORY_LEVELS[col] for col in CATEGORICAL_MODEL_INPUT_FEATURE_COLUMNS]
    try:
        one_hot_encoder = OneHotEncoder(
            categories=category_order,
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        one_hot_encoder = OneHotEncoder(
            categories=category_order,
            drop="first",
            handle_unknown="ignore",
            sparse=False,
        )

    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", one_hot_encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_MODEL_INPUT_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_INPUT_FEATURE_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_inference_pipeline(
    model: Any,
    feature_columns: list[str],
    *,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    fit_data: pd.DataFrame | None = None,
) -> tuple[SklearnPipeline, ModelInputSchema]:
    """Wrap the saved model in a fitted sklearn pipeline that owns categorical preprocessing."""
    input_schema = ModelInputSchema.from_feature_columns(
        numeric_feature_columns=NUMERIC_MODEL_INPUT_FEATURE_COLUMNS,
        categorical_feature_columns=CATEGORICAL_MODEL_INPUT_FEATURE_COLUMNS,
        version=schema_version,
    )
    fit_features = fit_data.copy() if fit_data is not None else load_model_input_fit_data()
    fit_features = input_schema.prepare_features_for_scoring(fit_features)

    inference_pipeline = SklearnPipeline(
        steps=[
            ("validator", ModelInputValidator(input_schema)),
            ("categorical_preprocessor", build_categorical_preprocessor()),
            ("feature_aligner", EncodedFeatureAligner(feature_columns)),
            ("model", FrozenEstimator(model)),
        ]
    )
    inference_pipeline.fit(fit_features)
    inference_pipeline.feature_schema_ = input_schema
    inference_pipeline.encoded_feature_columns_ = [str(col) for col in feature_columns]
    inference_pipeline.pipeline_version_ = PIPELINE_VERSION
    return inference_pipeline, input_schema


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
