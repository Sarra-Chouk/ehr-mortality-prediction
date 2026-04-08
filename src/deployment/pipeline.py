from __future__ import annotations

"""Deployment wrapper for validated model-input scoring and raw-source inference runs."""

import json
import joblib
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import feature_builder, preprocessing
from src.deployment.schema import DEFAULT_SCHEMA_VERSION, ModelInputSchema
from src.models.best_xgboost import (
    BEST_CALIBRATION_METHOD,
    BEST_PARAMS,
    BEST_THRESHOLD,
    PIPELINE_VERSION,
    build_inference_pipeline,
)


REFERENCE_COLS = ["subject_id", "hadm_id", "admittime"]


@dataclass
class ScoringResult:
    """Container for raw/master rows, model-ready features, and structured predictions."""

    master_df: pd.DataFrame
    model_ready_df: pd.DataFrame
    predictions_df: pd.DataFrame


class EHRMortalityEndToEndPipeline:
    """Validate model-ready inputs, run the frozen inference pipeline, and return predictions."""

    def __init__(
        self,
        *,
        model_pipeline: Any,
        input_schema: ModelInputSchema,
        threshold: float,
        model_metadata: dict[str, Any] | None = None,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> None:
        """Store the frozen sklearn inference pipeline and deployment metadata."""
        self.model_pipeline = model_pipeline
        self.input_schema = input_schema
        self.threshold = float(threshold)
        self.model_metadata = model_metadata or {}
        self.pipeline_version = pipeline_version

        # Compatibility aliases used elsewhere in the repo and metadata generation.
        self.expected_input_columns = list(self.input_schema.feature_columns)
        self.encoded_feature_columns = list(getattr(self.model_pipeline, "encoded_feature_columns_", []))
        model_step = getattr(self.model_pipeline, "named_steps", {}).get("model", self.model_pipeline)
        self.model = getattr(model_step, "estimator", model_step)

    @staticmethod
    def _validate_reference_columns(df: pd.DataFrame, df_name: str) -> None:
        """Ensure encounter keys required for structured scoring outputs are present."""
        missing_cols = [col for col in REFERENCE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing reference columns in {df_name}: {missing_cols}")

    @staticmethod
    def _normalize_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize encounter key dtypes before joining or returning predictions."""
        normalized_df = df.copy()
        normalized_df["admittime"] = pd.to_datetime(normalized_df["admittime"], errors="coerce")
        if normalized_df["admittime"].isna().any():
            raise ValueError("admittime contains invalid datetime values.")
        return normalized_df

    def predict_proba_from_features(self, df: pd.DataFrame) -> pd.Series:
        """Return predicted probabilities from model-ready feature inputs."""
        self.input_schema.validate_required_columns(df)
        self.input_schema.validate_unexpected_columns(df)
        predicted_probability = self.model_pipeline.predict_proba(df)[:, 1]
        return pd.Series(predicted_probability, index=df.index, name="predicted_probability")

    def predict_from_features(self, df: pd.DataFrame) -> pd.Series:
        """Return thresholded predictions from model-ready feature inputs."""
        predicted_probability = self.predict_proba_from_features(df)
        predicted_label = (predicted_probability >= self.threshold).astype(int)
        predicted_label.name = "predicted_label"
        return predicted_label

    def score_features(self, model_ready_df: pd.DataFrame) -> pd.DataFrame:
        """Return structured prediction outputs for model-ready rows."""
        self._validate_reference_columns(model_ready_df, "model-ready table")
        normalized_df = self._normalize_reference_columns(model_ready_df)
        predicted_probability = self.predict_proba_from_features(normalized_df)
        predicted_label = (predicted_probability >= self.threshold).astype(int)

        scoring_timestamp = datetime.now(timezone.utc).isoformat()
        predictions_df = normalized_df[REFERENCE_COLS].copy()
        predictions_df["predicted_probability"] = predicted_probability.to_numpy()
        predictions_df["applied_threshold"] = self.threshold
        predictions_df["predicted_label"] = predicted_label.to_numpy()
        predictions_df["model_version"] = self.model_metadata.get("model_name", "xgboost")
        predictions_df["pipeline_version"] = self.pipeline_version
        predictions_df["schema_version"] = self.input_schema.version
        predictions_df["scoring_timestamp"] = scoring_timestamp
        return predictions_df

    def run_from_processed_files(
        self,
        master_path: str | Path = feature_builder.MASTER_CSV,
        model_ready_path: str | Path = preprocessing.MODEL_READY_CSV,
    ) -> ScoringResult:
        """Load processed inputs from disk and return structured scoring outputs."""
        master_df = pd.read_csv(master_path)
        model_ready_df = pd.read_csv(model_ready_path)
        predictions_df = self.score_features(model_ready_df)
        return ScoringResult(master_df=master_df, model_ready_df=model_ready_df, predictions_df=predictions_df)

    def run_from_raw_sources(
        self,
        *,
        num_buckets: int = feature_builder.DEFAULT_NUM_BUCKETS,
        chunksize: int = feature_builder.DEFAULT_CHUNKSIZE,
        batch_size: int = preprocessing.DEFAULT_BATCH_SIZE,
        save_master_outputs: bool = True,
    ) -> ScoringResult:
        """Build features from raw tables, preprocess to model-input rows, and score them."""
        master_df = feature_builder.build_master_table(num_buckets=num_buckets, chunksize=chunksize)
        master_df["admittime"] = pd.to_datetime(master_df["admittime"], errors="coerce")

        if master_df["admittime"].isna().any():
            raise ValueError("Master table contains invalid admittime values.")

        temp_master_path: Path | None = None
        temp_model_ready_csv: Path | None = None
        temp_model_ready_parquet: Path | None = None

        if save_master_outputs:
            master_df.to_csv(feature_builder.MASTER_CSV, index=False)
            master_df.to_parquet(feature_builder.MASTER_PARQUET, index=False)
            preprocessing_input_path = feature_builder.MASTER_PARQUET
            output_model_ready_csv = preprocessing.MODEL_READY_CSV
            output_model_ready_parquet = preprocessing.MODEL_READY_PARQUET
        else:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_master:
                temp_master_path = Path(tmp_master.name)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
                temp_model_ready_csv = Path(tmp_csv.name)
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_parquet:
                temp_model_ready_parquet = Path(tmp_parquet.name)

            master_df.to_parquet(temp_master_path, index=False)
            preprocessing_input_path = temp_master_path
            output_model_ready_csv = temp_model_ready_csv
            output_model_ready_parquet = temp_model_ready_parquet

        try:
            preprocessing.preprocess_master_table(
                preprocessing_input_path,
                batch_size=batch_size,
                output_parquet_path=output_model_ready_parquet,
                output_csv_path=output_model_ready_csv,
                expected_model_columns=[*REFERENCE_COLS, *self.expected_input_columns, preprocessing.TARGET_COL],
            )
            model_ready_df = pd.read_csv(output_model_ready_csv)
        finally:
            for temp_path in [temp_master_path, temp_model_ready_csv, temp_model_ready_parquet]:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()

        predictions_df = self.score_features(model_ready_df)
        return ScoringResult(master_df=master_df, model_ready_df=model_ready_df, predictions_df=predictions_df)


def save_packaged_model(
    model: Any,
    output_dir: str | Path,
    *,
    input_feature_columns: list[str],
    threshold: float | None = None,
    model_metadata: dict[str, Any] | None = None,
    source_artifacts_dir: str | Path | None = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> None:
    """Save the packaged sklearn inference pipeline and deployment metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgb_end_to_end_batch_model.joblib"
    schema_path = output_dir / "expected_model_input_columns.json"
    metadata_path = output_dir / "deployment_metadata.json"

    source_metadata = model_metadata or {}
    resolved_threshold = threshold if threshold is not None else source_metadata.get("best_threshold", BEST_THRESHOLD)
    resolved_best_params = source_metadata.get("best_params", BEST_PARAMS)
    resolved_calibration_method = source_metadata.get(
        "best_calibration_method",
        source_metadata.get("calibration_method", BEST_CALIBRATION_METHOD),
    )

    model_pipeline, input_schema = build_inference_pipeline(
        model,
        input_feature_columns,
        schema_version=schema_version,
    )
    packaged_pipeline = EHRMortalityEndToEndPipeline(
        model_pipeline=model_pipeline,
        input_schema=input_schema,
        threshold=resolved_threshold,
        model_metadata=source_metadata,
        pipeline_version=PIPELINE_VERSION,
    )

    joblib.dump(packaged_pipeline, model_path)

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(list(input_schema.feature_columns), f, indent=4)

    metadata = {
        "model_file": model_path.name,
        "packaged_object_type": type(packaged_pipeline).__name__,
        "threshold": resolved_threshold,
        "best_params": resolved_best_params,
        "calibration_method": resolved_calibration_method,
        "expected_model_input_columns_file": schema_path.name,
        "trained_model_feature_count": len(input_feature_columns),
        "prediction_probability_column": "predicted_probability",
        "prediction_label_column": "predicted_label",
        "applied_threshold_column": "applied_threshold",
        "pipeline_version": PIPELINE_VERSION,
        "schema": input_schema.to_metadata(),
    }

    if "model_name" in source_metadata:
        metadata["model_name"] = source_metadata["model_name"]
    if "scale_pos_weight" in source_metadata:
        metadata["scale_pos_weight"] = source_metadata["scale_pos_weight"]
    if source_artifacts_dir is not None:
        metadata["source_artifacts_dir"] = str(Path(source_artifacts_dir))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
