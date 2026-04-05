from __future__ import annotations

"""Deployment pipeline that turns saved model artifacts into dashboard-ready outputs."""

import json
import joblib
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import feature_builder, preprocessing
from src.models.best_xgboost import BEST_CALIBRATION_METHOD, BEST_PARAMS, BEST_THRESHOLD


REFERENCE_COLS = ["subject_id", "hadm_id", "admittime"]
RISK_BAND_BINS = [-0.001, 0.10, 0.30, 0.60, 1.00]
RISK_BAND_LABELS = ["Low", "Moderate", "High", "Very High"]


class EHRMortalityEndToEndPipeline:
    """Turn processed or raw EHR inputs into dashboard-ready prediction rows."""

    def __init__(
        self,
        *,
        model: Any,
        expected_input_columns: list[str],
        threshold: float,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store the packaged model, expected schema, and deployment metadata."""
        self.model = model
        self.expected_input_columns = list(expected_input_columns)
        self.threshold = float(threshold)
        self.model_metadata = model_metadata or {}

    @staticmethod
    def _validate_reference_columns(df: pd.DataFrame, df_name: str) -> None:
        """Ensure the encounter key columns needed for merges are present."""
        missing_cols = [col for col in REFERENCE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing reference columns in {df_name}: {missing_cols}")

    @staticmethod
    def _normalize_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize encounter key columns before joining or scoring rows."""
        normalized_df = df.copy()
        normalized_df["admittime"] = pd.to_datetime(normalized_df["admittime"], errors="coerce")
        if normalized_df["admittime"].isna().any():
            raise ValueError("admittime contains invalid datetime values.")
        return normalized_df

    def build_predictions_df(self, model_ready_df: pd.DataFrame) -> pd.DataFrame:
        """Score model-ready rows and return encounter-level prediction records."""
        self._validate_reference_columns(model_ready_df, "model-ready table")
        model_ready_df = self._normalize_reference_columns(model_ready_df)

        # Reindex to the packaged schema so inference remains stable across runs.
        X_score = model_ready_df.reindex(columns=self.expected_input_columns, fill_value=0).copy()
        predicted_probability = self.model.predict_proba(X_score)[:, 1]
        predicted_label = (predicted_probability >= self.threshold).astype(int)
        risk_band = pd.cut(
            predicted_probability,
            bins=RISK_BAND_BINS,
            labels=RISK_BAND_LABELS,
        )

        predictions_df = model_ready_df[REFERENCE_COLS].copy()
        predictions_df["predicted_probability"] = predicted_probability
        predictions_df["predicted_label"] = predicted_label
        predictions_df["risk_band"] = risk_band.astype(str)
        predictions_df["threshold_used"] = self.threshold
        return predictions_df

    def build_dashboard_table(
        self,
        master_df: pd.DataFrame,
        model_ready_df: pd.DataFrame,
        *,
        template_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Merge predictions into a master table and return dashboard-ready rows."""
        self._validate_reference_columns(master_df, "master table")
        master_df = self._normalize_reference_columns(master_df)
        predictions_df = self.build_predictions_df(model_ready_df)

        dashboard_df = master_df.merge(
            predictions_df,
            on=REFERENCE_COLS,
            how="left",
            validate="one_to_one",
        )

        dashboard_df["predicted_label_text"] = dashboard_df["predicted_label"].map({
            0: "Predicted Survive",
            1: "Predicted Death",
        })

        if "target" in dashboard_df.columns:
            dashboard_df["actual_label_text"] = dashboard_df["target"].map({
                0: "Actual Survive",
                1: "Actual Death",
            })

        if "age" in dashboard_df.columns:
            dashboard_df["age_group_dashboard"] = pd.cut(
                dashboard_df["age"],
                bins=[-float("inf"), 39, 59, float("inf")],
                labels=["<40", "40-59", "60+"],
            )

        dashboard_df["predicted_probability_pct"] = (dashboard_df["predicted_probability"] * 100).round(2)

        if template_columns is not None:
            dashboard_df = dashboard_df.reindex(columns=template_columns)

        return dashboard_df

    def run_from_processed_files(
        self,
        master_path: str | Path = feature_builder.MASTER_CSV,
        model_ready_path: str | Path = preprocessing.MODEL_READY_CSV,
        *,
        template_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load processed tables from disk and build the final dashboard table."""
        master_df = pd.read_csv(master_path)
        model_ready_df = pd.read_csv(model_ready_path)
        return self.build_dashboard_table(master_df, model_ready_df, template_columns=template_columns)

    def run_from_raw_sources(
        self,
        *,
        num_buckets: int = feature_builder.DEFAULT_NUM_BUCKETS,
        chunksize: int = feature_builder.DEFAULT_CHUNKSIZE,
        batch_size: int = preprocessing.DEFAULT_BATCH_SIZE,
        save_master_outputs: bool = True,
        template_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run feature building, preprocessing, and scoring from the raw source tables."""
        master_df = feature_builder.build_master_table(num_buckets=num_buckets, chunksize=chunksize)
        master_df["admittime"] = pd.to_datetime(master_df["admittime"], errors="coerce")

        if master_df["admittime"].isna().any():
            raise ValueError("Master table contains invalid admittime values.")

        if save_master_outputs:
            master_df.to_csv(feature_builder.MASTER_CSV, index=False)
            master_df.to_parquet(feature_builder.MASTER_PARQUET, index=False)
            preprocessing_input_path = feature_builder.MASTER_PARQUET
        else:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
            master_df.to_parquet(temp_path, index=False)
            preprocessing_input_path = temp_path

        try:
            preprocessing.preprocess_master_table(preprocessing_input_path, batch_size=batch_size)
        finally:
            if not save_master_outputs and preprocessing_input_path.exists():
                preprocessing_input_path.unlink()

        model_ready_df = pd.read_csv(preprocessing.MODEL_READY_CSV)
        return self.build_dashboard_table(master_df, model_ready_df, template_columns=template_columns)

    @staticmethod
    def save_dashboard_outputs(
        dashboard_df: pd.DataFrame,
        *,
        csv_path: str | Path,
        parquet_path: str | Path,
    ) -> None:
        """Persist dashboard-ready outputs to CSV and Parquet."""
        dashboard_df.to_csv(csv_path, index=False)
        dashboard_df.to_parquet(parquet_path, index=False)


def save_packaged_model(
    model: Any,
    output_dir: str | Path,
    *,
    input_feature_columns: list[str],
    threshold: float | None = None,
    model_metadata: dict[str, Any] | None = None,
    source_artifacts_dir: str | Path | None = None,
) -> None:
    """Save the packaged inference pipeline and its deployment metadata."""
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
    packaged_pipeline = EHRMortalityEndToEndPipeline(
        model=model,
        expected_input_columns=input_feature_columns,
        threshold=resolved_threshold,
        model_metadata=source_metadata,
    )

    joblib.dump(packaged_pipeline, model_path)

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(input_feature_columns, f, indent=4)

    metadata = {
        "model_file": model_path.name,
        "packaged_object_type": type(packaged_pipeline).__name__,
        "threshold": resolved_threshold,
        "best_params": resolved_best_params,
        "calibration_method": resolved_calibration_method,
        "expected_model_input_columns_file": schema_path.name,
        "prediction_probability_column": "predicted_probability",
        "prediction_label_column": "predicted_label",
    }

    if "model_name" in source_metadata:
        metadata["model_name"] = source_metadata["model_name"]
    if "scale_pos_weight" in source_metadata:
        metadata["scale_pos_weight"] = source_metadata["scale_pos_weight"]
    if source_artifacts_dir is not None:
        metadata["source_artifacts_dir"] = str(Path(source_artifacts_dir))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
