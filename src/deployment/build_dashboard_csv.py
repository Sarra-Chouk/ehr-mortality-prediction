from __future__ import annotations

"""Build the dashboard-ready prediction table from processed project files."""

import json
import joblib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.pipeline import EHRMortalityEndToEndPipeline


MODEL_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "xgb_end_to_end_batch_model.joblib"
METADATA_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "deployment_metadata.json"
MASTER_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "master_table.csv"
MODEL_READY_PATH = PROJECT_ROOT / "data" / "processed" / "master_table_model_ready.csv"
DASHBOARD_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "dashboard_master_with_predictions.csv"
DASHBOARD_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "dashboard_master_with_predictions.parquet"


def main() -> None:
    """Build the dashboard-ready prediction table from processed project files."""
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL_PATH:", MODEL_PATH)

    print("Loading packaged pipeline and metadata...")
    pipeline: EHRMortalityEndToEndPipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Packaged object type:", metadata.get("packaged_object_type", type(pipeline).__name__))
    print("Threshold:", metadata["threshold"])

    template_columns = pd.read_csv(DASHBOARD_OUTPUT_CSV, nrows=0).columns.tolist() if DASHBOARD_OUTPUT_CSV.exists() else None

    print("\nBuilding dashboard table from processed inputs...")
    dashboard_df = pipeline.run_from_processed_files(
        master_path=MASTER_TABLE_PATH,
        model_ready_path=MODEL_READY_PATH,
        template_columns=template_columns,
    )

    print("dashboard_df shape:", dashboard_df.shape)

    print("\nSaving dashboard files...")
    pipeline.save_dashboard_outputs(
        dashboard_df,
        csv_path=DASHBOARD_OUTPUT_CSV,
        parquet_path=DASHBOARD_OUTPUT_PARQUET,
    )

    print("Saved dashboard CSV to:", DASHBOARD_OUTPUT_CSV.resolve())
    print("Saved dashboard Parquet to:", DASHBOARD_OUTPUT_PARQUET.resolve())

    print("\nSanity checks:")
    print("Rows with predictions:", dashboard_df["predicted_probability"].notna().sum())
    print("Rows without predictions:", dashboard_df["predicted_probability"].isna().sum())

    print("\nPredicted label distribution:")
    print(dashboard_df["predicted_label"].value_counts(dropna=False))

    print("\nRisk band distribution:")
    print(dashboard_df["risk_band"].value_counts(dropna=False))

    print(dashboard_df.head())


if __name__ == "__main__":
    main()
