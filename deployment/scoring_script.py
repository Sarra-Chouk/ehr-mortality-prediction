from __future__ import annotations

"""Run batch scoring from raw or prepared inputs and update dashboard-ready outputs."""

import argparse
import json
import joblib
from pathlib import Path
import sys
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.deployment.build_dashboard_csv import (
    apply_template_columns,
    build_dashboard_table,
    load_template_columns,
    save_dashboard_outputs,
)
from src.deployment.pipeline import EHRMortalityEndToEndPipeline, REFERENCE_COLS


DEPLOYMENT_DIR = BASE_DIR / "deployment"
PACKAGED_MODEL_DIR = DEPLOYMENT_DIR / "packaged_model"
OUTPUT_DIR = DEPLOYMENT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PACKAGED_MODEL_DIR / "xgb_end_to_end_batch_model.joblib"
METADATA_PATH = PACKAGED_MODEL_DIR / "deployment_metadata.json"

MASTER_TABLE_CSV = BASE_DIR / "data" / "processed" / "master_table.csv"
MODEL_READY_CSV = BASE_DIR / "data" / "processed" / "master_table_model_ready.csv"
DEPLOYMENT_FULL_CSV = BASE_DIR / "data" / "processed" / "splits" / "deployment_full.csv"
DASHBOARD_OUTPUT_CSV = BASE_DIR / "data" / "processed" / "dashboard_master_with_predictions.csv"
DASHBOARD_OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "dashboard_master_with_predictions.parquet"

SCORED_OUTPUT_PATH = OUTPUT_DIR / "scored_records.csv"


def load_artifacts() -> tuple[EHRMortalityEndToEndPipeline, dict[str, Any]]:
    """Load the packaged deployment wrapper and its metadata."""
    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata


def load_deployment_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deployment encounter keys, pre-encoded model inputs, and their master-table rows."""
    deployment_full_df = pd.read_csv(DEPLOYMENT_FULL_CSV)
    model_ready_df = pd.read_csv(MODEL_READY_CSV)
    master_df = pd.read_csv(MASTER_TABLE_CSV)

    missing_refs_deployment = [col for col in REFERENCE_COLS if col not in deployment_full_df.columns]
    missing_refs_model_ready = [col for col in REFERENCE_COLS if col not in model_ready_df.columns]
    missing_refs_master = [col for col in REFERENCE_COLS if col not in master_df.columns]

    if missing_refs_deployment:
        raise ValueError(f"Missing reference columns in deployment split: {missing_refs_deployment}")
    if missing_refs_model_ready:
        raise ValueError(f"Missing reference columns in model-ready table: {missing_refs_model_ready}")
    if missing_refs_master:
        raise ValueError(f"Missing reference columns in master table: {missing_refs_master}")

    deployment_full_df["admittime"] = pd.to_datetime(deployment_full_df["admittime"], errors="coerce")
    model_ready_df["admittime"] = pd.to_datetime(model_ready_df["admittime"], errors="coerce")
    master_df["admittime"] = pd.to_datetime(master_df["admittime"], errors="coerce")

    deployment_keys = deployment_full_df[REFERENCE_COLS].drop_duplicates()
    model_ready_subset_df = model_ready_df.merge(deployment_keys, on=REFERENCE_COLS, how="inner", validate="one_to_one")
    master_subset_df = master_df.merge(deployment_keys, on=REFERENCE_COLS, how="inner", validate="one_to_one")

    if len(model_ready_subset_df) != len(deployment_keys):
        raise ValueError(
            "Could not find a one-to-one model-ready match for every deployment row. "
            "Rebuild the processed model-input table from the same master table used for the split files."
        )

    if len(master_subset_df) != len(deployment_keys):
        raise ValueError(
            "Could not find a one-to-one master-table match for every deployment row. "
            "Rebuild the split files from the same master table used for the dashboard output."
        )

    return model_ready_subset_df, master_subset_df


def upsert_dashboard_output(new_rows_df: pd.DataFrame) -> pd.DataFrame:
    """Upsert scored dashboard rows using the encounter key columns."""
    if DASHBOARD_OUTPUT_CSV.exists():
        dashboard_df = pd.read_csv(DASHBOARD_OUTPUT_CSV)
        dashboard_df["admittime"] = pd.to_datetime(dashboard_df["admittime"], errors="coerce")
    else:
        dashboard_df = pd.DataFrame(columns=new_rows_df.columns)

    updated_dashboard_df = pd.concat([dashboard_df, new_rows_df], ignore_index=True)
    updated_dashboard_df = updated_dashboard_df.drop_duplicates(subset=REFERENCE_COLS, keep="last")

    template_columns = load_template_columns(DASHBOARD_OUTPUT_CSV)
    updated_dashboard_df = apply_template_columns(updated_dashboard_df, template_columns)

    save_dashboard_outputs(
        updated_dashboard_df,
        csv_path=DASHBOARD_OUTPUT_CSV,
        parquet_path=DASHBOARD_OUTPUT_PARQUET,
    )
    return updated_dashboard_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for raw or prepared deployment scoring."""
    parser = argparse.ArgumentParser(description="Score raw source data or prepared deployment rows.")
    parser.add_argument(
        "--source",
        choices=["deployment", "raw"],
        default="raw",
        help="Use the raw demo source tables by default, or score the prepared deployment split.",
    )
    return parser.parse_args()


def main() -> None:
    """Run batch scoring and refresh the dashboard-ready files."""
    args = parse_args()

    print("Step 1: Loading packaged pipeline...")
    pipeline, metadata = load_artifacts()
    template_columns = load_template_columns(DASHBOARD_OUTPUT_CSV)

    if args.source == "raw":
        print("Step 2: Building, preprocessing, and scoring rows from raw source data...")
        scoring_result = pipeline.run_from_raw_sources(save_master_outputs=False)
        dashboard_output = build_dashboard_table(
            scoring_result.master_df,
            scoring_result.predictions_df,
            template_columns=template_columns,
        )
        dashboard_output.to_csv(SCORED_OUTPUT_PATH, index=False)

        print("Step 3: Updating dashboard-ready files...")
        updated_dashboard_df = upsert_dashboard_output(dashboard_output)

        print("Scoring complete.")
        print("Rows scored from raw data:", len(dashboard_output))
        print("Packaged object type:", metadata.get("packaged_object_type", type(pipeline).__name__))
        print("Saved scored output to:", SCORED_OUTPUT_PATH.resolve())
        print("Updated dashboard CSV:", DASHBOARD_OUTPUT_CSV.resolve())
        print("Updated dashboard Parquet:", DASHBOARD_OUTPUT_PARQUET.resolve())
        print("Dashboard rows after update:", len(updated_dashboard_df))
        return

    print("Step 2: Loading prepared deployment features...")
    deployment_full_df, master_subset_df = load_deployment_rows()

    print("Step 3: Scoring prepared deployment rows...")
    predictions_df = pipeline.score_features(deployment_full_df)
    dashboard_output = build_dashboard_table(
        master_subset_df,
        predictions_df,
        template_columns=template_columns,
    )
    dashboard_output.to_csv(SCORED_OUTPUT_PATH, index=False)

    print("Step 4: Updating dashboard-ready files...")
    updated_dashboard_df = upsert_dashboard_output(dashboard_output)

    print("Scoring complete.")
    print("Rows scored:", len(dashboard_output))
    print("Packaged object type:", metadata.get("packaged_object_type", type(pipeline).__name__))
    print("Saved scored output to:", SCORED_OUTPUT_PATH.resolve())
    print("Updated dashboard CSV:", DASHBOARD_OUTPUT_CSV.resolve())
    print("Updated dashboard Parquet:", DASHBOARD_OUTPUT_PARQUET.resolve())
    print("Dashboard rows after update:", len(updated_dashboard_df))


if __name__ == "__main__":
    main()
