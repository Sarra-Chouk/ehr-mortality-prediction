from __future__ import annotations
"""Score deployment rows and refresh dashboard-ready prediction outputs."""

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

from src.deployment.pipeline import EHRMortalityEndToEndPipeline


DEPLOYMENT_DIR = BASE_DIR / "deployment"
PACKAGED_MODEL_DIR = DEPLOYMENT_DIR / "packaged_model"
OUTPUT_DIR = DEPLOYMENT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PACKAGED_MODEL_DIR / "xgb_end_to_end_batch_model.joblib"
METADATA_PATH = PACKAGED_MODEL_DIR / "deployment_metadata.json"

MASTER_TABLE_CSV = BASE_DIR / "data" / "processed" / "master_table.csv"
DEPLOYMENT_FULL_CSV = BASE_DIR / "data" / "processed" / "splits" / "deployment_full.csv"
DASHBOARD_OUTPUT_CSV = BASE_DIR / "data" / "processed" / "dashboard_master_with_predictions.csv"
DASHBOARD_OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "dashboard_master_with_predictions.parquet"

SCORED_OUTPUT_PATH = OUTPUT_DIR / "scored_records.csv"
REFERENCE_COLS = ["subject_id", "hadm_id", "admittime"]


def load_artifacts() -> tuple[EHRMortalityEndToEndPipeline, dict[str, Any]]:
    """Load the packaged pipeline object and its deployment metadata."""
    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata


def load_deployment_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deployment features and the corresponding raw rows used by the dashboard."""
    deployment_full_df = pd.read_csv(DEPLOYMENT_FULL_CSV)
    master_df = pd.read_csv(MASTER_TABLE_CSV)

    missing_refs_deployment = [col for col in REFERENCE_COLS if col not in deployment_full_df.columns]
    missing_refs_master = [col for col in REFERENCE_COLS if col not in master_df.columns]

    if missing_refs_deployment:
        raise ValueError(f"Missing reference columns in deployment split: {missing_refs_deployment}")
    if missing_refs_master:
        raise ValueError(f"Missing reference columns in master table: {missing_refs_master}")

    deployment_full_df["admittime"] = pd.to_datetime(deployment_full_df["admittime"], errors="coerce")
    master_df["admittime"] = pd.to_datetime(master_df["admittime"], errors="coerce")

    if deployment_full_df["admittime"].isna().any():
        raise ValueError("Deployment split contains invalid admittime values.")
    if master_df["admittime"].isna().any():
        raise ValueError("Master table contains invalid admittime values.")

    deployment_keys = deployment_full_df[REFERENCE_COLS].drop_duplicates()
    master_subset_df = master_df.merge(deployment_keys, on=REFERENCE_COLS, how="inner", validate="one_to_one")

    if len(master_subset_df) != len(deployment_keys):
        raise ValueError(
            "Could not find a one-to-one master-table match for every deployment row. "
            "Rebuild the split files from the same master table used for the dashboard output."
        )

    return deployment_full_df, master_subset_df


def upsert_dashboard_output(new_rows_df: pd.DataFrame) -> pd.DataFrame:
    """Upsert scored rows into the dashboard outputs using the dashboard key columns."""
    if DASHBOARD_OUTPUT_CSV.exists():
        dashboard_df = pd.read_csv(DASHBOARD_OUTPUT_CSV)
        dashboard_df["admittime"] = pd.to_datetime(dashboard_df["admittime"], errors="coerce")
    else:
        dashboard_df = pd.DataFrame(columns=new_rows_df.columns)

    updated_dashboard_df = pd.concat([dashboard_df, new_rows_df], ignore_index=True)
    # Keep the newly scored row when a dashboard record already exists for the same encounter.
    updated_dashboard_df = updated_dashboard_df.drop_duplicates(subset=REFERENCE_COLS, keep="last")

    if DASHBOARD_OUTPUT_CSV.exists():
        template_columns = pd.read_csv(DASHBOARD_OUTPUT_CSV, nrows=0).columns.tolist()
        updated_dashboard_df = updated_dashboard_df.reindex(columns=template_columns)

    updated_dashboard_df.to_csv(DASHBOARD_OUTPUT_CSV, index=False)
    updated_dashboard_df.to_parquet(DASHBOARD_OUTPUT_PARQUET, index=False)
    return updated_dashboard_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for deployment scoring modes."""
    parser = argparse.ArgumentParser(description="Score deployment data or rebuild dashboard outputs from raw sources.")
    parser.add_argument(
        "--source",
        choices=["deployment", "raw"],
        default="deployment",
        help="Use the deployment split for incremental scoring or rebuild the full dashboard from raw sources.",
    )
    return parser.parse_args()


def main() -> None:
    """Run deployment scoring and refresh the dashboard-ready files."""
    args = parse_args()

    print("Step 1: Loading packaged pipeline...")
    pipeline, metadata = load_artifacts()
    template_columns = pd.read_csv(DASHBOARD_OUTPUT_CSV, nrows=0).columns.tolist() if DASHBOARD_OUTPUT_CSV.exists() else None

    if args.source == "raw":
        print("Step 2: Rebuilding the full dashboard table from raw source data...")
        dashboard_output = pipeline.run_from_raw_sources(template_columns=template_columns)
        pipeline.save_dashboard_outputs(
            dashboard_output,
            csv_path=DASHBOARD_OUTPUT_CSV,
            parquet_path=DASHBOARD_OUTPUT_PARQUET,
        )
        dashboard_output.to_csv(SCORED_OUTPUT_PATH, index=False)

        print("Scoring complete.")
        print("Rows written:", len(dashboard_output))
        print("Packaged object type:", metadata.get("packaged_object_type", type(pipeline).__name__))
        print("Saved final dashboard table to:", SCORED_OUTPUT_PATH.resolve())
        print("Updated dashboard CSV:", DASHBOARD_OUTPUT_CSV.resolve())
        print("Updated dashboard Parquet:", DASHBOARD_OUTPUT_PARQUET.resolve())
        return

    print("Step 2: Loading deployment split and matching master rows...")
    deployment_full_df, master_subset_df = load_deployment_rows()

    print("Step 3: Scoring deployment rows...")
    scored_output = pipeline.build_dashboard_table(
        master_subset_df,
        deployment_full_df,
        template_columns=template_columns,
    )

    scored_output.to_csv(SCORED_OUTPUT_PATH, index=False)
    print("Step 4: Updating dashboard-ready files...")
    updated_dashboard_df = upsert_dashboard_output(scored_output)

    print("Scoring complete.")
    print("Rows scored:", len(scored_output))
    print("Packaged object type:", metadata.get("packaged_object_type", type(pipeline).__name__))
    print("Saved scored output to:", SCORED_OUTPUT_PATH.resolve())
    print("Updated dashboard CSV:", DASHBOARD_OUTPUT_CSV.resolve())
    print("Updated dashboard Parquet:", DASHBOARD_OUTPUT_PARQUET.resolve())
    print("Dashboard rows after update:", len(updated_dashboard_df))


if __name__ == "__main__":
    main()
