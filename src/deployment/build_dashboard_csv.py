from __future__ import annotations

"""Format structured prediction outputs into the dashboard-ready table shape."""

import json
import joblib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.pipeline import EHRMortalityEndToEndPipeline, REFERENCE_COLS

MODEL_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "xgb_end_to_end_batch_model.joblib"
METADATA_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "deployment_metadata.json"
MASTER_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "master_table.csv"
MODEL_READY_PATH = PROJECT_ROOT / "data" / "processed" / "master_table_model_ready.csv"
DASHBOARD_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "dashboard_master_with_predictions.csv"
DASHBOARD_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "dashboard_master_with_predictions.parquet"


def load_template_columns(csv_path: str | Path) -> list[str] | None:
    """Load the dashboard column order if a prior output file already exists."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def apply_template_columns(df: pd.DataFrame, template_columns: list[str] | None) -> pd.DataFrame:
    """Preserve an existing column order while appending any newly introduced fields."""
    if template_columns is None:
        return df

    ordered_columns = [col for col in template_columns if col in df.columns]
    new_columns = [col for col in df.columns if col not in ordered_columns]
    return df.reindex(columns=[*ordered_columns, *new_columns])


def build_dashboard_table(
    master_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    template_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Merge structured prediction outputs into the dashboard-ready table shape."""
    missing_master_cols = [col for col in REFERENCE_COLS if col not in master_df.columns]
    missing_prediction_cols = [col for col in REFERENCE_COLS if col not in predictions_df.columns]
    if missing_master_cols:
        raise ValueError(f"Missing reference columns in master table: {missing_master_cols}")
    if missing_prediction_cols:
        raise ValueError(f"Missing reference columns in predictions table: {missing_prediction_cols}")

    normalized_master_df = master_df.copy()
    normalized_predictions_df = predictions_df.copy()
    normalized_master_df["admittime"] = pd.to_datetime(normalized_master_df["admittime"], errors="coerce")
    normalized_predictions_df["admittime"] = pd.to_datetime(normalized_predictions_df["admittime"], errors="coerce")

    dashboard_df = normalized_master_df.merge(
        normalized_predictions_df,
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

    return apply_template_columns(dashboard_df, template_columns)


def save_dashboard_outputs(
    dashboard_df: pd.DataFrame,
    *,
    csv_path: str | Path,
    parquet_path: str | Path,
) -> None:
    """Persist dashboard-ready outputs to CSV and Parquet."""
    dashboard_df.to_csv(csv_path, index=False)
    dashboard_df.to_parquet(parquet_path, index=False)


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

    scoring_result = pipeline.run_from_processed_files(
        master_path=MASTER_TABLE_PATH,
        model_ready_path=MODEL_READY_PATH,
    )
    template_columns = load_template_columns(DASHBOARD_OUTPUT_CSV)

    print("\nBuilding dashboard table from processed inputs...")
    dashboard_df = build_dashboard_table(
        scoring_result.master_df,
        scoring_result.predictions_df,
        template_columns=template_columns,
    )

    print("dashboard_df shape:", dashboard_df.shape)

    print("\nSaving dashboard files...")
    save_dashboard_outputs(
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

    print("\nApplied threshold distribution:")
    if "applied_threshold" in dashboard_df.columns:
        print(dashboard_df["applied_threshold"].value_counts(dropna=False))
    else:
        print("applied_threshold column not present in template output.")

    print(dashboard_df.head())


if __name__ == "__main__":
    main()
