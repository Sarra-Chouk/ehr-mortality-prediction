from __future__ import annotations

import json
import joblib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = BASE_DIR / "deployment"
PACKAGED_MODEL_DIR = DEPLOYMENT_DIR / "packaged_model"
OUTPUT_DIR = DEPLOYMENT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PACKAGED_MODEL_DIR / "xgb_end_to_end_batch_model.joblib"
METADATA_PATH = PACKAGED_MODEL_DIR / "deployment_metadata.json"
SCHEMA_PATH = PACKAGED_MODEL_DIR / "expected_model_input_columns.json"

MASTER_TABLE_PARQUET = BASE_DIR / "data" / "processed" / "master_table.parquet"
MODEL_READY_CSV = BASE_DIR / "data" / "processed" / "master_table_model_ready.csv"

SCORED_OUTPUT_PATH = OUTPUT_DIR / "scored_records.csv"


def run_feature_builder() -> None:
    subprocess.run(
        [sys.executable, str(BASE_DIR / "src" / "data" / "feature_builder.py"), "--save-parquet"],
        check=True,
    )


def run_preprocessing() -> None:
    subprocess.run(
        [sys.executable, str(BASE_DIR / "src" / "data" / "preprocessing.py"), "--input-path", str(MASTER_TABLE_PARQUET)],
        check=True,
    )


def load_artifacts():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        expected_columns = json.load(f)

    return model, metadata, expected_columns


def main() -> None:
    print("Step 1: Building master table from raw data...")
    run_feature_builder()

    print("Step 2: Building model-ready table...")
    run_preprocessing()

    print("Step 3: Loading packaged model...")
    model, metadata, expected_columns = load_artifacts()
    threshold = metadata["threshold"]

    print("Step 4: Loading model-ready rows...")
    df = pd.read_csv(MODEL_READY_CSV)

    reference_cols = [c for c in ["subject_id", "hadm_id", "admittime"] if c in df.columns]

    X_new = df.reindex(columns=expected_columns, fill_value=0).copy()

    print("Step 5: Scoring...")
    predicted_probability = model.predict_proba(X_new)[:, 1]
    predicted_label = (predicted_probability >= threshold).astype(int)

    scored_output = df[reference_cols].copy() if reference_cols else pd.DataFrame(index=df.index)
    scored_output["predicted_probability"] = predicted_probability
    scored_output["predicted_label"] = predicted_label
    scored_output["threshold_used"] = threshold
    scored_output["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    scored_output.to_csv(SCORED_OUTPUT_PATH, index=False)

    print("Scoring complete.")
    print("Saved scored output to:", SCORED_OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()