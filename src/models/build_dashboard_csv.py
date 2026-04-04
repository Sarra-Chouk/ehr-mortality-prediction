import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# -----------------------------------
# Paths
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "xgb_end_to_end_batch_model.joblib"
METADATA_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "deployment_metadata.json"
SCHEMA_PATH = PROJECT_ROOT / "deployment" / "packaged_model" / "expected_model_input_columns.json"

MASTER_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "master_table.csv"
MODEL_READY_PATH = PROJECT_ROOT / "data" / "processed" / "master_table_model_ready.csv"

DASHBOARD_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "dashboard_master_with_predictions.csv"
DASHBOARD_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "dashboard_master_with_predictions.parquet"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("MODEL_PATH:", MODEL_PATH)

# -----------------------------------
# Load artifacts
# -----------------------------------
print("Loading packaged model and metadata...")
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    expected_model_columns = json.load(f)

threshold = metadata["threshold"]

print("Model loaded successfully.")
print("Threshold:", threshold)

# -----------------------------------
# Load dashboard base table
# -----------------------------------
print("\nLoading full master table...")
master_df = pd.read_csv(MASTER_TABLE_PATH)
print("master_df shape:", master_df.shape)

print("\nLoading model-ready table...")
model_ready_df = pd.read_csv(MODEL_READY_PATH)
print("model_ready_df shape:", model_ready_df.shape)

# -----------------------------------
# Ensure reference columns exist
# -----------------------------------
reference_cols = ["subject_id", "hadm_id", "admittime"]
missing_refs_master = [c for c in reference_cols if c not in master_df.columns]
missing_refs_model_ready = [c for c in reference_cols if c not in model_ready_df.columns]

if missing_refs_master:
    raise ValueError(f"Missing reference columns in master table: {missing_refs_master}")
if missing_refs_model_ready:
    raise ValueError(f"Missing reference columns in model-ready table: {missing_refs_model_ready}")

# Normalize datetime for safe merge
master_df["admittime"] = pd.to_datetime(master_df["admittime"], errors="coerce")
model_ready_df["admittime"] = pd.to_datetime(model_ready_df["admittime"], errors="coerce")

master_keys = master_df[reference_cols].drop_duplicates()
model_ready_keys = model_ready_df[reference_cols].drop_duplicates()
unscored_keys = master_keys.merge(model_ready_keys, on=reference_cols, how="left", indicator=True)
unscored_count = int((unscored_keys["_merge"] == "left_only").sum())

if unscored_count:
    raise ValueError(
        "model_ready_df does not cover the full master table. "
        f"{unscored_count} master rows have no matching scored row. "
        "Rebuild master_table_model_ready.csv from the same full master_table.csv before creating the dashboard file."
    )

# -----------------------------------
# Build model input and score
# -----------------------------------
print("\nScoring full model-ready dataset...")

X_score = model_ready_df.reindex(columns=expected_model_columns, fill_value=0).copy()

predicted_probability = model.predict_proba(X_score)[:, 1]
predicted_label = (predicted_probability >= threshold).astype(int)

# Optional: human-friendly risk band for Power BI
risk_band = pd.cut(
    predicted_probability,
    bins=[-0.001, 0.10, 0.30, 0.60, 1.00],
    labels=["Low", "Moderate", "High", "Very High"]
)

predictions_df = model_ready_df[reference_cols].copy()
predictions_df["predicted_probability"] = predicted_probability
predictions_df["predicted_label"] = predicted_label
predictions_df["risk_band"] = risk_band.astype(str)
predictions_df["threshold_used"] = threshold

print("predictions_df shape:", predictions_df.shape)

# -----------------------------------
# Merge predictions back to master table
# -----------------------------------
print("\nMerging predictions into full master table...")

dashboard_df = master_df.merge(
    predictions_df,
    on=reference_cols,
    how="left",
    validate="one_to_one"
)

print("dashboard_df shape:", dashboard_df.shape)

# -----------------------------------
# Optional dashboard helper columns
# -----------------------------------
dashboard_df["predicted_label_text"] = dashboard_df["predicted_label"].map({
    0: "Predicted Survive",
    1: "Predicted Death"
})

if "target" in dashboard_df.columns:
    dashboard_df["actual_label_text"] = dashboard_df["target"].map({
        0: "Actual Survive",
        1: "Actual Death"
    })

# Example age buckets if not already present
if "age" in dashboard_df.columns and "age_group_dashboard" not in dashboard_df.columns:
    dashboard_df["age_group_dashboard"] = pd.cut(
        dashboard_df["age"],
        bins=[-np.inf, 39, 59, np.inf],
        labels=["<40", "40-59", "60+"]
    )

# Example formatted percentage for cards / tooltips if you want
dashboard_df["predicted_probability_pct"] = (dashboard_df["predicted_probability"] * 100).round(2)

# -----------------------------------
# Save outputs
# -----------------------------------
print("\nSaving dashboard files...")
dashboard_df.to_csv(DASHBOARD_OUTPUT_CSV, index=False)
dashboard_df.to_parquet(DASHBOARD_OUTPUT_PARQUET, index=False)

print("Saved dashboard CSV to:", DASHBOARD_OUTPUT_CSV.resolve())
print("Saved dashboard Parquet to:", DASHBOARD_OUTPUT_PARQUET.resolve())

# -----------------------------------
# Quick sanity checks
# -----------------------------------
print("\nSanity checks:")
print("Rows with predictions:", dashboard_df["predicted_probability"].notna().sum())
print("Rows without predictions:", dashboard_df["predicted_probability"].isna().sum())

print("\nPredicted label distribution:")
print(dashboard_df["predicted_label"].value_counts(dropna=False))

print("\nRisk band distribution:")
print(dashboard_df["risk_band"].value_counts(dropna=False))

print(dashboard_df.head())
