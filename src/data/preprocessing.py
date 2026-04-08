from __future__ import annotations

"""Convert the master table into a stable model-ready dataset."""

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PARQUET = OUTPUT_DIR / "master_table.parquet"
MODEL_READY_PARQUET = OUTPUT_DIR / "master_table_model_ready.parquet"
MODEL_READY_CSV = OUTPUT_DIR / "master_table_model_ready.csv"

CATEGORICAL_COLS = ["admission_type", "marital_status", "race"]
REFERENCE_COLS = ["subject_id", "hadm_id", "admittime"]
TARGET_COL = "target"
DEFAULT_BATCH_SIZE = 50_000

ALLOWED_CATEGORY_LEVELS = {
    "admission_type": ["elective", "urgent", "other", "unknown"],
    "marital_status": ["married", "single", "divorced", "widowed", "other", "unknown"],
    "race": ["white", "black", "asian", "hispanic_latino", "other", "unknown"],
}

EXPECTED_MASTER_COLUMNS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "age",
    "gender",
    "admission_type",
    "marital_status",
    "race",
    "ed_los_hours",
    "came_from_ed",
    "admission_hour",
    "admission_day_of_week",
    "num_prev_visits",
    "time_since_last_visit_days",
    "avg_time_between_visits_days",
    "avg_prev_los",
    "max_prev_los",
    "last_los",
    "avg_prev_ed_los",
    "max_prev_ed_los",
    "num_prev_diagnoses",
    "num_unique_diagnoses_icd_codes",
    "num_prev_procedures",
    "num_unique_procedure_icd_codes",
    "avg_prev_drg_severity",
    "max_prev_drg_severity",
    "avg_prev_drg_mortality",
    "max_prev_drg_mortality",
    "num_abnormal_labs",
    "abnormal_lab_ratio",
    "stat_lab_ratio",
    "last_lab_flag",
    "last_norm_lab",
    "num_prev_medications",
    "num_unique_drugs",
    "num_prev_transfers",
    "num_prev_icu_visits",
    "num_prev_infections",
    "num_unique_organisms",
    "num_resistant_cases",
    "bmi_last",
    "weight_last",
    "last_systolic",
    "last_diastolic",
    "last_pulse_pressure",
    "target",
]

NUMERIC_MASTER_COLS = [
    col for col in EXPECTED_MASTER_COLUMNS
    if col not in set(CATEGORICAL_COLS + [TARGET_COL, "admittime"])
]

MODEL_BASE_COLUMNS = [
    *REFERENCE_COLS,
    *[
        col for col in EXPECTED_MASTER_COLUMNS
        if col not in set(REFERENCE_COLS + CATEGORICAL_COLS + [TARGET_COL])
    ],
]

MODEL_INPUT_FEATURE_COLUMNS = [
    col for col in EXPECTED_MASTER_COLUMNS
    if col not in set(REFERENCE_COLS + [TARGET_COL])
]
MODEL_INPUT_COLUMNS = [*REFERENCE_COLS, *MODEL_INPUT_FEATURE_COLUMNS, TARGET_COL]


def log(message: str) -> None:
    """Print progress messages immediately for long-running preprocessing jobs."""
    print(message, flush=True)


def open_parquet_file(path: Path) -> pq.ParquetFile:
    """Open a parquet file with a consistent missing-file error message."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return pq.ParquetFile(path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support is unavailable. Install `pyarrow` or `fastparquet` to read/write parquet files."
        ) from exc


def get_master_schema_columns(input_path: Path) -> list[str]:
    """Read the master-table column names without loading the full dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        parquet_file = open_parquet_file(input_path)
        columns = parquet_file.schema.names
        del parquet_file
        gc.collect()
        return columns
    if suffix == ".csv":
        return pd.read_csv(input_path, nrows=0).columns.tolist()

    raise ValueError(f"Unsupported input format: {input_path.suffix}. Use .parquet or .csv")


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common string placeholders to real missing values."""
    missing_placeholders = {"", "nan", "none", "<na>"}

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            normalized = df[col].astype("string").str.strip()
            df[col] = normalized.mask(normalized.str.lower().isin(missing_placeholders), np.nan)

    return df


def validate_master_schema(columns: list[str]) -> None:
    """Check that the master table matches the expected engineered schema."""
    missing_cols = [col for col in EXPECTED_MASTER_COLUMNS if col not in columns]
    extra_cols = [col for col in columns if col not in EXPECTED_MASTER_COLUMNS]
    if missing_cols:
        raise ValueError(f"Master table is missing expected columns: {missing_cols}")
    if extra_cols:
        raise ValueError(
            "Master table contains unexpected columns. "
            f"Update preprocessing.py if the master schema changed: {extra_cols}"
        )


def validate_and_cast_master_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Validate one master-table batch and cast columns to their modeling dtypes."""
    validate_master_schema(df.columns.tolist())

    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    if df["admittime"].isna().any():
        raise ValueError("admittime contains invalid datetime values after parsing.")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string")

    for col in NUMERIC_MASTER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["age", "gender"]:
        if df[col].isna().any():
            raise ValueError(f"{col} contains missing values and cannot be safely cast to integer.")
        df[col] = df[col].round().astype("int32")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if df[TARGET_COL].isna().any():
        raise ValueError("target contains missing or non-numeric values.")
    if not set(df[TARGET_COL].dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError("target must be binary {0,1}.")
    df[TARGET_COL] = df[TARGET_COL].astype("int8")

    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic missing-value handling that stays outside sklearn."""
    return df


def infer_category_levels(input_path: Path, batch_size: int) -> dict[str, list[str]]:
    """Infer observed category levels while enforcing the allowed vocabularies."""
    validate_master_schema(get_master_schema_columns(input_path))

    observed_values: dict[str, set[str]] = {col: set() for col in CATEGORICAL_COLS}
    if input_path.suffix.lower() == ".parquet":
        batch_iter = open_parquet_file(input_path).iter_batches(batch_size=batch_size, columns=CATEGORICAL_COLS)
    else:
        batch_iter = pd.read_csv(input_path, usecols=CATEGORICAL_COLS, chunksize=batch_size, low_memory=False)

    for batch in batch_iter:
        batch_df = batch.to_pandas(types_mapper=pd.ArrowDtype) if hasattr(batch, "to_pandas") else batch.copy()
        batch_df = standardize_missing_values(batch_df)
        for col in CATEGORICAL_COLS:
            values = batch_df[col].astype("string").fillna("unknown").dropna().unique().tolist()
            observed_values[col].update(str(v) for v in values if pd.notna(v))
        del batch, batch_df
        gc.collect()

    category_levels: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLS:
        invalid_values = sorted(v for v in observed_values[col] if v not in ALLOWED_CATEGORY_LEVELS[col])
        if invalid_values:
            raise ValueError(f"Unexpected values found in {col}: {invalid_values}")
        category_levels[col] = [level for level in ALLOWED_CATEGORY_LEVELS[col] if level in observed_values[col]]

    gc.collect()
    return category_levels


def get_expected_model_columns(category_levels: dict[str, list[str]]) -> list[str]:
    """Construct the stable pre-encoded model-input column order."""
    _ = category_levels
    return MODEL_INPUT_COLUMNS


def apply_category_dtypes(df: pd.DataFrame, category_levels: dict[str, list[str]]) -> pd.DataFrame:
    """Apply validated categorical dtypes while preserving raw category columns."""
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(CategoricalDtype(categories=category_levels[col], ordered=True))
    return df


def build_model_input_batch(master: pd.DataFrame, expected_model_columns: list[str]) -> pd.DataFrame:
    """Build the deterministic pre-encoded model-input batch for downstream pipelines."""
    model_df = master.reindex(columns=expected_model_columns).copy()

    for col in CATEGORICAL_COLS:
        model_df[col] = model_df[col].astype("string")
    model_df["age"] = model_df["age"].astype("int32")
    model_df["gender"] = model_df["gender"].astype("int8")
    model_df[TARGET_COL] = model_df[TARGET_COL].astype("int8")
    return model_df


def validate_model_batch(df: pd.DataFrame, expected_model_columns: list[str]) -> None:
    """Ensure the pre-encoded model input batch matches the expected schema."""
    if df.columns.tolist() != expected_model_columns:
        raise ValueError("Model-ready columns do not match the expected stable schema.")

    invalid_string_cols = [
        col for col in df.select_dtypes(include=["object", "string", "category"]).columns
        if col not in CATEGORICAL_COLS
    ]
    if invalid_string_cols:
        raise ValueError(
            "Unexpected non-numeric columns remain in model input: "
            f"{invalid_string_cols}"
        )

    if not set(df[TARGET_COL].dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError("target must remain binary {0,1} after preprocessing.")


def initialize_validation_state(expected_model_columns: list[str]) -> dict[str, Any]:
    """Initialize aggregate validation counters for batch processing."""
    return {
        "total_rows": 0,
        "class_distribution": pd.Series(0, index=pd.Index([0, 1], dtype="int64"), dtype="int64"),
        "missing_counts": pd.Series(0, index=pd.Index(expected_model_columns, dtype="object"), dtype="int64"),
    }


def update_validation_state(state: dict[str, Any], model_df: pd.DataFrame, expected_model_columns: list[str]) -> None:
    """Accumulate row counts, class counts, and missingness across batches."""
    state["total_rows"] += len(model_df)
    batch_target = model_df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0).astype("int64")
    state["class_distribution"] = state["class_distribution"].add(batch_target, fill_value=0).astype("int64")
    batch_missing = model_df.isna().sum().reindex(expected_model_columns, fill_value=0).astype("int64")
    state["missing_counts"] = state["missing_counts"].add(batch_missing, fill_value=0).astype("int64")


def finalize_validation_state(state: dict[str, Any], expected_model_columns: list[str]) -> dict[str, Any]:
    """Summarize preprocessing validation metrics after all batches finish."""
    total_rows = state["total_rows"]
    missing_counts = state["missing_counts"]
    missing_only = missing_counts[missing_counts > 0]
    missing_summary = (
        pd.DataFrame(
            {
                "missing_count": missing_only,
                "missing_pct": ((missing_only / total_rows) * 100).round(2),
            }
        )
        .sort_values(["missing_count", "missing_pct"], ascending=False)
        if total_rows > 0
        else pd.DataFrame(columns=["missing_count", "missing_pct"])
    )

    return {
        "class_distribution": state["class_distribution"],
        "missing_summary": missing_summary,
        "shape": (total_rows, len(expected_model_columns)),
    }


def iter_master_batches(input_path: Path, batch_size: int) -> Any:
    """Yield master-table batches from parquet or CSV input."""
    validate_master_schema(get_master_schema_columns(input_path))

    if input_path.suffix.lower() == ".parquet":
        parquet_file = open_parquet_file(input_path)
        return parquet_file.iter_batches(batch_size=batch_size)
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, chunksize=batch_size, low_memory=False)

    raise ValueError(f"Unsupported input format: {input_path.suffix}. Use .parquet or .csv")


def write_model_batch(
    model_df: pd.DataFrame,
    parquet_writer: pq.ParquetWriter | None,
    *,
    csv_header: bool,
    output_parquet_path: Path,
    output_csv_path: Path,
) -> tuple[pq.ParquetWriter, bool]:
    """Append one encoded batch to the model-ready parquet and CSV outputs."""
    table = pa.Table.from_pandas(model_df, preserve_index=False)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_parquet_path, table.schema)
    parquet_writer.write_table(table)

    model_df.to_csv(output_csv_path, mode="w" if csv_header else "a", header=csv_header, index=False)
    return parquet_writer, False


def preprocess_master_table(
    input_path: Path = MASTER_PARQUET,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_parquet_path: Path = MODEL_READY_PARQUET,
    output_csv_path: Path = MODEL_READY_CSV,
    expected_model_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build the full model-ready dataset and return preprocessing validation metrics."""
    output_parquet_path = Path(output_parquet_path)
    output_csv_path = Path(output_csv_path)

    category_levels = infer_category_levels(input_path, batch_size=batch_size)
    resolved_expected_model_columns = expected_model_columns or get_expected_model_columns(category_levels)
    batch_iter = iter_master_batches(input_path, batch_size=batch_size)
    validation_state = initialize_validation_state(resolved_expected_model_columns)

    if output_parquet_path.exists():
        output_parquet_path.unlink()
    if output_csv_path.exists():
        output_csv_path.unlink()

    parquet_writer: pq.ParquetWriter | None = None
    csv_header = True
    input_label = input_path.suffix.lower().lstrip(".") or "file"

    for batch_idx, batch in enumerate(batch_iter, start=1):
        log(f"Processing {input_label} batch {batch_idx:,}...")

        master_batch = batch.to_pandas(types_mapper=pd.ArrowDtype) if hasattr(batch, "to_pandas") else batch.copy()
        master_batch = standardize_missing_values(master_batch)
        master_batch = validate_and_cast_master_batch(master_batch)
        master_batch = handle_missing_data(master_batch)
        master_batch = apply_category_dtypes(master_batch, category_levels)
        model_batch = build_model_input_batch(master_batch, resolved_expected_model_columns)
        validate_model_batch(model_batch, resolved_expected_model_columns)
        update_validation_state(validation_state, model_batch, resolved_expected_model_columns)
        parquet_writer, csv_header = write_model_batch(
            model_batch,
            parquet_writer,
            csv_header=csv_header,
            output_parquet_path=output_parquet_path,
            output_csv_path=output_csv_path,
        )

        del batch, master_batch, model_batch
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()
    gc.collect()

    return finalize_validation_state(validation_state, resolved_expected_model_columns)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model-ready dataset generation."""
    parser = argparse.ArgumentParser(description="Prepare master table for modeling.")
    parser.add_argument("--input-path", type=Path, default=MASTER_PARQUET, help="Path to master_table.parquet or master_table.csv")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to process per batch",
    )
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing pipeline from the command line."""
    args = parse_args()

    validation = preprocess_master_table(args.input_path, batch_size=args.batch_size)

    log("Model input built successfully.")
    log(f"Saved parquet to: {MODEL_READY_PARQUET}")
    log(f"Saved CSV to: {MODEL_READY_CSV}")
    log(f"Shape: {validation['shape']}")
    log("Class distribution:")
    log(validation["class_distribution"].to_string())
    log("Remaining missingness summary:")
    if validation["missing_summary"].empty:
        log("No missing values remain.")
    else:
        log(validation["missing_summary"].to_string())


if __name__ == "__main__":
    main()
