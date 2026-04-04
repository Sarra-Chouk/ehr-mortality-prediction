import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "data" / "processed" / "master_table_model_ready.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "splits"
DEFAULT_IMPUTED_OUTPUT_DIR = ROOT / "data" / "processed" / "splits_imputed"

RANDOM_STATE = 42
TARGET_COL = "target"
GROUP_COL = "subject_id"
REFERENCE_COLS = ["subject_id", "hadm_id", "admittime"]

TRAIN_SIZE = 0.70
TEST_SIZE = 0.15
DEPLOYMENT_SIZE = 0.15


def validate_split_integrity(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
    group_col: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    deployment_df: pd.DataFrame,
    train_ids: set,
    test_ids: set,
    deployment_ids: set,
) -> None:
    if train_ids & test_ids:
        raise ValueError("Overlap detected between train and test patient IDs.")
    if train_ids & deployment_ids:
        raise ValueError("Overlap detected between train and deployment patient IDs.")
    if test_ids & deployment_ids:
        raise ValueError("Overlap detected between test and deployment patient IDs.")

    assigned_row_count = len(train_df) + len(test_df) + len(deployment_df)
    if assigned_row_count != len(df):
        raise ValueError(
            "Split row counts do not sum to the original dataset size: "
            f"{assigned_row_count} != {len(df)}"
        )

    assigned_patient_count = len(train_ids) + len(test_ids) + len(deployment_ids)
    if assigned_patient_count != len(patient_df):
        raise ValueError(
            "Split patient counts do not sum to the full patient set: "
            f"{assigned_patient_count} != {len(patient_df)}"
        )

    assigned_patients = train_ids | test_ids | deployment_ids
    all_patients = set(patient_df[group_col])
    if assigned_patients != all_patients:
        raise ValueError("Not all patients were assigned exactly once across splits.")

    assigned_index = train_df.index.union(test_df.index).union(deployment_df.index)
    if len(assigned_index) != len(df):
        raise ValueError("Some rows were duplicated or omitted across splits.")
    if set(assigned_index.tolist()) != set(df.index.tolist()):
        raise ValueError("All original rows were not assigned exactly once across splits.")


def get_numeric_feature_columns(df: pd.DataFrame, target_col: str, reference_cols: list[str]) -> list[str]:
    exclude_cols = set(reference_cols + [target_col, "_original_order"])
    return [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def temporal_patient_median_impute(
    fit_full_df: pd.DataFrame,
    apply_full_df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """
    Impute numeric feature columns in apply_full_df using:
    1) median of PREVIOUS visits of the same patient only
    2) fallback to global median computed from fit_full_df only

    This avoids:
    - future leakage
    - test/deployment leakage into fallback statistics
    """
    fit_df = fit_full_df.copy()
    app_df = apply_full_df.copy()

    global_medians = fit_df[numeric_cols].median(numeric_only=True)

    sort_cols = ["subject_id", "admittime"]
    if "hadm_id" in app_df.columns:
        sort_cols.append("hadm_id")

    app_df = app_df.sort_values(sort_cols).reset_index(drop=True)

    for col in numeric_cols:
        prev_patient_median = (
            app_df.groupby("subject_id")[col]
            .transform(lambda s: s.shift(1).expanding().median())
        )
        app_df[col] = app_df[col].fillna(prev_patient_median)
        app_df[col] = app_df[col].fillna(global_medians[col])

    return app_df


def save_split_files(
    split_df: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    target_col: str,
    reference_cols: list[str],
) -> None:
    feature_drop_cols = [target_col, *[col for col in reference_cols if col in split_df.columns]]
    X_split = split_df.drop(columns=feature_drop_cols, errors="ignore")
    y_split = split_df[[target_col]]

    X_split.to_csv(output_dir / f"X_{split_name}.csv", index=False)
    y_split.to_csv(output_dir / f"y_{split_name}.csv", index=False)
    split_df.to_csv(output_dir / f"{split_name}_full.csv", index=False)


def split_data(
    input_path: str,
    output_dir: str,
    imputed_output_dir: str,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    random_state: int = RANDOM_STATE,
) -> None:
    """
    Create:
    1) raw grouped splits for native-missing-value models
    2) imputed grouped splits for models that cannot handle missing values

    Imputation logic:
    - previous visits only within the same patient
    - fallback to TRAIN global median only
    """

    if abs(TRAIN_SIZE + TEST_SIZE + DEPLOYMENT_SIZE - 1.0) > 1e-8:
        raise ValueError("TRAIN_SIZE + TEST_SIZE + DEPLOYMENT_SIZE must sum to 1.0")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    imputed_output_dir = Path(imputed_output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    imputed_output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataset.")

    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")
    if df[group_col].isna().any():
        raise ValueError(f"Group column '{group_col}' contains missing values.")

    if "admittime" not in df.columns:
        raise ValueError("Column 'admittime' is required for temporal imputation logic.")

    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    if df["admittime"].isna().any():
        raise ValueError("Column 'admittime' contains invalid datetime values.")

    patient_df = (
        df.groupby(group_col, as_index=False)[target_col]
        .max()
        .rename(columns={target_col: "patient_target"})
    )

    n_positive_patients = int(patient_df["patient_target"].sum())
    if n_positive_patients < 3:
        raise ValueError("Not enough positive patients to create train/test/deployment splits safely.")

    remaining_size = TEST_SIZE + DEPLOYMENT_SIZE

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=remaining_size,
        stratify=patient_df["patient_target"],
        random_state=random_state,
    )

    deployment_relative_size = DEPLOYMENT_SIZE / (TEST_SIZE + DEPLOYMENT_SIZE)

    test_patients, deployment_patients = train_test_split(
        temp_patients,
        test_size=deployment_relative_size,
        stratify=temp_patients["patient_target"],
        random_state=random_state,
    )

    train_ids = set(train_patients[group_col])
    test_ids = set(test_patients[group_col])
    deployment_ids = set(deployment_patients[group_col])

    train_df = df[df[group_col].isin(train_ids)].copy()
    test_df = df[df[group_col].isin(test_ids)].copy()
    deployment_df = df[df[group_col].isin(deployment_ids)].copy()

    validate_split_integrity(
        df=df,
        patient_df=patient_df,
        group_col=group_col,
        train_df=train_df,
        test_df=test_df,
        deployment_df=deployment_df,
        train_ids=train_ids,
        test_ids=test_ids,
        deployment_ids=deployment_ids,
    )

    train_df["_original_order"] = np.arange(len(train_df))
    test_df["_original_order"] = np.arange(len(test_df))
    deployment_df["_original_order"] = np.arange(len(deployment_df))

    save_split_files(train_df, "train", output_dir, target_col, REFERENCE_COLS)
    save_split_files(test_df, "test", output_dir, target_col, REFERENCE_COLS)
    save_split_files(deployment_df, "deployment", output_dir, target_col, REFERENCE_COLS)

    numeric_feature_cols = get_numeric_feature_columns(train_df, target_col, REFERENCE_COLS)

    train_imputed = temporal_patient_median_impute(
        fit_full_df=train_df,
        apply_full_df=train_df,
        numeric_cols=numeric_feature_cols,
    )
    test_imputed = temporal_patient_median_impute(
        fit_full_df=train_df,
        apply_full_df=test_df,
        numeric_cols=numeric_feature_cols,
    )
    deployment_imputed = temporal_patient_median_impute(
        fit_full_df=train_df,
        apply_full_df=deployment_df,
        numeric_cols=numeric_feature_cols,
    )

    train_imputed = train_imputed.sort_values("_original_order").reset_index(drop=True)
    test_imputed = test_imputed.sort_values("_original_order").reset_index(drop=True)
    deployment_imputed = deployment_imputed.sort_values("_original_order").reset_index(drop=True)

    # Final safety check for imputed splits
    train_X_check = train_imputed.drop(columns=[target_col, *[c for c in REFERENCE_COLS if c in train_imputed.columns], "_original_order"], errors="ignore")
    test_X_check = test_imputed.drop(columns=[target_col, *[c for c in REFERENCE_COLS if c in test_imputed.columns], "_original_order"], errors="ignore")
    deployment_X_check = deployment_imputed.drop(columns=[target_col, *[c for c in REFERENCE_COLS if c in deployment_imputed.columns], "_original_order"], errors="ignore")

    if train_X_check.isna().sum().sum() > 0:
        raise ValueError("NaNs remain in imputed training features.")
    if test_X_check.isna().sum().sum() > 0:
        raise ValueError("NaNs remain in imputed test features.")
    if deployment_X_check.isna().sum().sum() > 0:
        raise ValueError("NaNs remain in imputed deployment features.")

    save_split_files(
        train_imputed.drop(columns=["_original_order"], errors="ignore"),
        "train",
        imputed_output_dir,
        target_col,
        REFERENCE_COLS,
    )
    save_split_files(
        test_imputed.drop(columns=["_original_order"], errors="ignore"),
        "test",
        imputed_output_dir,
        target_col,
        REFERENCE_COLS,
    )
    save_split_files(
        deployment_imputed.drop(columns=["_original_order"], errors="ignore"),
        "deployment",
        imputed_output_dir,
        target_col,
        REFERENCE_COLS,
    )

    metadata = {
        "input_file": str(input_path),
        "target_column": target_col,
        "group_column": group_col,
        "random_state": random_state,
        "split_sizes_requested": {
            "train": TRAIN_SIZE,
            "test": TEST_SIZE,
            "deployment": DEPLOYMENT_SIZE,
        },
        "output_directories": {
            "raw_splits": str(output_dir),
            "imputed_splits": str(imputed_output_dir),
        },
        "row_counts": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "deployment": int(len(deployment_df)),
        },
        "patient_counts": {
            "total": int(len(patient_df)),
            "train": int(len(train_patients)),
            "test": int(len(test_patients)),
            "deployment": int(len(deployment_patients)),
        },
        "row_level_positive_rate": {
            "full": float(df[target_col].mean()),
            "train": float(train_df[target_col].mean()),
            "test": float(test_df[target_col].mean()),
            "deployment": float(deployment_df[target_col].mean()),
        },
        "patient_level_positive_rate": {
            "full": float(patient_df["patient_target"].mean()),
            "train": float(train_patients["patient_target"].mean()),
            "test": float(test_patients["patient_target"].mean()),
            "deployment": float(deployment_patients["patient_target"].mean()),
        },
        "imputation": {
            "method": "previous_visit_patient_median_with_train_global_fallback",
            "numeric_feature_count": int(len(numeric_feature_cols)),
        },
    }

    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(imputed_output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Grouped data split completed successfully.\n")

    print("Patient counts:")
    print(f"Total patients:      {len(patient_df):,}")
    print(f"Train patients:      {len(train_patients):,}")
    print(f"Test patients:       {len(test_patients):,}")
    print(f"Deployment patients: {len(deployment_patients):,}\n")

    print("Row counts:")
    print(f"Total rows:          {len(df):,}")
    print(f"Train rows:          {len(train_df):,}")
    print(f"Test rows:           {len(test_df):,}")
    print(f"Deployment rows:     {len(deployment_df):,}\n")

    print("Row-level positive rate:")
    print(f"Full:                {df[target_col].mean():.4%}")
    print(f"Train:               {train_df[target_col].mean():.4%}")
    print(f"Test:                {test_df[target_col].mean():.4%}")
    print(f"Deployment:          {deployment_df[target_col].mean():.4%}\n")

    print("Patient-level positive rate (any positive visit):")
    print(f"Full:                {patient_df['patient_target'].mean():.4%}")
    print(f"Train:               {train_patients['patient_target'].mean():.4%}")
    print(f"Test:                {test_patients['patient_target'].mean():.4%}")
    print(f"Deployment:          {deployment_patients['patient_target'].mean():.4%}\n")

    print("Saved raw splits to:")
    print(output_dir)
    print("\nSaved imputed splits to:")
    print(imputed_output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped raw and imputed train/test/deployment splits.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the model-ready master table CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where raw split files will be written.",
    )
    parser.add_argument(
        "--imputed-output-dir",
        type=Path,
        default=DEFAULT_IMPUTED_OUTPUT_DIR,
        help="Directory where imputed split files will be written.",
    )
    parser.add_argument("--target-col", default=TARGET_COL, help="Binary target column.")
    parser.add_argument("--group-col", default=GROUP_COL, help="Grouping column used to avoid patient leakage.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducible splits.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_data(
        input_path=str(args.input_path),
        output_dir=str(args.output_dir),
        imputed_output_dir=str(args.imputed_output_dir),
        target_col=args.target_col,
        group_col=args.group_col,
        random_state=args.random_state,
    )