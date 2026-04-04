from __future__ import annotations

import argparse
import gc
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
HOSP_DIR = DATA_DIR / "hosp"
OUTPUT_DIR = DATA_DIR / "processed"
STAGING_DIR = OUTPUT_DIR / "master_table_staging"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = OUTPUT_DIR / "master_table.csv"
MASTER_PARQUET = OUTPUT_DIR / "master_table.parquet"

DEFAULT_CHUNKSIZE = 1_000_000
DEFAULT_NUM_BUCKETS = 128

ADMISSIONS_USECOLS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "edregtime",
    "edouttime",
    "admission_type",
    "marital_status",
    "race",
    "hospital_expire_flag",
]
PATIENTS_USECOLS = ["subject_id", "gender", "anchor_age", "anchor_year"]

ICU_KEYWORDS = [
    "icu",
    "medical intensive care unit",
    "surgical intensive care unit",
    "micu",
    "sicu",
    "tsicu",
    "nicu",
    "ccu",
    "cardiac care unit",
    "coronary care unit",
    "neuro intensive care unit",
]
ICU_REGEX = "|".join(re.escape(k) for k in ICU_KEYWORDS)


def log(message: str) -> None:
    print(message, flush=True)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_csv_chunks(
    path: Path,
    *,
    usecols: list[str],
    dtype: dict[str, Any] | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.io.parsers.TextFileReader:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(
        path,
        low_memory=False,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
    )


def to_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .replace({"<na>": pd.NA, "nan": pd.NA, "none": pd.NA, "": pd.NA})
    )


def normalize_category_value(val: Any) -> Any:
    if pd.isna(val):
        return pd.NA
    return str(val).strip().lower()


def round_to_int_or_nan(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return int(round(float(value)))


def round_to_2_or_nan(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return round(float(value), 2)


def map_gender(val: Any) -> Any:
    val = normalize_category_value(val)
    if val == "m":
        return 1
    if val == "f":
        return 0
    return np.nan


def clean_admission_type(val: Any) -> Any:
    val = normalize_category_value(val)
    if pd.isna(val):
        return pd.NA
    if "elective" in val:
        return "elective"
    if "emerg" in val:
        return "emergency"
    if "urgent" in val:
        return "urgent"
    return "other"


def clean_marital_status(val: Any) -> Any:
    val = normalize_category_value(val)
    if pd.isna(val):
        return pd.NA
    if "married" in val:
        return "married"
    if "single" in val:
        return "single"
    if "divorc" in val:
        return "divorced"
    if "widow" in val:
        return "widowed"
    return "other"


def clean_race(val: Any) -> Any:
    val = normalize_category_value(val)
    if pd.isna(val):
        return pd.NA
    if "white" in val:
        return "white"
    if "black" in val:
        return "black"
    if "asian" in val:
        return "asian"
    if "hispanic" in val or "latino" in val:
        return "hispanic_latino"
    return "other"


def parse_bp_string(value: Any) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()
    match = re.match(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$", text)
    if not match:
        return np.nan, np.nan

    return float(match.group(1)), float(match.group(2))


def first_valid_time_cols(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in candidates:
        if col in df.columns:
            result = result.fillna(df[col])
    return result


def load_subject_filter() -> set[int] | None:
    path = DATA_DIR / "demo_subject_id.csv"
    if not path.exists():
        return None
    demo = safe_read_csv(path, usecols=["subject_id"])
    return set(pd.to_numeric(demo["subject_id"], errors="coerce").dropna().astype(int).tolist())


def filter_subjects(df: pd.DataFrame, subject_ids: set[int] | None) -> pd.DataFrame:
    if subject_ids is None or "subject_id" not in df.columns:
        return df
    return df[df["subject_id"].isin(subject_ids)]


def compute_subject_bucket(series: pd.Series, num_buckets: int) -> pd.Series:
    return (series.astype("int64") % num_buckets).astype("int16")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_partitioned_chunk(df: pd.DataFrame, dataset_dir: Path) -> None:
    if df.empty:
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(dataset_dir),
        partition_cols=["bucket"],
        existing_data_behavior="overwrite_or_ignore",
    )


def read_bucket_dataset(dataset_dir: Path, bucket: int) -> pd.DataFrame:
    if not dataset_dir.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(dataset_dir, filters=[("bucket", "==", bucket)])
    except (FileNotFoundError, ValueError, pa.ArrowInvalid):
        return pd.DataFrame()


def release_memory(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()


def load_core_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_filter = load_subject_filter()

    log("Stage A: loading core visit tables (admissions, patients)")

    admissions = safe_read_csv(
        HOSP_DIR / "admissions.csv",
        usecols=ADMISSIONS_USECOLS,
        dtype={
            "subject_id": "Int32",
            "hadm_id": "Int32",
            "admission_type": "string",
            "marital_status": "string",
            "race": "string",
            "hospital_expire_flag": "float32",
        },
    )
    admissions = filter_subjects(admissions, subject_filter)
    admissions = to_datetime(admissions, ["admittime", "dischtime", "edregtime", "edouttime"])
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime"])
    admissions["admission_type"] = admissions["admission_type"].map(clean_admission_type).astype("category")
    admissions["marital_status"] = admissions["marital_status"].map(clean_marital_status).astype("category")
    admissions["race"] = admissions["race"].map(clean_race).astype("category")
    admissions["hospital_expire_flag"] = pd.to_numeric(admissions["hospital_expire_flag"], errors="coerce").fillna(0).astype("int8")

    patients = safe_read_csv(
        HOSP_DIR / "patients.csv",
        usecols=PATIENTS_USECOLS,
        dtype={
            "subject_id": "Int32",
            "gender": "string",
            "anchor_age": "float32",
            "anchor_year": "float32",
        },
    )
    patients = filter_subjects(patients, subject_filter)
    patients = patients.dropna(subset=["subject_id"])
    patients["gender"] = patients["gender"].map(map_gender).astype("float32")
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce").astype("float32")
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"], errors="coerce").astype("float32")

    return admissions, patients


def build_base_visits(admissions: pd.DataFrame, patients: pd.DataFrame, num_buckets: int) -> pd.DataFrame:
    patient_cols = ["subject_id", "gender", "anchor_age", "anchor_year"]
    visits = admissions.merge(
        patients[patient_cols].drop_duplicates(subset=["subject_id"]),
        on="subject_id",
        how="left",
    )

    adm_year = visits["admittime"].dt.year.astype("float32")
    visits["age"] = visits["anchor_age"]
    anchor_year_mask = visits["anchor_year"].notna()
    visits.loc[anchor_year_mask, "age"] = (
        visits.loc[anchor_year_mask, "anchor_age"]
        + (adm_year[anchor_year_mask] - visits.loc[anchor_year_mask, "anchor_year"])
    )
    visits["age"] = visits["age"].apply(round_to_int_or_nan).astype("float32")

    visits["target"] = pd.to_numeric(visits["hospital_expire_flag"], errors="coerce").fillna(0).astype("int8")

    visits["ed_los_hours"] = (
        (visits["edouttime"] - visits["edregtime"]).dt.total_seconds() / 3600.0
    ).astype("float32")
    no_ed_mask = visits["edregtime"].isna() & visits["edouttime"].isna()
    visits.loc[no_ed_mask, "ed_los_hours"] = 0.0
    visits["ed_los_hours"] = visits["ed_los_hours"].apply(round_to_2_or_nan).astype("float32")

    visits["came_from_ed"] = (
        visits["edregtime"].notna() | visits["edouttime"].notna()
    ).astype("int8")
    visits["admission_hour"] = visits["admittime"].dt.hour.astype("int8")
    visits["admission_day_of_week"] = visits["admittime"].dt.dayofweek.astype("int8")

    visits = visits.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
    visits["bucket"] = compute_subject_bucket(visits["subject_id"], num_buckets)

    grouped = visits.groupby("subject_id", sort=False)
    visit_gap_days = grouped["admittime"].diff().dt.total_seconds() / 86400.0
    los_days = (visits["dischtime"] - visits["admittime"]).dt.total_seconds() / 86400.0

    visits["num_prev_visits"] = grouped.cumcount().astype("int32")
    visits["time_since_last_visit_days"] = visit_gap_days.fillna(0).astype("float32")
    visits["avg_time_between_visits_days"] = (
        visit_gap_days.groupby(visits["subject_id"])
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0)
        .astype("float32")
    )
    visits["avg_prev_los"] = (
        los_days.groupby(visits["subject_id"])
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0)
        .astype("float32")
    )
    visits["max_prev_los"] = (
        los_days.groupby(visits["subject_id"])
        .transform(lambda s: s.shift(1).expanding().max())
        .fillna(0)
        .astype("float32")
    )
    visits["last_los"] = los_days.groupby(visits["subject_id"]).shift(1).fillna(0).astype("float32")

    prev_ed_mean = visits["ed_los_hours"].groupby(visits["subject_id"]).transform(
        lambda s: s.shift(1).expanding().mean()
    )
    prev_ed_max = visits["ed_los_hours"].groupby(visits["subject_id"]).transform(
        lambda s: s.shift(1).expanding().max()
    )
    visits["avg_prev_ed_los"] = prev_ed_mean.astype("float32")
    visits["max_prev_ed_los"] = prev_ed_max.astype("float32")
    visits.loc[visits["num_prev_visits"] == 0, "avg_prev_ed_los"] = 0
    visits.loc[visits["num_prev_visits"] == 0, "max_prev_ed_los"] = 0

    visits = visits.drop(columns=["anchor_age", "anchor_year", "hospital_expire_flag"])
    return visits


def build_hadm_to_times(admissions: pd.DataFrame) -> pd.DataFrame:
    cols = ["subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "edouttime"]
    hadm_times = admissions[cols].drop_duplicates(subset=["hadm_id"]).copy()
    hadm_times["los_days"] = (
        (hadm_times["dischtime"] - hadm_times["admittime"]).dt.total_seconds() / 86400.0
    ).astype("float32")
    hadm_times["ed_los_hours"] = (
        (hadm_times["edouttime"] - hadm_times["edregtime"]).dt.total_seconds() / 3600.0
    ).astype("float32")
    return hadm_times


def stage_table_chunks(
    *,
    name: str,
    path: Path,
    usecols: list[str],
    dtype: dict[str, Any] | None,
    transform_fn: Callable[[pd.DataFrame, set[int], set[int], int], pd.DataFrame],
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
    chunksize: int,
) -> Path:
    dataset_dir = STAGING_DIR / name
    reset_dir(dataset_dir)
    log(f"Stage C: processing {name}")

    total_input_rows = 0
    total_output_rows = 0

    for chunk_idx, chunk in enumerate(
        read_csv_chunks(path, usecols=usecols, dtype=dtype, chunksize=chunksize),
        start=1,
    ):
        total_input_rows += len(chunk)
        reduced = transform_fn(chunk, subject_ids, hadm_ids, num_buckets)
        if not reduced.empty:
            total_output_rows += len(reduced)
            write_partitioned_chunk(reduced, dataset_dir)

        log(
            f"  {name}: chunk {chunk_idx:,} processed "
            f"(input_rows={len(chunk):,}, kept_rows={len(reduced):,})"
        )
        release_memory(chunk, reduced)

    log(
        f"Completed {name}: total_input_rows={total_input_rows:,}, "
        f"total_kept_rows={total_output_rows:,}"
    )
    return dataset_dir


def transform_diagnoses_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].isin(hadm_ids)]
    chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
    if chunk.empty:
        return chunk
    chunk["icd_code"] = chunk["icd_code"].astype("string").str.strip()
    chunk = chunk[["subject_id", "hadm_id", "icd_code"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_procedures_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].isin(hadm_ids)]
    chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
    if chunk.empty:
        return chunk
    chunk["icd_code"] = chunk["icd_code"].astype("string").str.strip()
    chunk = chunk[["subject_id", "hadm_id", "icd_code"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_drgcodes_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].isin(hadm_ids)]
    chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
    if chunk.empty:
        return chunk
    chunk["drg_severity"] = pd.to_numeric(chunk["drg_severity"], errors="coerce").astype("float32")
    chunk["drg_mortality"] = pd.to_numeric(chunk["drg_mortality"], errors="coerce").astype("float32")
    chunk = chunk[["subject_id", "hadm_id", "drg_severity", "drg_mortality"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_labevents_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids)]
    if chunk.empty:
        return chunk

    chunk = to_datetime(chunk, ["charttime"])
    chunk = chunk.dropna(subset=["subject_id", "charttime"])
    if chunk.empty:
        return chunk

    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce").astype("float32")
    chunk["ref_range_lower"] = pd.to_numeric(chunk["ref_range_lower"], errors="coerce").astype("float32")
    chunk["ref_range_upper"] = pd.to_numeric(chunk["ref_range_upper"], errors="coerce").astype("float32")

    flag_norm = normalize_text_series(chunk["flag"])
    priority_norm = normalize_text_series(chunk["priority"])
    chunk["flag"] = flag_norm.isin({"abnormal", "high", "low", "h", "l", "hh", "ll", "a"}).astype("int8")
    chunk["priority_stat"] = priority_norm.str.contains("stat", na=False).astype("int8")

    denom = chunk["ref_range_upper"] - chunk["ref_range_lower"]
    valid_mask = (
        chunk["valuenum"].notna()
        & chunk["ref_range_lower"].notna()
        & chunk["ref_range_upper"].notna()
        & denom.gt(0)
    )
    chunk["norm_lab"] = np.nan
    chunk.loc[valid_mask, "norm_lab"] = (
        (chunk.loc[valid_mask, "valuenum"] - chunk.loc[valid_mask, "ref_range_lower"]) / denom.loc[valid_mask]
    ).astype("float32")

    chunk = chunk[["subject_id", "hadm_id", "charttime", "flag", "priority_stat", "norm_lab"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_microbiology_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids)]
    if chunk.empty:
        return chunk

    chunk = to_datetime(chunk, ["chartdate", "charttime", "storedate", "storetime"])
    chunk["event_time"] = first_valid_time_cols(chunk, ["charttime", "chartdate", "storetime", "storedate"])
    chunk = chunk.dropna(subset=["subject_id", "event_time"])
    if chunk.empty:
        return chunk

    chunk["org_name"] = normalize_text_series(chunk["org_name"])
    chunk["interpretation"] = normalize_text_series(chunk["interpretation"])
    chunk["is_resistant"] = chunk["interpretation"].isin(["r", "resistant"]).astype("int8")
    chunk = chunk[["subject_id", "hadm_id", "event_time", "org_name", "is_resistant"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_omr_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids)]
    if chunk.empty:
        return chunk

    chunk = to_datetime(chunk, ["chartdate"])
    chunk = chunk.dropna(subset=["subject_id", "chartdate"])
    if chunk.empty:
        return chunk

    chunk["result_name"] = normalize_text_series(chunk["result_name"])
    chunk["result_value"] = chunk["result_value"].astype("string").str.strip()
    chunk = chunk[["subject_id", "chartdate", "result_name", "result_value"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_pharmacy_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].isin(hadm_ids)]
    chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
    if chunk.empty:
        return chunk

    chunk = to_datetime(chunk, ["starttime", "entertime", "verifiedtime"])
    chunk["event_time"] = first_valid_time_cols(chunk, ["starttime", "entertime", "verifiedtime"])
    chunk["med_name"] = normalize_text_series(chunk["medication"])
    chunk = chunk[["subject_id", "hadm_id", "event_time", "med_name"]]
    chunk = chunk.dropna(subset=["event_time"])
    if chunk.empty:
        return chunk
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_prescriptions_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].isin(hadm_ids)]
    chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
    if chunk.empty:
        return chunk

    chunk = to_datetime(chunk, ["starttime", "stoptime"])
    chunk["event_time"] = first_valid_time_cols(chunk, ["starttime", "stoptime"])
    chunk["med_name"] = normalize_text_series(chunk["drug"])
    chunk = chunk[["subject_id", "hadm_id", "event_time", "med_name"]]
    chunk = chunk.dropna(subset=["event_time"])
    if chunk.empty:
        return chunk
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def transform_transfers_chunk(
    chunk: pd.DataFrame,
    subject_ids: set[int],
    hadm_ids: set[int],
    num_buckets: int,
) -> pd.DataFrame:
    chunk = chunk[chunk["subject_id"].isin(subject_ids)]
    if chunk.empty:
        return chunk

    chunk = to_datetime(chunk, ["intime", "outtime"])
    chunk["event_time"] = first_valid_time_cols(chunk, ["intime", "outtime"])
    chunk = chunk.dropna(subset=["subject_id", "event_time"])
    if chunk.empty:
        return chunk

    careunit_norm = normalize_text_series(chunk["careunit"])
    chunk["is_icu"] = careunit_norm.str.contains(ICU_REGEX, regex=True, na=False).astype("int8")
    chunk = chunk[["subject_id", "hadm_id", "event_time", "is_icu"]]
    chunk["bucket"] = compute_subject_bucket(chunk["subject_id"], num_buckets)
    return chunk


def stage_reduced_tables(subject_ids: set[int], hadm_ids: set[int], num_buckets: int, chunksize: int) -> dict[str, Path]:
    reset_dir(STAGING_DIR)

    datasets = {
        "diagnoses": stage_table_chunks(
            name="diagnoses",
            path=HOSP_DIR / "diagnoses_icd.csv",
            usecols=["subject_id", "hadm_id", "icd_code"],
            dtype={"subject_id": "Int32", "hadm_id": "Int32", "icd_code": "string"},
            transform_fn=transform_diagnoses_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "procedures": stage_table_chunks(
            name="procedures",
            path=HOSP_DIR / "procedures_icd.csv",
            usecols=["subject_id", "hadm_id", "icd_code"],
            dtype={"subject_id": "Int32", "hadm_id": "Int32", "icd_code": "string"},
            transform_fn=transform_procedures_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "drgcodes": stage_table_chunks(
            name="drgcodes",
            path=HOSP_DIR / "drgcodes.csv",
            usecols=["subject_id", "hadm_id", "drg_severity", "drg_mortality"],
            dtype={
                "subject_id": "Int32",
                "hadm_id": "Int32",
                "drg_severity": "float32",
                "drg_mortality": "float32",
            },
            transform_fn=transform_drgcodes_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "labevents": stage_table_chunks(
            name="labevents",
            path=HOSP_DIR / "labevents.csv",
            usecols=[
                "subject_id",
                "hadm_id",
                "charttime",
                "valuenum",
                "ref_range_lower",
                "ref_range_upper",
                "flag",
                "priority",
            ],
            dtype={
                "subject_id": "Int32",
                "hadm_id": "Int32",
                "valuenum": "float32",
                "ref_range_lower": "float32",
                "ref_range_upper": "float32",
                "flag": "string",
                "priority": "string",
            },
            transform_fn=transform_labevents_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "microbiology": stage_table_chunks(
            name="microbiology",
            path=HOSP_DIR / "microbiologyevents.csv",
            usecols=["subject_id", "hadm_id", "chartdate", "charttime", "storedate", "storetime", "org_name", "interpretation"],
            dtype={
                "subject_id": "Int32",
                "hadm_id": "Int32",
                "org_name": "string",
                "interpretation": "string",
            },
            transform_fn=transform_microbiology_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "omr": stage_table_chunks(
            name="omr",
            path=HOSP_DIR / "omr.csv",
            usecols=["subject_id", "chartdate", "result_name", "result_value"],
            dtype={"subject_id": "Int32", "result_name": "string", "result_value": "string"},
            transform_fn=transform_omr_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "pharmacy": stage_table_chunks(
            name="pharmacy",
            path=HOSP_DIR / "pharmacy.csv",
            usecols=["subject_id", "hadm_id", "starttime", "entertime", "verifiedtime", "medication"],
            dtype={"subject_id": "Int32", "hadm_id": "Int32", "medication": "string"},
            transform_fn=transform_pharmacy_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "prescriptions": stage_table_chunks(
            name="prescriptions",
            path=HOSP_DIR / "prescriptions.csv",
            usecols=["subject_id", "hadm_id", "starttime", "stoptime", "drug"],
            dtype={"subject_id": "Int32", "hadm_id": "Int32", "drug": "string"},
            transform_fn=transform_prescriptions_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
        "transfers": stage_table_chunks(
            name="transfers",
            path=HOSP_DIR / "transfers.csv",
            usecols=["subject_id", "hadm_id", "intime", "outtime", "careunit"],
            dtype={"subject_id": "Int32", "hadm_id": "Int32", "careunit": "string"},
            transform_fn=transform_transfers_chunk,
            subject_ids=subject_ids,
            hadm_ids=hadm_ids,
            num_buckets=num_buckets,
            chunksize=chunksize,
        ),
    }
    gc.collect()
    return datasets


def get_subject_frame(groups: dict[int, pd.DataFrame], subject_id: int, columns: list[str]) -> pd.DataFrame:
    frame = groups.get(subject_id)
    if frame is None:
        return pd.DataFrame(columns=columns)
    return frame


def group_bucket_by_subject(df: pd.DataFrame, sort_cols: list[str] | None = None) -> dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    if sort_cols is not None:
        df = df.sort_values(sort_cols)
    return {int(subject_id): group for subject_id, group in df.groupby("subject_id", sort=False)}


def build_feature_row(
    row: pd.Series,
    patient_visits: pd.DataFrame,
    hadm_times: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    drgcodes: pd.DataFrame,
    labevents: pd.DataFrame,
    microbiology: pd.DataFrame,
    pharmacy: pd.DataFrame,
    prescriptions: pd.DataFrame,
    transfers: pd.DataFrame,
    omr: pd.DataFrame,
) -> dict[str, Any]:
    admittime = row["admittime"]

    prev_visits = patient_visits[patient_visits["admittime"] < admittime]
    prev_hadm_ids = set(prev_visits["hadm_id"].dropna().astype(int).tolist())

    out: dict[str, Any] = {
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "admittime": row["admittime"],
        "age": row["age"],
        "gender": row["gender"],
        "admission_type": row["admission_type"],
        "marital_status": row["marital_status"],
        "race": row["race"],
        "ed_los_hours": row["ed_los_hours"],
        "came_from_ed": row["came_from_ed"],
        "admission_hour": row["admission_hour"],
        "admission_day_of_week": row["admission_day_of_week"],
        "num_prev_visits": row["num_prev_visits"],
        "time_since_last_visit_days": row["time_since_last_visit_days"],
        "avg_time_between_visits_days": row["avg_time_between_visits_days"],
        "avg_prev_los": row["avg_prev_los"],
        "max_prev_los": row["max_prev_los"],
        "last_los": row["last_los"],
        "avg_prev_ed_los": row["avg_prev_ed_los"],
        "max_prev_ed_los": row["max_prev_ed_los"],
        "target": row["target"],
    }

    prev_dx = diagnoses[diagnoses["hadm_id"].isin(prev_hadm_ids)]
    out["num_prev_diagnoses"] = int(len(prev_dx))
    out["num_unique_diagnoses_icd_codes"] = int(prev_dx["icd_code"].dropna().nunique())

    prev_proc = procedures[procedures["hadm_id"].isin(prev_hadm_ids)]
    out["num_prev_procedures"] = int(len(prev_proc))
    out["num_unique_procedure_icd_codes"] = int(prev_proc["icd_code"].dropna().nunique())

    prev_drg = drgcodes[drgcodes["hadm_id"].isin(prev_hadm_ids)]
    if len(prev_hadm_ids) == 0:
        out["avg_prev_drg_severity"] = 0
        out["max_prev_drg_severity"] = 0
        out["avg_prev_drg_mortality"] = 0
        out["max_prev_drg_mortality"] = 0
    else:
        sev = prev_drg["drg_severity"] if "drg_severity" in prev_drg.columns else pd.Series(dtype=float)
        mort = prev_drg["drg_mortality"] if "drg_mortality" in prev_drg.columns else pd.Series(dtype=float)
        out["avg_prev_drg_severity"] = float(sev.mean()) if not sev.empty and sev.notna().any() else np.nan
        out["max_prev_drg_severity"] = float(sev.max()) if not sev.empty and sev.notna().any() else np.nan
        out["avg_prev_drg_mortality"] = float(mort.mean()) if not mort.empty and mort.notna().any() else np.nan
        out["max_prev_drg_mortality"] = float(mort.max()) if not mort.empty and mort.notna().any() else np.nan

    prev_labs = labevents[labevents["charttime"] < admittime]
    out["num_abnormal_labs"] = int(prev_labs["flag"].sum()) if not prev_labs.empty else 0
    if not prev_labs.empty:
        out["abnormal_lab_ratio"] = float(prev_labs["flag"].sum()) / len(prev_labs)
        out["stat_lab_ratio"] = float(prev_labs["priority_stat"].sum()) / len(prev_labs)
        last_lab = prev_labs.iloc[-1]
        out["last_lab_flag"] = int(last_lab["flag"])
        out["last_norm_lab"] = last_lab["norm_lab"]
    else:
        out["abnormal_lab_ratio"] = 0
        out["stat_lab_ratio"] = 0
        out["last_lab_flag"] = 0
        out["last_norm_lab"] = np.nan

    prev_pharmacy = pharmacy[pharmacy["event_time"] < admittime][["med_name"]]
    prev_rx = prescriptions[prescriptions["event_time"] < admittime][["med_name"]]
    all_meds = pd.concat([prev_pharmacy, prev_rx], ignore_index=True)
    all_meds = all_meds.dropna(subset=["med_name"])
    out["num_prev_medications"] = int(len(all_meds))
    out["num_unique_drugs"] = int(all_meds["med_name"].nunique())

    prev_transfers = transfers[transfers["event_time"] < admittime]
    out["num_prev_transfers"] = int(len(prev_transfers))
    prev_icu_transfers = prev_transfers[prev_transfers["is_icu"] == 1]
    out["num_prev_icu_visits"] = int(prev_icu_transfers["hadm_id"].nunique()) if not prev_icu_transfers.empty else 0

    prev_micro = microbiology[microbiology["event_time"] < admittime]
    out["num_prev_infections"] = int(len(prev_micro))
    out["num_unique_organisms"] = int(prev_micro["org_name"].dropna().nunique())
    out["num_resistant_cases"] = int(prev_micro["is_resistant"].sum()) if not prev_micro.empty else 0

    omr_cutoff = admittime.normalize()
    prev_omr = omr[omr["chartdate"] < omr_cutoff]
    bmi_rows = prev_omr[prev_omr["result_name"].str.contains("bmi", na=False)]
    out["bmi_last"] = pd.to_numeric(bmi_rows["result_value"], errors="coerce").iloc[-1] if not bmi_rows.empty else np.nan
    weight_rows = prev_omr[prev_omr["result_name"].str.contains("weight", na=False)]
    out["weight_last"] = pd.to_numeric(weight_rows["result_value"], errors="coerce").iloc[-1] if not weight_rows.empty else np.nan

    bp_rows = prev_omr[prev_omr["result_name"].str.contains("blood pressure", na=False)]
    if not bp_rows.empty:
        last_bp = bp_rows.iloc[-1]
        systolic, diastolic = parse_bp_string(last_bp["result_value"])
        out["last_systolic"] = systolic
        out["last_diastolic"] = diastolic
        out["last_pulse_pressure"] = systolic - diastolic if pd.notna(systolic) and pd.notna(diastolic) else np.nan
    else:
        out["last_systolic"] = np.nan
        out["last_diastolic"] = np.nan
        out["last_pulse_pressure"] = np.nan

    return out


def validate_master_table(master: pd.DataFrame) -> None:
    if master.empty:
        raise ValueError("Master table is empty.")

    if master.duplicated(subset=["subject_id", "hadm_id"]).any():
        dups = master[master.duplicated(subset=["subject_id", "hadm_id"], keep=False)]
        raise ValueError(
            f"Duplicate (subject_id, hadm_id) rows found in master table. Example:\n{dups.head()}"
        )

    required_cols = ["subject_id", "hadm_id", "target"]
    missing = [c for c in required_cols if c not in master.columns]
    if missing:
        raise ValueError(f"Missing required columns in master table: {missing}")

    if master["target"].isna().any():
        raise ValueError("Target column contains missing values.")

    if not set(master["target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column must be binary {0,1}.")


def finalize_master_table(master: pd.DataFrame) -> pd.DataFrame:
    zero_fill_cols = [
        "time_since_last_visit_days",
        "avg_time_between_visits_days",
        "avg_prev_los",
        "max_prev_los",
        "last_los",
        "abnormal_lab_ratio",
        "stat_lab_ratio",
        "last_lab_flag",
    ]
    for col in zero_fill_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    round2_cols = [
        "ed_los_hours",
        "time_since_last_visit_days",
        "avg_time_between_visits_days",
        "avg_prev_los",
        "max_prev_los",
        "last_los",
        "avg_prev_ed_los",
        "max_prev_ed_los",
        "avg_prev_drg_severity",
        "max_prev_drg_severity",
        "avg_prev_drg_mortality",
        "max_prev_drg_mortality",
        "abnormal_lab_ratio",
        "stat_lab_ratio",
        "last_norm_lab",
    ]
    for col in round2_cols:
        if col in master.columns:
            master[col] = master[col].apply(round_to_2_or_nan)

    int_cols = [
        "age",
        "num_prev_visits",
        "num_prev_diagnoses",
        "num_unique_diagnoses_icd_codes",
        "num_prev_procedures",
        "num_unique_procedure_icd_codes",
        "num_abnormal_labs",
        "last_lab_flag",
        "num_prev_medications",
        "num_unique_drugs",
        "num_prev_transfers",
        "num_prev_icu_visits",
        "num_prev_infections",
        "num_unique_organisms",
        "num_resistant_cases",
        "last_systolic",
        "last_diastolic",
        "last_pulse_pressure",
    ]
    for col in int_cols:
        if col in master.columns:
            mask = master[col].notna()
            master.loc[mask, col] = master.loc[mask, col].apply(lambda x: int(round(float(x))))

    if "target" in master.columns:
        cols = [c for c in master.columns if c != "target"] + ["target"]
        master = master[cols]

    return master


def build_master_table(num_buckets: int = DEFAULT_NUM_BUCKETS, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    admissions, patients = load_core_tables()
    visits = build_base_visits(admissions, patients, num_buckets=num_buckets)
    hadm_times = build_hadm_to_times(admissions)

    subject_ids = set(visits["subject_id"].astype(int).tolist())
    hadm_ids = set(visits["hadm_id"].astype(int).tolist())

    release_memory(patients)
    datasets = stage_reduced_tables(subject_ids, hadm_ids, num_buckets=num_buckets, chunksize=chunksize)
    release_memory(admissions)

    master_parts: list[pd.DataFrame] = []
    total_rows = len(visits)
    processed_rows = 0

    log("Stage D: building final master table from reduced bucketed datasets")
    for bucket in range(num_buckets):
        bucket_visits = visits[visits["bucket"] == bucket].copy()
        if bucket_visits.empty:
            continue

        log(f"  Building bucket {bucket + 1:,}/{num_buckets:,} with {len(bucket_visits):,} visits")

        diagnoses_bucket = read_bucket_dataset(datasets["diagnoses"], bucket)
        procedures_bucket = read_bucket_dataset(datasets["procedures"], bucket)
        drgcodes_bucket = read_bucket_dataset(datasets["drgcodes"], bucket)
        labevents_bucket = read_bucket_dataset(datasets["labevents"], bucket)
        microbiology_bucket = read_bucket_dataset(datasets["microbiology"], bucket)
        omr_bucket = read_bucket_dataset(datasets["omr"], bucket)
        pharmacy_bucket = read_bucket_dataset(datasets["pharmacy"], bucket)
        prescriptions_bucket = read_bucket_dataset(datasets["prescriptions"], bucket)
        transfers_bucket = read_bucket_dataset(datasets["transfers"], bucket)

        diagnoses_groups = group_bucket_by_subject(diagnoses_bucket, ["subject_id", "hadm_id"])
        procedures_groups = group_bucket_by_subject(procedures_bucket, ["subject_id", "hadm_id"])
        drgcodes_groups = group_bucket_by_subject(drgcodes_bucket, ["subject_id", "hadm_id"])
        labevents_groups = group_bucket_by_subject(labevents_bucket, ["subject_id", "charttime"])
        microbiology_groups = group_bucket_by_subject(microbiology_bucket, ["subject_id", "event_time"])
        omr_groups = group_bucket_by_subject(omr_bucket, ["subject_id", "chartdate"])
        pharmacy_groups = group_bucket_by_subject(pharmacy_bucket, ["subject_id", "event_time"])
        prescriptions_groups = group_bucket_by_subject(prescriptions_bucket, ["subject_id", "event_time"])
        transfers_groups = group_bucket_by_subject(transfers_bucket, ["subject_id", "event_time"])

        feature_rows: list[dict[str, Any]] = []
        for subject_id, patient_visits in bucket_visits.groupby("subject_id", sort=False):
            subject_id_int = int(subject_id)
            subject_diagnoses = get_subject_frame(diagnoses_groups, subject_id_int, ["subject_id", "hadm_id", "icd_code"])
            subject_procedures = get_subject_frame(procedures_groups, subject_id_int, ["subject_id", "hadm_id", "icd_code"])
            subject_drgcodes = get_subject_frame(drgcodes_groups, subject_id_int, ["subject_id", "hadm_id", "drg_severity", "drg_mortality"])
            subject_labevents = get_subject_frame(labevents_groups, subject_id_int, ["subject_id", "hadm_id", "charttime", "flag", "priority_stat", "norm_lab"])
            subject_microbiology = get_subject_frame(microbiology_groups, subject_id_int, ["subject_id", "hadm_id", "event_time", "org_name", "is_resistant"])
            subject_omr = get_subject_frame(omr_groups, subject_id_int, ["subject_id", "chartdate", "result_name", "result_value"])
            subject_pharmacy = get_subject_frame(pharmacy_groups, subject_id_int, ["subject_id", "hadm_id", "event_time", "med_name"])
            subject_prescriptions = get_subject_frame(prescriptions_groups, subject_id_int, ["subject_id", "hadm_id", "event_time", "med_name"])
            subject_transfers = get_subject_frame(transfers_groups, subject_id_int, ["subject_id", "hadm_id", "event_time", "is_icu"])

            for _, row in patient_visits.iterrows():
                feature_rows.append(
                    build_feature_row(
                        row=row,
                        patient_visits=patient_visits,
                        hadm_times=hadm_times,
                        diagnoses=subject_diagnoses,
                        procedures=subject_procedures,
                        drgcodes=subject_drgcodes,
                        labevents=subject_labevents,
                        microbiology=subject_microbiology,
                        pharmacy=subject_pharmacy,
                        prescriptions=subject_prescriptions,
                        transfers=subject_transfers,
                        omr=subject_omr,
                    )
                )
                processed_rows += 1
                if processed_rows % 5_000 == 0:
                    log(f"    Master rows built: {processed_rows:,}/{total_rows:,}")

        master_parts.append(pd.DataFrame(feature_rows))
        release_memory(
            bucket_visits,
            diagnoses_bucket,
            procedures_bucket,
            drgcodes_bucket,
            labevents_bucket,
            microbiology_bucket,
            omr_bucket,
            pharmacy_bucket,
            prescriptions_bucket,
            transfers_bucket,
            diagnoses_groups,
            procedures_groups,
            drgcodes_groups,
            labevents_groups,
            microbiology_groups,
            omr_groups,
            pharmacy_groups,
            prescriptions_groups,
            transfers_groups,
            feature_rows,
        )

    master = pd.concat(master_parts, ignore_index=True)
    master = master.drop(columns=["bucket"], errors="ignore")
    master = finalize_master_table(master)
    validate_master_table(master)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe master table for mortality prediction.")
    parser.add_argument("--save-csv", action="store_true", help="Save CSV output")
    parser.add_argument("--save-parquet", action="store_true", help="Save Parquet output")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="CSV chunksize for staging large tables")
    parser.add_argument("--num-buckets", type=int, default=DEFAULT_NUM_BUCKETS, help="Number of subject buckets for reduced parquet staging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    save_csv = args.save_csv or (not args.save_csv and not args.save_parquet)
    save_parquet = args.save_parquet or (not args.save_csv and not args.save_parquet)

    master = build_master_table(num_buckets=args.num_buckets, chunksize=args.chunksize)
    master["admittime"] = pd.to_datetime(master["admittime"], errors="coerce")

    log("\nMaster table built successfully.")
    log(f"Shape: {master.shape}")
    log(f"Target rate: {master['target'].mean():.4f}")

    if save_csv:
        master.to_csv(MASTER_CSV, index=False)
        log(f"Saved CSV to: {MASTER_CSV}")

    if save_parquet:
        master.to_parquet(MASTER_PARQUET, index=False)
        log(f"Saved Parquet to: {MASTER_PARQUET}")


if __name__ == "__main__":
    main()
