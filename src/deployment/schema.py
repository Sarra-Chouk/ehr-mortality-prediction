from __future__ import annotations

"""Schema validation helpers for model-input deployment contracts."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCHEMA_VERSION = "3.0.0"
OPTIONAL_INPUT_COLUMNS = ("subject_id", "hadm_id", "admittime", "target", "_original_order")


class SchemaValidationError(ValueError):
    """Raised when model-input data does not satisfy the packaged schema contract."""


@dataclass(frozen=True)
class ModelInputSchema:
    """Describe the pre-encoded feature contract expected by the deployment pipeline."""

    version: str
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = OPTIONAL_INPUT_COLUMNS

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the ordered full feature list used by the packaged pipeline."""
        return (*self.numeric_feature_columns, *self.categorical_feature_columns)

    @classmethod
    def from_feature_columns(
        cls,
        *,
        numeric_feature_columns: list[str] | tuple[str, ...],
        categorical_feature_columns: list[str] | tuple[str, ...],
        version: str = DEFAULT_SCHEMA_VERSION,
        optional_columns: tuple[str, ...] = OPTIONAL_INPUT_COLUMNS,
    ) -> "ModelInputSchema":
        """Build a schema from the ordered numeric and categorical feature lists."""
        return cls(
            version=version,
            numeric_feature_columns=tuple(str(col) for col in numeric_feature_columns),
            categorical_feature_columns=tuple(str(col) for col in categorical_feature_columns),
            optional_columns=tuple(optional_columns),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the schema to metadata-friendly primitives."""
        return {
            "version": self.version,
            "numeric_feature_columns": list(self.numeric_feature_columns),
            "categorical_feature_columns": list(self.categorical_feature_columns),
            "feature_columns": list(self.feature_columns),
            "optional_columns": list(self.optional_columns),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "ModelInputSchema":
        """Rebuild a schema from saved deployment metadata."""
        schema_metadata = metadata.get("schema", metadata)
        if "numeric_feature_columns" in schema_metadata and "categorical_feature_columns" in schema_metadata:
            return cls(
                version=str(schema_metadata["version"]),
                numeric_feature_columns=tuple(schema_metadata["numeric_feature_columns"]),
                categorical_feature_columns=tuple(schema_metadata["categorical_feature_columns"]),
                optional_columns=tuple(schema_metadata.get("optional_columns", OPTIONAL_INPUT_COLUMNS)),
            )

        feature_columns = tuple(schema_metadata["feature_columns"])
        return cls(
            version=str(schema_metadata["version"]),
            numeric_feature_columns=feature_columns,
            categorical_feature_columns=tuple(),
            optional_columns=tuple(schema_metadata.get("optional_columns", OPTIONAL_INPUT_COLUMNS)),
        )

    def validate_required_columns(self, df: pd.DataFrame) -> None:
        """Ensure all required model-input columns are present."""
        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            raise SchemaValidationError(
                "Model input is missing required feature columns: "
                f"{missing_columns}"
            )

    def validate_unexpected_columns(self, df: pd.DataFrame) -> None:
        """Reject unexpected columns outside the feature set and allowed extras."""
        allowed_columns = set(self.feature_columns).union(self.optional_columns)
        unexpected_columns = [col for col in df.columns if col not in allowed_columns]
        if unexpected_columns:
            raise SchemaValidationError(
                "Model input contains unexpected columns: "
                f"{unexpected_columns}"
            )

    def prepare_features_for_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, reorder, and coerce pre-encoded model-input features before scoring."""
        self.validate_required_columns(df)
        self.validate_unexpected_columns(df)

        feature_df = df.loc[:, list(self.feature_columns)].copy()
        invalid_numeric_columns: list[str] = []

        for col in self.numeric_feature_columns:
            original_values = feature_df[col]
            coerced_values = pd.to_numeric(original_values, errors="coerce")
            introduced_nans = coerced_values.isna() & original_values.notna()
            if introduced_nans.any():
                invalid_numeric_columns.append(col)
            feature_df[col] = coerced_values

        if invalid_numeric_columns:
            raise SchemaValidationError(
                "Model input contains non-numeric values in numeric feature columns: "
                f"{sorted(set(invalid_numeric_columns))}"
            )

        for col in self.categorical_feature_columns:
            feature_df[col] = feature_df[col].astype("object").where(pd.notna(feature_df[col]), np.nan)

        return feature_df
