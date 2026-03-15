from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ColumnType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.5
CATEGORICAL_MAX_UNIQUE = 100
TEXT_MIN_AVG_LENGTH = 50
TOP_VALUES_COUNT = 10


class CategoricalProfile(BaseModel):
    profile_type: Literal["categorical"] = "categorical"
    unique_count: int = Field(..., description="Number of unique values")
    top_values: list[dict[str, Any]] = Field(
        ...,
        description="Top N values with counts",
        max_length=TOP_VALUES_COUNT
    )
    null_count: int = Field(default=0, description="Number of null/NaN values")
    null_percentage: float = Field(default=0.0, description="Percentage of nulls")

    @field_validator("top_values", mode="before")
    @classmethod
    def ensure_serializable(cls, v) -> list:
        if not isinstance(v, list):
            return v
        result = []
        for item in v:
            if not isinstance(item, dict):
                result.append(item)
                continue
            serialized = {}
            for key, val in item.items():
                if pd.isna(val):
                    serialized[key] = None
                elif isinstance(val, (np.integer, np.floating)):
                    serialized[key] = val.item()
                else:
                    serialized[key] = val
            result.append(serialized)
        return result


class NumericalProfile(BaseModel):
    profile_type: Literal["numerical"] = "numerical"
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    mean: Optional[float] = Field(None, description="Mean value")
    median: Optional[float] = Field(None, description="Median value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")
    percentile_25: Optional[float] = Field(None, description="25th percentile")
    percentile_75: Optional[float] = Field(None, description="75th percentile")
    null_count: int = Field(default=0, description="Number of null/NaN values")
    null_percentage: float = Field(default=0.0, description="Percentage of nulls")
    has_negative: bool = Field(default=False, description="Contains negative values")
    has_decimals: bool = Field(default=False, description="Contains decimal values")

    @model_validator(mode="before")
    @classmethod
    def convert_numpy_types(cls, data: dict) -> dict:
        for key, val in data.items():
            if isinstance(val, (np.integer, np.floating)):
                data[key] = None if pd.isna(val) else val.item()
            elif pd.isna(val) if not isinstance(val, bool) else False:
                data[key] = None
        return data


class DatetimeProfile(BaseModel):
    profile_type: Literal["datetime"] = "datetime"
    min_date: Optional[str] = Field(None, description="Earliest date (ISO format)")
    max_date: Optional[str] = Field(None, description="Latest date (ISO format)")
    date_range_days: Optional[int] = Field(None, description="Span in days")
    null_count: int = Field(default=0, description="Number of null/NaN values")
    null_percentage: float = Field(default=0.0, description="Percentage of nulls")
    inferred_frequency: Optional[str] = Field(
        None,
        description="Detected frequency (daily, monthly, etc.)"
    )


class BooleanProfile(BaseModel):
    profile_type: Literal["boolean"] = "boolean"
    true_count: int = Field(..., description="Count of True values")
    false_count: int = Field(..., description="Count of False values")
    true_percentage: float = Field(..., description="Percentage of True values")
    null_count: int = Field(default=0, description="Number of null/NaN values")
    null_percentage: float = Field(default=0.0, description="Percentage of nulls")


class TextProfile(BaseModel):
    profile_type: Literal["text"] = "text"
    unique_count: int = Field(..., description="Number of unique values")
    avg_length: float = Field(..., description="Average string length")
    min_length: int = Field(..., description="Minimum string length")
    max_length: int = Field(..., description="Maximum string length")
    sample_values: list[str] = Field(
        ...,
        description="Sample values for context",
        max_length=5
    )
    null_count: int = Field(default=0, description="Number of null/NaN values")
    null_percentage: float = Field(default=0.0, description="Percentage of nulls")


class ColumnSchema(BaseModel):
    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Pandas dtype as string")
    inferred_type: ColumnType = Field(..., description="Semantic type classification")
    profile: Annotated[
        Union[
            CategoricalProfile,
            NumericalProfile,
            DatetimeProfile,
            BooleanProfile,
            TextProfile,
        ],
        Field(discriminator="profile_type")
    ] | None = Field(default=None, description="Type-specific statistical profile")

    model_config = {"use_enum_values": True}


class SchemaProfile(BaseModel):
    file_name: str = Field(..., description="Source file name")
    file_path: str = Field(..., description="Absolute path to source file")
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Profile generation timestamp"
    )
    scanner_version: str = Field(default="1.0.0", description="Scanner version")

    row_count: int = Field(..., description="Total number of rows")
    column_count: int = Field(..., description="Total number of columns")
    memory_usage_mb: float = Field(..., description="DataFrame memory usage in MB")

    grain: Optional[str] = Field(
        default=None,
        description="Data grain definition (e.g., '1 row = 1 transaction'). Set manually."
    )
    grain_hint: Optional[str] = Field(
        default=None,
        description="Auto-detected hint about potential grain columns"
    )

    columns: list[ColumnSchema] = Field(..., description="Schema for each column")

    categorical_columns: list[str] = Field(
        default_factory=list,
        description="List of categorical column names"
    )
    numerical_columns: list[str] = Field(
        default_factory=list,
        description="List of numerical column names"
    )
    datetime_columns: list[str] = Field(
        default_factory=list,
        description="List of datetime column names"
    )

    total_null_cells: int = Field(default=0, description="Total null cells in dataset")
    null_percentage: float = Field(default=0.0, description="Overall null percentage")
    duplicate_row_count: int = Field(default=0, description="Number of duplicate rows")

    model_config = {"use_enum_values": True}


class DataScanner:
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path).resolve()
        self._df: Optional[pd.DataFrame] = None
        self._profile: Optional[SchemaProfile] = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected CSV file, got: {self.file_path.suffix}")

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading CSV: {self.file_path}")

        self._df = pd.read_csv(
            self.file_path,
            low_memory=False,
            na_values=["", "NA", "N/A", "null", "NULL", "None", "NONE", "-"]
        )

        for col in self._df.select_dtypes(include=["object"]).columns:
            self._try_parse_datetime(col)

        logger.info(f"Loaded {len(self._df)} rows, {len(self._df.columns)} columns")
        return self._df

    def _try_parse_datetime(self, column: str) -> None:
        try:
            sample = self._df[column].dropna().head(100)
            if len(sample) == 0:
                return

            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")

            if parsed.notna().sum() / len(sample) > 0.8:
                self._df[column] = pd.to_datetime(
                    self._df[column],
                    errors="coerce",
                    format="mixed"
                )
                logger.debug(f"Converted '{column}' to datetime")
        except Exception:
            pass

    def _infer_column_type(self, column: str) -> ColumnType:
        series = self._df[column]
        dtype = series.dtype

        if dtype == bool or set(series.dropna().unique()).issubset({True, False, 0, 1}):
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({True, False}) or unique_vals.issubset({0, 1}):
                return ColumnType.BOOLEAN

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME

        if pd.api.types.is_numeric_dtype(dtype):
            return ColumnType.NUMERICAL

        if dtype == object:
            non_null = series.dropna()
            if len(non_null) == 0:
                return ColumnType.UNKNOWN

            unique_count = series.nunique()
            total_count = len(series)
            unique_ratio = unique_count / total_count if total_count > 0 else 0

            avg_length = non_null.astype(str).str.len().mean()

            if avg_length > TEXT_MIN_AVG_LENGTH:
                return ColumnType.TEXT

            if unique_count <= CATEGORICAL_MAX_UNIQUE or unique_ratio < CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
                return ColumnType.CATEGORICAL

            return ColumnType.TEXT

        return ColumnType.UNKNOWN

    def _profile_categorical(self, column: str) -> CategoricalProfile:
        series = self._df[column]
        value_counts = series.value_counts(dropna=False).head(TOP_VALUES_COUNT)

        top_values = []
        for value, count in value_counts.items():
            top_values.append({
                "value": value if pd.notna(value) else None,
                "count": int(count),
                "percentage": round(count / len(series) * 100, 2)
            })

        null_count = int(series.isna().sum())

        return CategoricalProfile(
            unique_count=int(series.nunique()),
            top_values=top_values,
            null_count=null_count,
            null_percentage=round(null_count / len(series) * 100, 2)
        )

    def _profile_numerical(self, column: str) -> NumericalProfile:
        series = self._df[column]
        null_count = int(series.isna().sum())
        non_null = series.dropna()

        if len(non_null) == 0:
            return NumericalProfile(
                null_count=null_count,
                null_percentage=round(null_count / len(series) * 100, 2)
            )

        return NumericalProfile(
            min_value=float(non_null.min()),
            max_value=float(non_null.max()),
            mean=round(float(non_null.mean()), 4),
            median=round(float(non_null.median()), 4),
            std_dev=round(float(non_null.std()), 4) if len(non_null) > 1 else None,
            percentile_25=round(float(non_null.quantile(0.25)), 4),
            percentile_75=round(float(non_null.quantile(0.75)), 4),
            null_count=null_count,
            null_percentage=round(null_count / len(series) * 100, 2),
            has_negative=bool((non_null < 0).any()),
            has_decimals=bool((non_null % 1 != 0).any())
        )

    def _profile_datetime(self, column: str) -> DatetimeProfile:
        series = self._df[column]
        null_count = int(series.isna().sum())
        non_null = series.dropna()

        if len(non_null) == 0:
            return DatetimeProfile(
                null_count=null_count,
                null_percentage=round(null_count / len(series) * 100, 2)
            )

        min_date = non_null.min()
        max_date = non_null.max()
        date_range = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None

        inferred_freq = None
        if len(non_null) > 2:
            try:
                freq = pd.infer_freq(non_null.sort_values().head(100))
                if freq:
                    freq_map = {
                        "D": "daily", "B": "business_daily", "W": "weekly",
                        "M": "monthly", "MS": "month_start", "Q": "quarterly",
                        "Y": "yearly", "H": "hourly", "T": "minutely"
                    }
                    inferred_freq = freq_map.get(freq, freq)
            except Exception:
                pass

        return DatetimeProfile(
            min_date=min_date.isoformat() if pd.notna(min_date) else None,
            max_date=max_date.isoformat() if pd.notna(max_date) else None,
            date_range_days=date_range,
            null_count=null_count,
            null_percentage=round(null_count / len(series) * 100, 2),
            inferred_frequency=inferred_freq
        )

    def _profile_boolean(self, column: str) -> BooleanProfile:
        series = self._df[column]
        null_count = int(series.isna().sum())
        non_null = series.dropna()

        true_count = int(((non_null == True) | (non_null == 1)).sum())
        false_count = int(len(non_null) - true_count)

        return BooleanProfile(
            true_count=true_count,
            false_count=false_count,
            true_percentage=round(true_count / len(non_null) * 100, 2) if len(non_null) > 0 else 0.0,
            null_count=null_count,
            null_percentage=round(null_count / len(series) * 100, 2)
        )

    def _profile_text(self, column: str) -> TextProfile:
        series = self._df[column]
        null_count = int(series.isna().sum())
        non_null = series.dropna().astype(str)

        if len(non_null) == 0:
            return TextProfile(
                unique_count=0,
                avg_length=0.0,
                min_length=0,
                max_length=0,
                sample_values=[],
                null_count=null_count,
                null_percentage=round(null_count / len(series) * 100, 2)
            )

        lengths = non_null.str.len()

        return TextProfile(
            unique_count=int(series.nunique()),
            avg_length=round(float(lengths.mean()), 2),
            min_length=int(lengths.min()),
            max_length=int(lengths.max()),
            sample_values=non_null.head(5).tolist(),
            null_count=null_count,
            null_percentage=round(null_count / len(series) * 100, 2)
        )

    def _detect_grain_hint(self) -> Optional[str]:
        hints = []

        for col in self._df.columns:
            series = self._df[col]
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

            if unique_ratio > 0.95:
                col_lower = col.lower()
                if any(id_hint in col_lower for id_hint in ["id", "key", "code", "number", "no"]):
                    hints.append(f"'{col}' appears to be a unique identifier")

        date_cols = [c for c in self._df.columns if pd.api.types.is_datetime64_any_dtype(self._df[c])]
        cat_cols = [c for c in self._df.columns
                    if self._df[c].dtype == object and self._df[c].nunique() < CATEGORICAL_MAX_UNIQUE]

        if date_cols and cat_cols:
            hints.append(f"Possible time-series grain: {date_cols[0]} + dimensions like {cat_cols[:2]}")

        return "; ".join(hints) if hints else None

    def scan(self) -> SchemaProfile:
        if self._df is None:
            self.load_data()

        logger.info("Starting schema profiling...")

        columns = []
        categorical_cols = []
        numerical_cols = []
        datetime_cols = []
        total_nulls = 0

        for col in self._df.columns:
            logger.debug(f"Profiling column: {col}")

            inferred_type = self._infer_column_type(col)

            profile = None
            if inferred_type == ColumnType.CATEGORICAL:
                profile = self._profile_categorical(col)
                categorical_cols.append(col)
            elif inferred_type == ColumnType.NUMERICAL:
                profile = self._profile_numerical(col)
                numerical_cols.append(col)
            elif inferred_type == ColumnType.DATETIME:
                profile = self._profile_datetime(col)
                datetime_cols.append(col)
            elif inferred_type == ColumnType.BOOLEAN:
                profile = self._profile_boolean(col)
            elif inferred_type == ColumnType.TEXT:
                profile = self._profile_text(col)

            total_nulls += int(self._df[col].isna().sum())

            columns.append(ColumnSchema(
                name=col,
                dtype=str(self._df[col].dtype),
                inferred_type=inferred_type,
                profile=profile
            ))

        memory_mb = round(self._df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        duplicate_count = int(self._df.duplicated().sum())
        total_cells = len(self._df) * len(self._df.columns)

        self._profile = SchemaProfile(
            file_name=self.file_path.name,
            file_path=str(self.file_path),
            row_count=len(self._df),
            column_count=len(self._df.columns),
            memory_usage_mb=memory_mb,
            grain_hint=self._detect_grain_hint(),
            columns=columns,
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols,
            datetime_columns=datetime_cols,
            total_null_cells=total_nulls,
            null_percentage=round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0.0,
            duplicate_row_count=duplicate_count
        )

        logger.info("Schema profiling complete")
        return self._profile

    def to_json(self, output_path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        if self._profile is None:
            self.scan()

        json_str = self._profile.model_dump_json(indent=indent)

        if output_path:
            output_path = Path(output_path)
            output_path.write_text(json_str, encoding="utf-8")
            logger.info(f"Schema profile saved to: {output_path}")

        return json_str

    def to_dict(self) -> dict:
        if self._profile is None:
            self.scan()
        return self._profile.model_dump()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan a CSV file and generate a schema profile for LLM consumption.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scanner.py data.csv
    python scanner.py data.csv --output profiles/my_profile.json
    python scanner.py data.csv --grain "1 row = 1 sales transaction"
        """
    )

    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file to analyze"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="schema_profile.json",
        help="Output path for the JSON profile (default: schema_profile.json)"
    )

    parser.add_argument(
        "--grain", "-g",
        type=str,
        default=None,
        help="Manually specify the data grain (e.g., '1 row = 1 order line item')"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        scanner = DataScanner(args.csv_file)
        profile = scanner.scan()

        if args.grain:
            profile.grain = args.grain

        scanner.to_json(args.output)

        print(f"\n{'='*60}")
        print(f"SCHEMA PROFILE GENERATED")
        print(f"{'='*60}")
        print(f"Source:     {profile.file_name}")
        print(f"Rows:       {profile.row_count:,}")
        print(f"Columns:    {profile.column_count}")
        print(f"Memory:     {profile.memory_usage_mb} MB")
        print(f"Nulls:      {profile.null_percentage}%")
        print(f"Duplicates: {profile.duplicate_row_count:,}")
        print(f"{'='*60}")
        print(f"Column Types:")
        print(f"  - Categorical: {len(profile.categorical_columns)}")
        print(f"  - Numerical:   {len(profile.numerical_columns)}")
        print(f"  - Datetime:    {len(profile.datetime_columns)}")
        print(f"{'='*60}")
        if profile.grain_hint:
            print(f"Grain Hint: {profile.grain_hint}")
        if profile.grain:
            print(f"Grain:      {profile.grain}")
        print(f"{'='*60}")
        print(f"Profile saved to: {args.output}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise SystemExit(1)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Error scanning file: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
