from pathlib import Path

import polars as pl
import pyreadstat


project_root = Path(__file__).resolve().parents[1]

stata_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

parquet_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

TARGET_VARIABLES = [
    "idstd",
    "id",
    "strata",
    "panel",
    "a4a",
    "a2",
    "a6a",
    "a6b",
]

if not stata_path.exists():
    raise FileNotFoundError(f"Stata file not found: {stata_path}")

if not parquet_path.exists():
    raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

_, metadata = pyreadstat.read_dta(
    str(stata_path),
    metadataonly=True,
)

df = pl.read_parquet(parquet_path)
stata_n_rows = metadata.number_rows
parquet_n_rows = df.height

print(f"Stata observations: {stata_n_rows:,}")
print(f"Parquet observations: {parquet_n_rows:,}")

if stata_n_rows != parquet_n_rows:
    print("WARNING: Observation counts do not match.")
else:
    print("Observation counts match.")
stata_columns = set(metadata.column_names)
parquet_columns = set(df.columns)

print("=" * 72)
print("SAMPLING VARIABLE METADATA AUDIT")
print("=" * 72)

for name in TARGET_VARIABLES:
    print(f"\nVariable: {name}")
    print("-" * 50)

    print(f"In Stata: {'YES' if name in stata_columns else 'NO'}")
    print(f"In Parquet: {'YES' if name in parquet_columns else 'NO'}")

    if name in stata_columns:
        label = metadata.column_names_to_labels.get(
            name,
            "Label not available",
        )
        print(f"Label: {label}")

        value_labels = metadata.variable_value_labels.get(name, {})
        if value_labels:
            print("Value labels:")
            for value, description in value_labels.items():
                print(f"  {value}: {description}")

    if name in parquet_columns:
        series = df.get_column(name)
        print(f"Unique values: {series.n_unique():,}")
        print(f"Null values: {series.null_count():,}")

print("\nMetadata audit completed.")
