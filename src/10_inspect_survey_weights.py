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

if not stata_path.exists():
    raise FileNotFoundError(f"Stata dataset not found: {stata_path}")

if not parquet_path.exists():
    raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

# 1. Read Stata metadata without loading the entire dataset
_, metadata = pyreadstat.read_dta(
    str(stata_path),
    metadataonly=True
)

weight_columns = [
    "wstrict",
    "wmedian",
    "wweak",
    "wstrict_BR",
    "wmedian_BR",
    "wweak_BR",
]

print("SURVEY WEIGHT VARIABLE LABELS")
print("=" * 60)

for column in weight_columns:
    label = metadata.column_names_to_labels.get(
        column,
        "Label not available"
    )
    print(f"{column}: {label}")

# 2. Check column availability in Parquet
df = pl.read_parquet(parquet_path)

print("\nSURVEY WEIGHT DATA QUALITY")
print("=" * 60)

for column in weight_columns:
    if column not in df.columns:
        print(f"\n{column}: column not found")
        continue

    series = df.get_column(column)

    print(f"\nVariable: {column}")
    print(f"Non-null values: {series.len() - series.null_count():,}")
    print(f"Null values: {series.null_count():,}")
    print(f"Minimum: {series.min()}")
    print(f"Maximum: {series.max()}")
    print(f"Mean: {series.mean()}")

print("\nInspection completed.")
