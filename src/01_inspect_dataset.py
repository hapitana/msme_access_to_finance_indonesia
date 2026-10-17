from pathlib import Path
import polars as pl
import pyreadstat

# 1. Define dataset location

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

print("WORLD BANK INDONESIA ENTERPRISE DATA")
print("DATA INSPECTION USING POLARS")

# 2. Read Stata dataset

stata_df, metadata = pyreadstat.read_dta(
    str(data_path),
    apply_value_formats=False
)

df = pl.from_pandas(stata_df)

del stata_df

# 3. Display dataset dimensions

print("\n1. DATASET DIMENSIONS")

print(f"Number of rows: {df.height:,}")
print(f"Number of columns: {df.width:,}")

# 4. Display first five rows

print("\n2. FIRST FIVE ROWS")

print(df.head())

# 5. Display column names

print("\n3. COLUMN NAMES")

for column in df.columns:
    print(column)

# 6. Display data types

print("\n4. DATA TYPES")

for column, dtype in zip(df.columns, df.dtypes):
    print(f"{column}: {dtype}")

# 7. Count missing values

print("\n5. MISSING VALUES")

missing_values = df.null_count()

missing_summary = pl.DataFrame({
    "column": df.columns,
    "missing_count": missing_values.row(0),
})

missing_summary = (
    missing_summary
    .filter(pl.col("missing_count") > 0)
    .sort("missing_count", descending=True)
)

if missing_summary.height > 0:
    print(missing_summary)
else:
    print("No null values detected.")

# 8. Display dataset schema

print("\n6. DATASET SCHEMA")

print(df.schema)

# 9. Display Stata variable labels

print("\n7. VARIABLE LABELS")

variable_labels = metadata.column_names_to_labels

for column in df.columns:
    label = variable_labels.get(column, "")
    print(f"{column}: {label}")

# 10. Completion message

print("\nDataset inspection completed.")