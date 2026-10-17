
from pathlib import Path
import polars as pl
import pyreadstat

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

print("Reading Stata file into Polars...")

df, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="polars"
)

print("\nDATASET LOADED SUCCESSFULLY")
print("Polars DataFrame:", isinstance(df, pl.DataFrame))
print("Rows:", df.height)
print("Columns:", df.width)

print("\nFIRST 5 ROWS")
print(df.head(5))

print("\nSELECTED FINANCE VARIABLES")
finance_variables = [
    "k82",
    "k162",
    "k20a1",
    "k17",
    "k30",
    "k6",
    "k3a",
    "k3bc",
    "k3e",
    "k3f",
]

available_variables = [
    name for name in finance_variables
    if name in df.columns
]

print(df.select(available_variables).head(10))

print("\nDATA TYPES")
print(df.schema)
