
from pathlib import Path
import polars as pl
import pyreadstat

project_dir = Path(r"D:\portofolio\02_msme_access_to_finance")

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

output_dir = project_dir / "data" / "processed"
output_path = output_dir / "indonesia_2023.parquet"

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

output_dir.mkdir(parents=True, exist_ok=True)

print("Loading Stata dataset into Polars...")

df, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="polars"
)

print(f"Original rows: {df.height}")
print(f"Original columns: {df.width}")

print("\nSaving as Parquet...")

df.write_parquet(output_path)

print(f"Saved to: {output_path}")

print("\nVerifying saved file...")

df_check = pl.read_parquet(output_path)

print(f"Verified rows: {df_check.height}")
print(f"Verified columns: {df_check.width}")
print(f"File size: {output_path.stat().st_size:,} bytes")

if df.shape != df_check.shape:
    raise ValueError("Verification failed: dataset shape changed.")

print("\nSUCCESS: Parquet file verified.")
