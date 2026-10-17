
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "model_dataset.parquet"
)

df = pl.read_parquet(data_path)

print("=" * 60)
print("MODEL FEATURE INSPECTION")
print("=" * 60)

print("\n1. CREDIT STATUS")
print(
    df.group_by("credit_status")
    .agg(pl.len().alias("observations"))
    .sort("credit_status")
)

print("\n2. FIRM SIZE")
print(
    df.group_by("firm_size_code")
    .agg(pl.len().alias("observations"))
    .sort("firm_size_code")
)

print("\n3. REGION")
print(
    df.group_by("region_code")
    .agg(
        pl.len().alias("observations"),
        pl.col("major_obstacle").mean().alias("major_obstacle_rate"),
    )
    .sort("observations")
)

print("\n4. INDUSTRY")
print(
    df.group_by("industry_code")
    .agg(
        pl.len().alias("observations"),
        pl.col("major_obstacle").mean().alias("major_obstacle_rate"),
    )
    .sort("observations", descending=True)
)

print("\nInspection completed.")
