from pathlib import Path
import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "indonesia_2023.parquet"
)
df = pl.read_parquet(data_path)

print("=" * 70)
print("SURVEY DESIGN INSPECTION")
print("=" * 70)

print(f"Dataset rows: {df.height}")
print(f"Dataset columns: {df.width}")

candidates = [
    "idstd",
    "id",
    "strata",
    "panel",
    "a4a",
    "a2",
    "a6a",
    "a6b",
    "a3a",
    "wstrict",
    "wstrong",
    "wmedian",
    "wweak",
    "wstrict_BR",
    "wstrong_BR",
    "wmedian_BR",
    "wweak_BR",
]

print("\n1. VARIABLE AVAILABILITY")

for name in candidates:
    if name in df.columns:
        print(f"FOUND   {name}: {df.schema[name]}")
    else:
        print(f"MISSING {name}")

print("\n2. WEIGHT SUMMARY")

weight_candidates = [
    "wstrict",
    "wstrong",
    "wmedian",
    "wweak",
    "wstrict_BR",
    "wstrong_BR",
    "wmedian_BR",
    "wweak_BR",
]

for name in weight_candidates:
    if name not in df.columns:
        continue

    summary = df.select(
        pl.len().alias("rows"),
        pl.col(name).null_count().alias("nulls"),
        pl.col(name).n_unique().alias("unique_values"),
        pl.col(name).min().alias("minimum"),
        pl.col(name).max().alias("maximum"),
        pl.col(name).mean().alias("mean"),
    )

    print(f"\n{name}")
    print(summary)

print("\n3. STRATA AND PANEL COUNTS")

for name in ["strata", "panel", "a4a", "a2", "a6a", "a6b"]:
    if name not in df.columns:
        continue

    print(f"\n{name}:")
    print(
        df.group_by(name)
        .agg(pl.len().alias("observations"))
        .sort(name)
    )

print("\nInspection completed.")
