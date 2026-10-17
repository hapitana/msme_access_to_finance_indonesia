
from pathlib import Path
import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "indonesia_2023.parquet"
)
df = pl.read_parquet(data_path)

print("=" * 70)
print("VALIDATE STRATA WITH PANEL")
print("=" * 70)

# Officially documented sampling dimensions
dimensions = [
    "a4a",
    "a2",
    "a6a",
    "stratificationpanelcode",
]

print("\n1. NUMBER OF DISTINCT COMBINATIONS")

for columns in [
    ["a4a", "a2", "a6a"],
    dimensions,
    [
        "stratificationsectorcode",
        "stratificationregioncode",
        "stratificationsizecode",
        "stratificationpanelcode",
    ],
]:
    if all(c in df.columns for c in columns):
        n = df.select(columns).unique().height
        print(f"{columns}: {n}")

print("\n2. DOES EACH STRATUM MAP TO ONE COMBINATION?")

if all(c in df.columns for c in dimensions + ["strata"]):
    mapping = (
        df.group_by("strata")
        .agg(
            pl.struct(dimensions)
            .n_unique()
            .alias("combination_count")
        )
    )

    violations = mapping.filter(
        pl.col("combination_count") > 1
    )

    print("Strata with multiple combinations:", violations.height)
    print(violations.head(20))

print("\n3. DOES EACH COMBINATION MAP TO ONE STRATUM?")

if all(c in df.columns for c in dimensions + ["strata"]):
    mapping = (
        df.group_by(dimensions)
        .agg(
            pl.col("strata")
            .n_unique()
            .alias("strata_count")
        )
    )

    violations = mapping.filter(
        pl.col("strata_count") > 1
    )

    print("Combinations associated with multiple strata:", violations.height)
    print(violations.head(20))

print("\n4. COMPARE SAMPLING DIMENSION CODES")

pairs = [
    ("a4a", "stratificationsectorcode"),
    ("a2", "stratificationregioncode"),
    ("a6a", "stratificationsizecode"),
]

for original, coded in pairs:
    if original in df.columns and coded in df.columns:
        comparison = df.select(
            (pl.col(original) == pl.col(coded))
            .sum()
            .alias("equal_rows"),
            (pl.col(original) != pl.col(coded))
            .sum()
            .alias("different_rows"),
        )
        print(f"\n{original} vs {coded}")
        print(comparison)

print("\nValidation completed.")
