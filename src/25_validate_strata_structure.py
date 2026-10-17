
from pathlib import Path
import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "indonesia_2023.parquet"
)
df = pl.read_parquet(data_path)

print("=" * 70)
print("VALIDATE WBES STRATA STRUCTURE")
print("=" * 70)

# 1. Find all columns related to panel, strata, and weights
print("\n1. RELATED COLUMN NAMES")

keywords = ("panel", "strat", "weight", "wstrict", "wmedian", "wweak")

for name in df.columns:
    if any(keyword in name.lower() for keyword in keywords):
        print(name)

# 2. Check identifier uniqueness
print("\n2. IDENTIFIER CHECK")

for name in ["id", "idstd"]:
    if name in df.columns:
        result = df.select(
            pl.len().alias("rows"),
            pl.col(name).null_count().alias("nulls"),
            pl.col(name).n_unique().alias("unique_values"),
        )
        print(f"\n{name}")
        print(result)

# 3. Check the number of distinct strata and combinations
print("\n3. STRATA AND SAMPLING COMBINATIONS")

if "strata" in df.columns:
    print("Distinct strata:", df["strata"].n_unique())

    print("\nObservations per strata (frequency distribution):")
    print(
        df.group_by("strata")
        .agg(pl.len().alias("n"))
        .group_by("n")
        .agg(pl.len().alias("number_of_strata"))
        .sort("n")
    )

combination_columns = ["a4a", "a2", "a6a"]

if all(name in df.columns for name in combination_columns):
    combinations = df.select(combination_columns).unique()

    print(
        "\nDistinct combinations of a4a, a2, a6a:",
        combinations.height,
    )

    if "strata" in df.columns:
        strata_combinations = (
            df.group_by("strata")
            .agg(
                pl.struct(combination_columns)
                .n_unique()
                .alias("combination_count")
            )
        )

        print("\nNumber of strata mapping to multiple combinations:")
        print(
            strata_combinations.filter(
                pl.col("combination_count") > 1
            )
        )

        combination_strata = (
            df.group_by(combination_columns)
            .agg(
                pl.col("strata").n_unique().alias("strata_count")
            )
        )

        print("\nCombinations associated with multiple strata:")
        print(
            combination_strata.filter(
                pl.col("strata_count") > 1
            )
        )

# 4. Inspect possible panel and stratification code columns
print("\n4. POSSIBLE STRATIFICATION CODE DISTRIBUTIONS")

for name in [
    "stratificationpanelcode",
    "stratificationregioncode",
    "stratificationsizecode",
    "stratificationsectorcode",
    "q_version",
]:
    if name in df.columns:
        print(f"\n{name}")
        print(
            df.group_by(name)
            .agg(pl.len().alias("observations"))
            .sort(name)
        )

print("\nValidation completed.")
