from pathlib import Path
import polars as pl

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

input_file = processed_dir / "indonesia_2023.parquet"
df = pl.read_parquet(input_file)

required_columns = ["idstd", "b5", "l1", "a6a"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("=" * 70)
print("WBES INDONESIA 2023: FIRM AGE ANOMALY AND SIZE CATEGORY AUDIT")
print("=" * 70)

print("\n1. ESTABLISHMENT YEAR ANOMALIES")

year_anomalies = (
    df.filter(
        (pl.col("b5") != -9)
        & (
            (pl.col("b5") < 1900)
            | (pl.col("b5") > 2023)
        )
    )
    .select("idstd", "b5", "l1", "a6a")
    .sort("b5")
)

print(f"Anomaly count: {year_anomalies.height}")
print(year_anomalies)

print("\n2. ESTABLISHMENT YEAR FREQUENCY BEFORE 1900")

early_years = (
    df.filter(
        (pl.col("b5") >= 0)
        & (pl.col("b5") < 1900)
    )
    .group_by("b5")
    .len()
    .sort("b5")
)

print(early_years)

print("\n3. LARGEST EMPLOYMENT OBSERVATIONS")

largest_firms = (
    df.select("idstd", "l1", "a6a", "b5")
    .sort("l1", descending=True)
    .head(20)
)

print(largest_firms)

print("\n4. ACTUAL EMPLOYMENT SIZE GROUPS")

df = df.with_columns(
    pl.when(pl.col("l1") < 10)
    .then(pl.lit("1_to_9"))
    .when(pl.col("l1") < 50)
    .then(pl.lit("10_to_49"))
    .when(pl.col("l1") < 250)
    .then(pl.lit("50_to_249"))
    .otherwise(pl.lit("250_or_more"))
    .alias("employment_group")
)

actual_size_summary = (
    df.group_by("employment_group")
    .len()
    .sort("employment_group")
)

print(actual_size_summary)

print("\n5. ACTUAL EMPLOYMENT GROUP BY SAMPLING SIZE")

size_crosstab = (
    df.group_by(["a6a", "employment_group"])
    .len()
    .sort(["a6a", "employment_group"])
)

print(size_crosstab)

size_crosstab.write_csv(
    processed_dir / "employment_vs_sampling_size_crosstab.csv"
)

print("\n6. SAMPLING SIZE FREQUENCY")

sampling_size_frequency = (
    df.group_by("a6a")
    .len()
    .sort("a6a")
)

print(sampling_size_frequency)

print("\nAudit completed.")