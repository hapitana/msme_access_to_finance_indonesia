from pathlib import Path
import polars as pl

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

input_file = processed_dir / "indonesia_2023.parquet"

if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pl.read_parquet(input_file)

required_columns = ["idstd", "b5", "l1"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("=" * 70)
print("WBES INDONESIA 2023: FIRM AGE AND SIZE AUDIT")
print("=" * 70)
print(f"Rows: {df.height}")

for variable in ["b5", "l1"]:
    values = df[variable].cast(pl.Float64, strict=False)

    print("\n" + "=" * 70)
    print(f"VARIABLE: {variable}")
    print("=" * 70)

    print(f"Null values: {values.null_count()}")
    print(f"Unique values: {values.n_unique()}")
    print(f"Minimum: {values.min()}")
    print(f"Maximum: {values.max()}")

    print("\nNegative values:")
    negative_counts = (
        df.filter(pl.col(variable) < 0)
        .group_by(variable)
        .len()
        .sort(variable)
    )
    print(negative_counts)

    print("\nMost frequent values:")
    frequency = (
        df.group_by(variable)
        .len()
        .sort("len", descending=True)
        .head(30)
    )
    print(frequency)

    valid_values = values.filter(values >= 0)

    print("\nNon-negative value distribution:")
    print(f"Count: {valid_values.len()}")
    print(f"Minimum: {valid_values.min()}")
    print(f"Median: {valid_values.median()}")
    print(f"Mean: {valid_values.mean()}")
    print(f"90th percentile: {valid_values.quantile(0.90)}")
    print(f"95th percentile: {valid_values.quantile(0.95)}")
    print(f"99th percentile: {valid_values.quantile(0.99)}")
    print(f"Maximum: {valid_values.max()}")

    if variable == "b5":
        print("\nEstablishment year range:")
        print(
            values.filter(
                values.is_between(1900, 2023)
            ).describe()
        )

    if variable == "l1":
        print("\nEmployment distribution by size:")
        size_summary = (
            df.with_columns(
                pl.when(pl.col("l1") <= 0)
                .then(pl.lit("zero_or_negative"))
                .when(pl.col("l1") < 10)
                .then(pl.lit("1_to_9"))
                .when(pl.col("l1") < 50)
                .then(pl.lit("10_to_49"))
                .when(pl.col("l1") < 250)
                .then(pl.lit("50_to_249"))
                .otherwise(pl.lit("250_or_more"))
                .alias("employment_group")
            )
            .group_by("employment_group")
            .len()
            .sort("employment_group")
        )
        print(size_summary)

print("\nAudit completed.")