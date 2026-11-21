from pathlib import Path
import polars as pl

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"
output_dir = processed_dir

input_file = processed_dir / "indonesia_2023.parquet"

if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pl.read_parquet(input_file)

required_columns = [
    "idstd", "b4", "b4a", "b5", "l1",
    "d3a", "d3b", "d3c"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("=" * 70)
print("WBES INDONESIA 2023: OBSERVATION-LEVEL RELATIONSHIP AUDIT")
print("=" * 70)
print(f"Input file: {input_file.name}")
print(f"Rows: {df.height}")
print(f"Columns: {df.width}")

if df["idstd"].n_unique() != df.height:
    raise ValueError("Duplicate observation identifiers detected.")

numeric_variables = [
    "b5", "l1", "b4a", "d3a", "d3b", "d3c"
]

summary_records = []

for variable in numeric_variables:
    values = df[variable].cast(pl.Float64, strict=False)

    record = {
        "variable": variable,
        "non_null": values.count(),
        "null": values.null_count(),
        "negative_values": values.filter(values < 0).len(),
        "minus_9": values.filter(values == -9).len(),
        "minus_7": values.filter(values == -7).len(),
        "minimum": values.min(),
        "maximum": values.max(),
        "unique_values": values.n_unique(),
    }

    if variable in ["b4a", "d3a", "d3b", "d3c"]:
        record["valid_0_to_100"] = values.filter(
            values.is_between(0, 100)
        ).len()
        record["outside_0_to_100"] = values.filter(
            values.is_not_null()
            & ~values.is_between(0, 100)
        ).len()
    else:
        record["valid_0_to_100"] = None
        record["outside_0_to_100"] = None

    summary_records.append(record)

variable_summary = pl.DataFrame(summary_records)

variable_summary.write_csv(
    output_dir / "observation_level_variable_summary.csv"
)

print("\n1. VARIABLE SUMMARY")
print(variable_summary)

b4a_status = (
    pl.when(pl.col("b4a").is_null())
    .then(pl.lit("system_missing"))
    .when(pl.col("b4a") == -9)
    .then(pl.lit("dont_know"))
    .when(pl.col("b4a") == -7)
    .then(pl.lit("not_applicable"))
    .when(pl.col("b4a").is_between(0, 100))
    .then(pl.lit("valid_percentage"))
    .otherwise(pl.lit("other_value"))
)

ownership_df = df.with_columns(
    b4a_status.alias("b4a_status")
)

ownership_crosstab = (
    ownership_df
    .group_by(["b4", "b4a_status"])
    .len()
    .sort(["b4", "b4a_status"])
)

ownership_crosstab.write_csv(
    output_dir / "female_ownership_cross_tabulation.csv"
)

print("\n2. FEMALE OWNERSHIP CROSS-TABULATION")
print(ownership_crosstab)

ownership_summary = (
    ownership_df
    .group_by(["b4", "b4a_status"])
    .agg(
        pl.len().alias("observations"),
        pl.col("b4a").filter(
            pl.col("b4a").is_between(0, 100)
        ).mean().alias("mean_female_ownership"),
        pl.col("b4a").filter(
            pl.col("b4a").is_between(0, 100)
        ).min().alias("minimum_female_ownership"),
        pl.col("b4a").filter(
            pl.col("b4a").is_between(0, 100)
        ).max().alias("maximum_female_ownership"),
    )
    .sort(["b4", "b4a_status"])
)

ownership_summary.write_csv(
    output_dir / "female_ownership_consistency_summary.csv"
)

print("\n3. FEMALE OWNERSHIP CONSISTENCY SUMMARY")
print(ownership_summary)

sales_columns = ["d3a", "d3b", "d3c"]

sales_df = df.with_columns([
    pl.when(pl.col(column).is_between(0, 100))
    .then(pl.col(column).cast(pl.Float64))
    .otherwise(None)
    .alias(f"{column}_valid")
    for column in sales_columns
])

sales_df = sales_df.with_columns(
    pl.sum_horizontal([
        pl.col(f"{column}_valid")
        for column in sales_columns
    ]).alias("sales_share_sum")
)

jointly_valid = (
    pl.col("d3a_valid").is_not_null()
    & pl.col("d3b_valid").is_not_null()
    & pl.col("d3c_valid").is_not_null()
)

sales_audit = (
    sales_df
    .filter(jointly_valid)
    .select(
        pl.len().alias("jointly_valid_observations"),
        pl.col("sales_share_sum").min().alias("minimum_share_sum"),
        pl.col("sales_share_sum").max().alias("maximum_share_sum"),
        pl.col("sales_share_sum").mean().alias("mean_share_sum"),
        pl.col("sales_share_sum").median().alias("median_share_sum"),
        (pl.col("sales_share_sum") - 100).abs()
        .le(0.5)
        .sum()
        .alias("observations_with_sum_approximately_100"),
        (pl.col("sales_share_sum") - 100).abs()
        .gt(0.5)
        .sum()
        .alias("observations_with_sum_not_100"),
    )
)

sales_audit.write_csv(
    output_dir / "sales_composition_audit.csv"
)

sales_status = (
    sales_df
    .select(
        pl.len().alias("total_observations"),
        jointly_valid.sum().alias("jointly_valid_observations"),
        (~jointly_valid).sum().alias("not_jointly_valid"),
    )
)

sales_status.write_csv(
    output_dir / "sales_composition_status.csv"
)

print("\n4. SALES COMPOSITION AUDIT")
print(sales_audit)

print("\n5. SALES COMPOSITION COVERAGE")
print(sales_status)

print("\nAudit completed.")
print(f"Output directory: {output_dir}")