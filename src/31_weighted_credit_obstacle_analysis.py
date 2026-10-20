
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path)

required = ["k82", "k30", "wmedian"]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Required columns not found: {missing}")

# Define credit status and finance-obstacle categories.
analysis = df.with_columns(
    pl.when(pl.col("k82").is_in([1, 2, 3]))
    .then(pl.lit("Has credit"))
    .when(pl.col("k82") == 4)
    .then(pl.lit("No credit"))
    .otherwise(None)
    .alias("credit_status"),

    pl.when(pl.col("k30").is_in([0, 1, 2, 3, 4]))
    .then(pl.col("k30"))
    .otherwise(None)
    .alias("obstacle_code"),

    pl.when(pl.col("k30").is_in([3, 4]))
    .then(1)
    .when(pl.col("k30").is_in([0, 1, 2]))
    .then(0)
    .otherwise(None)
    .alias("major_obstacle"),
)

# Use complete cases for credit status and obstacle response.

valid = analysis.filter(
    pl.col("credit_status").is_not_null()
    & pl.col("obstacle_code").is_not_null()
    & pl.col("wmedian").is_not_null()
    & pl.col("wmedian").is_finite()
    & (pl.col("wmedian") > 0)
)

if valid.height == 0:
    raise ValueError("No valid observations remain in the analysis sample.")

expected_groups = {"Has credit", "No credit"}
observed_groups = set(valid.get_column("credit_status").unique().to_list())

if observed_groups != expected_groups:
    raise ValueError(
        f"Expected both credit-status groups; found: {observed_groups}"
    )

# Weighted distribution of obstacle severity within credit-status groups.
weighted = (
    valid.group_by(["credit_status", "obstacle_code"])
    .agg(
        pl.len().alias("sample_count"),
        pl.col("wmedian").sum().alias("weight_sum"),
    )
    .with_columns(
        pl.col("sample_count")
        .sum()
        .over("credit_status")
        .alias("group_sample_total"),

        pl.col("weight_sum")
        .sum()
        .over("credit_status")
        .alias("group_weight_total"),
    )
    .with_columns(
        (
            pl.col("sample_count")
            / pl.col("group_sample_total")
            * 100
        ).round(2).alias("sample_percent"),

        (
            pl.col("weight_sum")
            / pl.col("group_weight_total")
            * 100
        ).round(2).alias("weighted_percent"),
    )
    .sort(["credit_status", "obstacle_code"])
)

# Compare the prevalence of major or very severe obstacles.
severe = (
    valid.group_by("credit_status")
    .agg(
        pl.len().alias("sample_count"),
        pl.col("wmedian").sum().alias("group_weight_total"),
        pl.when(pl.col("major_obstacle") == 1)
        .then(pl.col("wmedian"))
        .otherwise(0.0)
        .sum()
        .alias("major_severe_weight"),
        pl.when(pl.col("major_obstacle") == 1)
        .then(1)
        .otherwise(0)
        .sum()
        .alias("major_severe_count"),
    )
    .with_columns(
        (
            pl.col("major_severe_weight")
            / pl.col("group_weight_total")
            * 100
        ).round(2).alias("weighted_major_severe_pct"),

        (
            pl.col("major_severe_count")
            / pl.col("sample_count")
            * 100
        ).round(2).alias("unweighted_major_severe_pct"),
    )
    .sort("credit_status")
)

print("=" * 70)
print("WEIGHTED CREDIT STATUS VS FINANCE OBSTACLE")
print("=" * 70)
print(f"Original observations: {df.height:,}")
print(f"Valid analysis observations: {valid.height:,}")
print(f"Excluded observations: {df.height - valid.height:,}")
print("Primary weight: wmedian")
print("\nOBSTACLE DISTRIBUTION WITHIN CREDIT-STATUS GROUPS")
print(weighted)
print("\nMAJOR/VERY SEVERE OBSTACLE COMPARISON")
print(severe)

weighted_path = (
    output_dir / "weighted_credit_vs_obstacle_crosstab.csv"
)
severe_path = (
    output_dir / "weighted_credit_vs_obstacle_comparison.csv"
)

weighted.write_csv(weighted_path)
severe.write_csv(severe_path)

print(f"\nSaved: {weighted_path}")
print(f"Saved: {severe_path}")
print("Analysis completed.")
