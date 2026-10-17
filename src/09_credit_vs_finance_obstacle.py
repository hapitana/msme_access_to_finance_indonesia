
from pathlib import Path
import polars as pl

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "processed" / "indonesia_2023.parquet"
output_dir = project_root / "data" / "processed"

df = pl.read_parquet(data_path)

# Classify known credit status
analysis = df.with_columns(
    pl.when(pl.col("k82").is_in([1, 2, 3]))
    .then(pl.lit("Has credit"))
    .when(pl.col("k82") == 4)
    .then(pl.lit("No credit"))
    .otherwise(None)
    .alias("credit_status"),

    pl.when(pl.col("k30") == 0)
    .then(pl.lit("0 No obstacle"))
    .when(pl.col("k30") == 1)
    .then(pl.lit("1 Minor"))
    .when(pl.col("k30") == 2)
    .then(pl.lit("2 Moderate"))
    .when(pl.col("k30") == 3)
    .then(pl.lit("3 Major"))
    .when(pl.col("k30") == 4)
    .then(pl.lit("4 Very severe"))
    .otherwise(None)
    .alias("finance_obstacle")
)

# Keep observations with valid answers to both variables
valid = analysis.filter(
    pl.col("credit_status").is_not_null()
    & pl.col("finance_obstacle").is_not_null()
)

print("CREDIT STATUS VS ACCESS-TO-FINANCE OBSTACLE")
print("=" * 65)
print(f"Original observations: {df.height:,}")
print(f"Observations with both valid answers: {valid.height:,}")

# Count observations in each combination
cross_tab = (
    valid.group_by(["credit_status", "finance_obstacle"])
    .len(name="count")
    .with_columns(
        pl.col("count")
        .sum()
        .over("credit_status")
        .alias("credit_group_total")
    )
    .with_columns(
        (
            pl.col("count")
            / pl.col("credit_group_total")
            * 100
        ).round(2).alias("row_percent")
    )
    .sort(["credit_status", "finance_obstacle"])
)

print("\nCross-tabulation:")
print(cross_tab)

# Create a focused comparison: major or very severe obstacle
comparison = (
    valid.with_columns(
        pl.when(pl.col("k30").is_in([3, 4]))
        .then(pl.lit("Major/severe obstacle"))
        .otherwise(pl.lit("No major/severe obstacle"))
        .alias("obstacle_group")
    )
    .group_by(["credit_status", "obstacle_group"])
    .len(name="count")
    .with_columns(
        pl.col("count")
        .sum()
        .over("credit_status")
        .alias("credit_group_total")
    )
    .with_columns(
        (
            pl.col("count")
            / pl.col("credit_group_total")
            * 100
        ).round(2).alias("row_percent")
    )
    .sort(["credit_status", "obstacle_group"])
)

print("\nMajor/severe obstacle comparison:")
print(comparison)

# Export results for the portfolio
cross_tab.write_csv(
    output_dir / "credit_vs_obstacle_crosstab.csv"
)
comparison.write_csv(
    output_dir / "credit_vs_obstacle_comparison.csv"
)

print("\nResults saved to data/processed/")
print("Analysis completed.")
