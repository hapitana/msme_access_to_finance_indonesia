
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

df = pl.read_parquet(data_path)

# Reconstruct the same complete-case sample.
df = (
    df.with_columns(
        pl.when(pl.col("k30").is_in([3, 4]))
        .then(1)
        .when(pl.col("k30").is_in([0, 1, 2]))
        .then(0)
        .otherwise(None)
        .alias("major_obstacle"),

        pl.when(pl.col("k82").is_in([1, 2, 3]))
        .then(pl.lit("Has credit"))
        .when(pl.col("k82") == 4)
        .then(pl.lit("No credit"))
        .otherwise(None)
        .alias("credit_status"),
    )
    .filter(
        pl.col("major_obstacle").is_not_null()
        & pl.col("k82").is_in([1, 2, 3, 4])
        & pl.col("a6b").is_in([1, 2, 3])
        & pl.col("a4b_v4").is_not_null()
        & pl.col("a3a").is_not_null()
    )
)

weights = ["wstrict", "wmedian", "wweak"]
summary_rows = []

print("=" * 60)
print("WEIGHT SENSITIVITY CHECK")
print("=" * 60)
print(f"Complete-case observations: {df.height}")

for weight in weights:
    valid = df.filter(
        pl.col(weight).is_not_null()
        & (pl.col(weight) > 0)
    )

    weighted_rate = (
        (
            pl.col("major_obstacle")
            * pl.col(weight)
        ).sum()
        / pl.col(weight).sum()
    )

    result = valid.select(
        pl.len().alias("observations"),
        pl.col("major_obstacle").mean().alias("unweighted_rate"),
        weighted_rate.alias("weighted_rate"),
        pl.col(weight).sum().alias("weight_sum"),
    )

    row = result.row(0, named=True)

    summary_rows.append(
        {
            "weight": weight,
            "observations": row["observations"],
            "unweighted_pct": row["unweighted_rate"] * 100,
            "weighted_pct": row["weighted_rate"] * 100,
            "weight_sum": row["weight_sum"],
        }
    )

summary = pl.DataFrame(summary_rows).with_columns(
    pl.col("unweighted_pct").round(2),
    pl.col("weighted_pct").round(2),
    pl.col("weight_sum").round(2),
)

print("\nOVERALL ESTIMATES")
print(summary)

summary.write_csv(
    output_dir / "weight_sensitivity_summary.csv"
)

print("\nSaved to:")
print(output_dir / "weight_sensitivity_summary.csv")
print("Sensitivity check completed.")
