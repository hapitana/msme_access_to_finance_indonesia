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

# 1. Label firm size categories
analysis = df.with_columns(
    pl.when(pl.col("a6b") == 0)
    .then(pl.lit("Micro"))
    .when(pl.col("a6b") == 1)
    .then(pl.lit("Small"))
    .when(pl.col("a6b") == 2)
    .then(pl.lit("Medium"))
    .when(pl.col("a6b") == 3)
    .then(pl.lit("Large"))
    .otherwise(None)
    .alias("firm_size"),

    pl.when(pl.col("k30").is_between(0, 4))
    .then(pl.col("k30"))
    .otherwise(None)
    .alias("obstacle_code"),
)

# 2. Select observations with valid size, obstacle, and weight values
valid = analysis.filter(
    pl.col("firm_size").is_not_null()
    & pl.col("obstacle_code").is_not_null()
    & pl.col("wmedian").is_not_null()
    & (pl.col("wmedian") > 0)
)

print("FIRM SIZE VS FINANCE OBSTACLE")
print("=" * 65)
print(f"Original observations: {df.height:,}")
print(f"Valid observations: {valid.height:,}")

# 3. Obstacle level distribution by firm size
weighted = (
    valid.group_by(["firm_size", "obstacle_code"])
    .agg(
        pl.len().alias("sample_count"),
        pl.col("wmedian").sum().alias("weight_sum"),
    )
    .with_columns(
        pl.col("weight_sum")
        .sum()
        .over("firm_size")
        .alias("size_group_weight_total")
    )
    .with_columns(
        (
            pl.col("weight_sum")
            / pl.col("size_group_weight_total")
            * 100
        )
        .round(2)
        .alias("weighted_percent")
    )
    .sort(["firm_size", "obstacle_code"])
)

print("\nWEIGHTED OBSTACLE DISTRIBUTION")
print(weighted)

# 4. Summary of major or very severe obstacles
severe = (
    valid.with_columns(
        pl.col("k30")
        .is_in([3, 4])
        .alias("major_severe")
    )
    .group_by("firm_size")
    .agg(
        pl.len().alias("sample_count"),
        pl.col("wmedian").sum().alias("group_weight_total"),
        pl.when(pl.col("major_severe"))
        .then(pl.col("wmedian"))
        .otherwise(0.0)
        .sum()
        .alias("major_severe_weight"),
        pl.when(pl.col("major_severe"))
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
        )
        .round(2)
        .alias("weighted_major_severe_percent")
    )
    .sort("firm_size")
)

print("\nMAJOR/SEVERE OBSTACLE BY FIRM SIZE")
print(severe)

# 5. Save the results
weighted.write_csv(
    output_dir / "weighted_size_vs_obstacle_crosstab.csv"
)

severe.write_csv(
    output_dir / "weighted_size_vs_obstacle_comparison.csv"
)

print("\nResults saved to data/processed/")
print("Analysis completed.")
