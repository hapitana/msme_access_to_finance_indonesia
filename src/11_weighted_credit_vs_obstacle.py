
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

# Kelompokkan status kredit dan hambatan keuangan
analysis = df.with_columns(
    pl.when(pl.col("k82").is_in([1, 2, 3]))
    .then(pl.lit("Has credit"))
    .when(pl.col("k82") == 4)
    .then(pl.lit("No credit"))
    .otherwise(None)
    .alias("credit_status"),

    pl.when(pl.col("k30").is_between(0, 4))
    .then(pl.col("k30"))
    .otherwise(None)
    .alias("obstacle_code"),
)

# Gunakan hanya observasi dengan kedua jawaban valid
# dan bobot survei yang valid
valid = analysis.filter(
    pl.col("credit_status").is_not_null()
    & pl.col("obstacle_code").is_not_null()
    & pl.col("wmedian").is_not_null()
    & (pl.col("wmedian") > 0)
)

# Hitung jumlah observasi dan total bobot per kategori
weighted = (
    valid.group_by(["credit_status", "obstacle_code"])
    .agg(
        pl.len().alias("sample_count"),
        pl.col("wmedian").sum().alias("weight_sum"),
    )
    .with_columns(
        pl.col("weight_sum")
        .sum()
        .over("credit_status")
        .alias("group_weight_total")
    )
    .with_columns(
        (
            pl.col("weight_sum")
            / pl.col("group_weight_total")
            * 100
        )
        .round(2)
        .alias("weighted_percent")
    )
    .sort(["credit_status", "obstacle_code"])
)

print("WEIGHTED CREDIT STATUS VS FINANCE OBSTACLE")
print("=" * 65)
print(f"Original observations: {df.height:,}")
print(f"Valid observations: {valid.height:,}")
print("\nWeighted results:")
print(weighted)

# Ringkas: hambatan besar atau sangat berat (kode 3 atau 4)
severe = (
    valid.with_columns(
        pl.col("k30")
        .is_in([3, 4])
        .alias("major_severe")
    )
    .group_by("credit_status")
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
    .sort("credit_status")
)

print("\nMAJOR/SEVERE OBSTACLE COMPARISON")
print(severe)

weighted.write_csv(
    output_dir / "weighted_credit_vs_obstacle_crosstab.csv"
)

severe.write_csv(
    output_dir / "weighted_credit_vs_obstacle_comparison.csv"
)

print("\nFiles saved to data/processed/")
print("Analysis completed.")
