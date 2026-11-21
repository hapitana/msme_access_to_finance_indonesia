from pathlib import Path

import polars as pl


project_root = Path(__file__).resolve().parents[1]

source_path = (
    project_root
    / "data"
    / "processed"
    / "outcome_robustness"
    / "24c19_geographic_label_audit"
    / "24c19_weighted_geographic_prevalence.csv"
)

output_dir = (
    project_root
    / "data"
    / "processed"
    / "outcome_robustness"
    / "24c20_effective_sample_size_audit"
)

output_dir.mkdir(parents=True, exist_ok=True)

output_detail = output_dir / "24c20_effective_sample_size_detail.csv"
output_summary = output_dir / "24c20_effective_sample_size_summary.csv"

data = pl.read_csv(source_path)

data = data.with_columns(
    pl.col("a3a").cast(pl.Int64),
    pl.col("valid_n").cast(pl.Int64),
    pl.col("event_n").cast(pl.Int64),
    pl.col("kish_effective_n").cast(pl.Float64),
    pl.col("weighted_prevalence_pct").cast(pl.Float64),
    pl.col("sparse_cell_flag").cast(pl.Boolean),
)

data = data.with_columns(
    (pl.col("kish_effective_n") < 10)
    .alias("kish_below_10"),

    (
        (pl.col("kish_effective_n") >= 10)
        & (pl.col("kish_effective_n") < 20)
    ).alias("kish_between_10_and_20"),

    (pl.col("kish_effective_n") < 20)
    .alias("kish_below_20"),
)

data = data.with_columns(
    pl.when(pl.col("kish_below_10"))
    .then(pl.lit("Very low effective sample size"))
    .when(pl.col("kish_below_20"))
    .then(pl.lit("Low effective sample size"))
    .otherwise(pl.lit("No low Kish flag"))
    .alias("kish_assessment"),

    (
        pl.col("sparse_cell_flag")
        | pl.col("kish_below_20")
    ).alias("combined_warning"),
)

data = data.sort(
    ["outcome", "weight", "kish_effective_n"]
)

data.write_csv(output_detail)

summary = (
    data.group_by(["outcome", "weight"])
    .agg(
        pl.len().alias("region_count"),
        pl.col("kish_effective_n").min().alias("minimum_kish_n"),
        pl.col("kish_effective_n").median().alias("median_kish_n"),
        pl.col("kish_effective_n").max().alias("maximum_kish_n"),
        pl.col("kish_below_10").sum().alias("regions_below_10"),
        pl.col("kish_below_20").sum().alias("regions_below_20"),
        pl.col("sparse_cell_flag").sum().alias("sparse_cell_regions"),
        pl.col("combined_warning").sum().alias("regions_with_warning"),
    )
    .sort(["outcome", "weight"])
)

summary.write_csv(output_summary)

print()
print("24C-20 KISH EFFECTIVE SAMPLE SIZE AUDIT")
print()
print("INPUT VALIDATION")
print(f"Rows: {data.height}")
print(f"Unique regions: {data.get_column('a3a').n_unique()}")
print(f"Outcomes: {data.get_column('outcome').n_unique()}")
print(f"Weights: {data.get_column('weight').n_unique()}")
print()
print("EFFECTIVE SAMPLE SIZE SUMMARY")
print(summary)
print()
print("LOW EFFECTIVE SAMPLE SIZE OBSERVATIONS")
print(
    data.filter(pl.col("kish_below_20")).select(
        "a3a",
        "region",
        "outcome",
        "weight",
        "valid_n",
        "event_n",
        "kish_effective_n",
        "kish_assessment",
        "sparse_cell_flag",
    )
)
print()
print("24C-20 AUDIT COMPLETED")
print(f"Output directory: {output_dir}")