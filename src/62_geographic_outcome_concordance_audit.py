from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parents[1]

candidate_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023_candidate_analytic.parquet"
)

source_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "outcome_robustness"
    / "24c20b_geographic_outcome_concordance_audit"
)

output_dir.mkdir(parents=True, exist_ok=True)

output_summary = output_dir / "24c20b_outcome_concordance_summary.csv"
output_crosstab = output_dir / "24c20b_major_obstacle_crosstab.csv"
output_mismatches = output_dir / "24c20b_major_obstacle_mismatches.csv"

print("=" * 72)
print("24C-20B GEOGRAPHIC OUTCOME CONCORDANCE AUDIT")
print("=" * 72)

if not candidate_path.exists():
    raise FileNotFoundError(
        f"Candidate dataset not found: {candidate_path}"
    )

if not source_path.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {source_path}"
    )

candidate = pl.read_parquet(candidate_path)

source = pl.read_parquet(source_path).select(
    "idstd",
    "k30",
)

required_candidate_columns = [
    "idstd",
    "major_obstacle",
]

missing_columns = [
    column
    for column in required_candidate_columns
    if column not in candidate.columns
]

if missing_columns:
    raise ValueError(
        f"Missing candidate columns: {missing_columns}"
    )

if candidate["idstd"].null_count() > 0:
    raise ValueError("Null idstd values found in candidate dataset.")

if source["idstd"].null_count() > 0:
    raise ValueError("Null idstd values found in source dataset.")

if candidate["idstd"].n_unique() != candidate.height:
    raise ValueError("Duplicate idstd values in candidate dataset.")

if source["idstd"].n_unique() != source.height:
    raise ValueError("Duplicate idstd values in source dataset.")

data = candidate.select(
    "idstd",
    "major_obstacle",
).join(
    source,
    on="idstd",
    how="left",
    validate="1:1",
)

if data["k30"].null_count() > 0:
    raise ValueError(
        "Some candidate observations could not be matched to k30."
    )

data = data.with_columns(
    pl.col("major_obstacle")
    .cast(pl.Float64, strict=False)
    .alias("major_obstacle_original"),

    pl.col("k30")
    .cast(pl.Float64, strict=False)
    .alias("k30_numeric"),
)

data = data.with_columns(
    pl.when(
        pl.col("k30_numeric").is_between(
            0,
            4,
            closed="both",
        )
    )
    .then(
        (pl.col("k30_numeric") >= 3).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("major_obstacle_rebuilt"),
)

data = data.with_columns(
    (
        pl.col("major_obstacle_original").is_in([0.0, 1.0])
    ).alias("original_binary_valid"),

    pl.col("major_obstacle_rebuilt")
    .is_not_null()
    .alias("rebuilt_valid"),

    (
        pl.col("k30_numeric").is_not_null()
        & ~pl.col("k30_numeric").is_in(
            [-9.0, -7.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        )
    ).alias("unexpected_k30_code"),
)

data = data.with_columns(
    (
        pl.col("original_binary_valid")
        & pl.col("rebuilt_valid")
    ).alias("both_valid"),

    (
        pl.col("original_binary_valid")
        & pl.col("rebuilt_valid")
        & (
            pl.col("major_obstacle_original")
            != pl.col("major_obstacle_rebuilt")
        )
    ).alias("outcome_mismatch"),

    (
        pl.col("original_binary_valid")
        & ~pl.col("rebuilt_valid")
    ).alias("original_only_valid"),

    (
        ~pl.col("original_binary_valid")
        & pl.col("rebuilt_valid")
    ).alias("rebuilt_only_valid"),
)

summary = data.select(
    pl.len().alias("total_observations"),
    pl.col("idstd").n_unique().alias("unique_establishments"),
    pl.col("original_binary_valid")
    .sum()
    .alias("original_binary_valid_n"),
    pl.col("rebuilt_valid")
    .sum()
    .alias("rebuilt_valid_n"),
    pl.col("both_valid")
    .sum()
    .alias("comparable_n"),
    pl.col("outcome_mismatch")
    .sum()
    .alias("mismatch_n"),
    pl.col("original_only_valid")
    .sum()
    .alias("original_only_valid_n"),
    pl.col("rebuilt_only_valid")
    .sum()
    .alias("rebuilt_only_valid_n"),
    pl.col("unexpected_k30_code")
    .sum()
    .alias("unexpected_k30_code_n"),
)

summary = summary.with_columns(
    pl.when(pl.col("comparable_n") > 0)
    .then(
        (
            1
            - pl.col("mismatch_n") / pl.col("comparable_n")
        ) * 100
    )
    .otherwise(None)
    .alias("agreement_pct"),

    (
        pl.col("mismatch_n") == 0
    ).alias("no_definition_mismatch"),
)

crosstab = (
    data.filter(pl.col("both_valid"))
    .group_by(
        "major_obstacle_original",
        "major_obstacle_rebuilt",
    )
    .agg(
        pl.len().alias("observations")
    )
    .sort(
        [
            "major_obstacle_original",
            "major_obstacle_rebuilt",
        ]
    )
)

mismatches = (
    data.filter(pl.col("outcome_mismatch"))
    .select(
        "idstd",
        "k30_numeric",
        "major_obstacle_original",
        "major_obstacle_rebuilt",
    )
)

k30_distribution = (
    data.group_by("k30_numeric")
    .agg(
        pl.len().alias("observations")
    )
    .sort("k30_numeric")
)

summary.write_csv(output_summary)
crosstab.write_csv(output_crosstab)
mismatches.write_csv(output_mismatches)

k30_distribution.write_csv(
    output_dir / "24c20b_k30_distribution.csv"
)

print()
print("DATA VALIDATION")
print(f"Candidate rows: {candidate.height:,}")
print(f"Source rows: {source.height:,}")
print(f"Joined rows: {data.height:,}")

print()
print("K30 DISTRIBUTION")
print(k30_distribution)

print()
print("OUTCOME CONCORDANCE SUMMARY")
print(summary)

print()
print("MAJOR OBSTACLE CROSS-TABULATION")
print(crosstab)

print()
print("MISMATCH OBSERVATIONS")
print(f"Mismatch rows: {mismatches.height:,}")

print()
print("24C-20B AUDIT COMPLETED")
print(f"Output directory: {output_dir}")