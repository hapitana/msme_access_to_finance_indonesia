from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

output_dir = (
    project_root
    / "data"
    / "processed"
    / "outcome_robustness"
)

output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path)

required = [
    "k30",
    "k162",
    "wmedian",
    "idstd",
    "strata",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Required columns not found: {missing}"
    )

analysis = df.with_columns(
    pl.when(pl.col("k30").is_in([3, 4]))
    .then(1)
    .when(pl.col("k30").is_in([0, 1, 2]))
    .then(0)
    .otherwise(None)
    .alias("obstacle_major"),

    pl.when(pl.col("k162").is_in([1, 2, 3]))
    .then(1)
    .when(pl.col("k162") == 4)
    .then(0)
    .otherwise(None)
    .alias("credit_application"),
)

# Retain valid survey design information.
design_data = analysis.filter(
    pl.col("wmedian").is_not_null()
    & pl.col("wmedian").is_finite()
    & (pl.col("wmedian") > 0)
    & pl.col("idstd").is_not_null()
    & pl.col("strata").is_not_null()
)

# Export the design data, including records with missing
# substantive responses. R will define the complete-case domain.
export_columns = [
    "k30",
    "k162",
    "obstacle_major",
    "credit_application",
    "wmedian",
    "idstd",
    "strata",
]

design_data.select(export_columns).write_parquet(
    output_dir / "k30_k162_raoscott_input.parquet"
)

print("RAO-SCOTT DATA PREPARATION")
print("=" * 60)
print(f"Original observations: {df.height:,}")
print(f"Valid design records: {design_data.height:,}")

print("\nk30 response distribution:")
print(
    df.group_by("k30")
    .len()
    .sort("k30")
)

print("\nk162 response distribution:")
print(
    df.group_by("k162")
    .len()
    .sort("k162")
)

print("\nComplete substantive response count:")
print(
    design_data.filter(
        pl.col("obstacle_major").is_not_null()
        & pl.col("credit_application").is_not_null()
    ).height
)

print("\nExport completed:")
print(
    output_dir
    / "k30_k162_raoscott_input.parquet"
)