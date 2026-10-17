from pathlib import Path

import polars as pl

# 1. Define project paths
project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

output_dir = project_root / "data" / "processed"

if not data_path.exists():
    raise FileNotFoundError(f"Input dataset not found: {data_path}")

output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path)

# 2. Create the binary target variable:
# 1 = Major or very severe obstacle
# 0 = No obstacle, minor obstacle, or moderate obstacle
# Special codes and missing values are excluded.
model_df = (
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
    .select(
        "major_obstacle",
        "a6b",
        "credit_status",
        "a4b_v4",
        "a3a",
        "wmedian",
    )
    .rename({
        "a6b": "firm_size_code",
        "a4b_v4": "industry_code",
        "a3a": "region_code",
        "wmedian": "survey_weight",
    })
)

print("MODEL DATASET PREPARATION")
print("=" * 60)
print(f"Original observations: {df.height:,}")

# 3. Check missing values in each model variable
print("\nMISSING VALUES BY VARIABLE")

for column in model_df.columns:
    print(f"{column}: {model_df[column].null_count():,}")

# 4. Retain complete observations with valid model inputs
complete_df = model_df.filter(
    pl.col("major_obstacle").is_not_null()
    & pl.col("firm_size_code").is_in([1, 2, 3])
    & pl.col("credit_status").is_not_null()
    & pl.col("industry_code").is_not_null()
    & pl.col("region_code").is_not_null()
    & pl.col("survey_weight").is_not_null()
    & (pl.col("survey_weight") > 0)
)

print("\nCOMPLETE MODEL SAMPLE")
print(f"Complete observations: {complete_df.height:,}")
print(f"Excluded observations: {df.height - complete_df.height:,}")

print("\nTARGET DISTRIBUTION")
print(
    complete_df.group_by("major_obstacle")
    .len(name="count")
    .sort("major_obstacle")
)

print("\nFIRM SIZE DISTRIBUTION")
print(
    complete_df.group_by("firm_size_code")
    .len(name="count")
    .sort("firm_size_code")
)

# 5. Save the prepared model dataset
output_path = output_dir / "model_dataset.parquet"
complete_df.write_parquet(output_path)

print(f"\nOutput file: {output_path}")
print("Model dataset preparation completed successfully.")
