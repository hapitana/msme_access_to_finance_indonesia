from pathlib import Path
import polars as pl

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

input_file = processed_dir / "indonesia_2023.parquet"
output_file = processed_dir / "indonesia_2023_candidate_analytic.parquet"

if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pl.read_parquet(input_file)

required_columns = [
    "idstd", "k30", "b4", "b4a", "b5", "a14y",
    "l1", "d3a", "d3b", "d3c", "a4a", "a2",
    "a6a", "strata", "wmedian", "wweak"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df["idstd"].n_unique() != df.height:
    raise ValueError("Duplicate observation identifiers detected.")

print("=" * 70)
print("WBES INDONESIA 2023: CANDIDATE ANALYTIC DATASET")
print("=" * 70)
print(f"Source observations: {df.height}")

df = df.with_columns([
    pl.col("b5").cast(pl.Float64),
    pl.col("a14y").cast(pl.Float64),
    pl.col("l1").cast(pl.Float64),
    pl.col("b4a").cast(pl.Float64),
    pl.col("d3a").cast(pl.Float64),
    pl.col("d3b").cast(pl.Float64),
    pl.col("d3c").cast(pl.Float64),
])

df = df.with_columns([
    pl.when(pl.col("k30").is_in([0, 1, 2, 3, 4]))
    .then(
        pl.when(pl.col("k30").is_in([3, 4]))
        .then(1)
        .otherwise(0)
    )
    .otherwise(None)
    .cast(pl.Int8)
    .alias("major_obstacle"),

    pl.when(
        pl.col("b5").is_between(1800, 2023)
        & pl.col("a14y").is_between(2022, 2023)
        & (pl.col("b5") <= pl.col("a14y"))
    )
    .then(pl.col("a14y") - pl.col("b5"))
    .otherwise(None)
    .alias("firm_age"),

    pl.when(pl.col("l1") > 0)
    .then(pl.col("l1").log())
    .otherwise(None)
    .alias("log_employees"),

    pl.when(pl.col("b4") == 1)
    .then(1)
    .when(pl.col("b4") == 2)
    .then(0)
    .otherwise(None)
    .cast(pl.Int8)
    .alias("female_owner"),

    pl.when(
        (pl.col("b4") == 1)
        & pl.col("b4a").is_between(0, 100)
    )
    .then(pl.col("b4a"))
    .otherwise(None)
    .alias("female_ownership_pct"),
])

df = df.with_columns(
    pl.when(
        pl.col("d3a").is_between(0, 100)
        & pl.col("d3b").is_between(0, 100)
        & pl.col("d3c").is_between(0, 100)
        & (
            pl.col("d3a")
            + pl.col("d3b")
            + pl.col("d3c")
            == 100
        )
    )
    .then(pl.col("d3b") + pl.col("d3c"))
    .otherwise(None)
    .alias("export_share")
)

df = df.with_columns([
    pl.when(pl.col("export_share").is_not_null())
    .then(
        pl.when(pl.col("export_share") > 0)
        .then(1)
        .otherwise(0)
    )
    .otherwise(None)
    .cast(pl.Int8)
    .alias("exporter"),

    pl.when(pl.col("export_share").is_not_null())
    .then(pl.col("d3a"))
    .otherwise(None)
    .alias("domestic_sales_share"),
])

candidate_columns = [
    "idstd",
    "k30",
    "major_obstacle",
    "b5",
    "a14y",
    "firm_age",
    "l1",
    "log_employees",
    "b4",
    "b4a",
    "female_owner",
    "female_ownership_pct",
    "d3a",
    "d3b",
    "d3c",
    "export_share",
    "exporter",
    "domestic_sales_share",
    "a4a",
    "a2",
    "a6a",
    "strata",
    "wmedian",
    "wweak",
]

optional_columns = [
    "stratificationpanelcode",
    "a20y",
    "d1a2_v4",
]

for column in optional_columns:
    if column in df.columns:
        candidate_columns.append(column)

analytic_df = df.select(candidate_columns)

main_model_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "export_share",
    "a4a",
    "a2",
    "strata",
    "idstd",
    "wmedian",
]

analytic_df = analytic_df.with_columns(
    pl.all_horizontal([
        pl.col(column).is_not_null()
        for column in main_model_variables
    ])
    .alias("main_model_complete")
)

analytic_df.write_parquet(output_file)

print("\n1. ANALYTIC DATASET")
print(f"Rows: {analytic_df.height}")
print(f"Columns: {analytic_df.width}")
print(f"Output: {output_file}")

print("\n2. DERIVED VARIABLE SUMMARY")

derived_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "female_ownership_pct",
    "export_share",
    "exporter",
    "domestic_sales_share",
]

summary_records = []

for variable in derived_variables:
    values = analytic_df[variable]

    summary_records.append({
        "variable": variable,
        "valid_observations": values.count(),
        "missing_observations": values.null_count(),
        "unique_values": values.n_unique(),
        "minimum": values.min(),
        "maximum": values.max(),
        "mean": values.mean(),
        "median": values.median(),
    })

derived_summary = pl.DataFrame(summary_records)

print(derived_summary)

derived_summary.write_csv(
    processed_dir / "candidate_analytic_variable_summary.csv"
)

print("\n3. MAIN MODEL SAMPLE")

model_sample = analytic_df.filter(
    pl.col("main_model_complete")
)

print(f"Source observations: {analytic_df.height}")
print(f"Complete observations: {model_sample.height}")
print(
    "Excluded observations: "
    f"{analytic_df.height - model_sample.height}"
)

print("\n4. MAIN MODEL SAMPLE BY OUTCOME")

outcome_summary = (
    model_sample
    .group_by("major_obstacle")
    .len()
    .sort("major_obstacle")
)

print(outcome_summary)

outcome_summary.write_csv(
    processed_dir / "candidate_main_model_outcome_summary.csv"
)

print("\n5. WEIGHT CHECK")

weight_summary = (
    analytic_df
    .select([
        pl.col("wmedian").min().alias("wmedian_minimum"),
        pl.col("wmedian").max().alias("wmedian_maximum"),
        pl.col("wmedian").null_count().alias("wmedian_missing"),
        (pl.col("wmedian") <= 0).sum().alias("wmedian_nonpositive"),
        pl.col("wweak").min().alias("wweak_minimum"),
        pl.col("wweak").max().alias("wweak_maximum"),
        pl.col("wweak").null_count().alias("wweak_missing"),
        (pl.col("wweak") <= 0).sum().alias("wweak_nonpositive"),
    ])
)

print(weight_summary)

print("\n6. AGE VALIDATION")

age_validation = (
    analytic_df
    .select([
        pl.col("firm_age").min().alias("minimum_age"),
        pl.col("firm_age").median().alias("median_age"),
        pl.col("firm_age").mean().alias("mean_age"),
        pl.col("firm_age").max().alias("maximum_age"),
        pl.col("firm_age").null_count().alias("missing_age"),
        (pl.col("firm_age") < 0).sum().alias("negative_age"),
    ])
)

print(age_validation)

print("\n7. ORIGINAL DATA PRESERVATION CHECK")

print(f"Original rows: {df.height}")
print(f"Analytic rows: {analytic_df.height}")
print(f"Original unique IDs: {df['idstd'].n_unique()}")
print(f"Analytic unique IDs: {analytic_df['idstd'].n_unique()}")

if analytic_df.height != df.height:
    raise RuntimeError("Observation count changed unexpectedly.")

if analytic_df["idstd"].n_unique() != df["idstd"].n_unique():
    raise RuntimeError("Observation identifiers changed unexpectedly.")

print("\nDataset construction completed.")