from pathlib import Path

import polars as pl
import pyreadstat
import svy

project_dir = Path(__file__).resolve().parent.parent

data_path = (
    project_dir
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise FileNotFoundError(data_path)

data, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

df = pl.DataFrame(data)

required_columns = [
    "k30",
    "strata",
    "idstd",
    "wmedian",
    "wweak",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required variables are missing: {missing_columns}"
    )

df = df.with_columns(
    pl.when(pl.col("k30").is_in([0, 1, 2]))
    .then(0)
    .when(pl.col("k30").is_in([3, 4]))
    .then(1)
    .otherwise(None)
    .cast(pl.Int8)
    .alias("major_obstacle")
)

valid_n = df["major_obstacle"].count()
missing_n = df["major_obstacle"].null_count()

unweighted_frequency = (
    df.group_by("major_obstacle")
    .agg(
        pl.len().alias("observations")
    )
    .sort("major_obstacle")
)

print("=" * 65)
print("WBES INDONESIA 2023: MAJOR FINANCING OBSTACLE")
print("=" * 65)

print("\n1. Outcome definition")
print("0 = No, minor, or moderate obstacle")
print("1 = Major or very severe obstacle")
print("Null = Don't know or does not apply")

print("\n2. Unweighted outcome frequency")
print(unweighted_frequency)

print("\nValid observations:", valid_n)
print("Excluded observations:", missing_n)

if valid_n == 0:
    raise ValueError("No valid observations for major_obstacle.")

if valid_n + missing_n != df.height:
    raise ValueError("Observation count validation failed.")

weights = ["wmedian", "wweak"]
singleton_methods = ["scale", "center"]

results = []

print("\n3. Survey-weighted estimation")

for weight in weights:
    for method in singleton_methods:
        print("\n" + "-" * 65)
        print(f"Weight: {weight}")
        print(f"Singleton method: {method}")
        print("-" * 65)

        design = svy.Design(
            stratum="strata",
            psu="idstd",
            wgt=weight,
            singleton=method,
        )

        sample = svy.Sample(
            data=df,
            design=design,
        )

        estimate = sample.estimation.prop(
            y="major_obstacle",
            drop_nulls=True,
        )

        result_df = estimate.to_polars()

        result_df = result_df.with_columns(
            pl.lit(weight).alias("weight"),
            pl.lit(method).alias("singleton_method"),
        )

        print(result_df)

        results.append(result_df)

print("\n4. Combine estimation results")

final_results = pl.concat(
    results,
    how="diagonal_relaxed",
)

column_order = [
    "weight",
    "singleton_method",
] + [
    column
    for column in final_results.columns
    if column not in ["weight", "singleton_method"]
]

final_results = final_results.select(column_order)

print(final_results)

output_path = (
    output_dir
    / "major_financing_obstacle_estimates.csv"
)

final_results.write_csv(output_path)

print("\nResults saved to:")
print(output_path)

print("\nInterpretation note:")
print(
    "The outcome measures the proportion of valid responses "
    "reporting major or very severe access-to-finance obstacles. "
    "Don't know and does not apply responses are excluded."
)

print(
    "Estimates remain preliminary pending validation "
    "of the survey variance specification."
)

print("\nEstimation completed.")