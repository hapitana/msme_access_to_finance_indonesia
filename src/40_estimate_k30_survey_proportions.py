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

if "k30" not in df.columns:
    raise ValueError("Variable k30 was not found.")

required_columns = [
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

print("=" * 65)
print("WBES INDONESIA 2023: SURVEY-WEIGHTED K30 PROPORTIONS")
print("=" * 65)

print("\n1. K30 variable label")

print(
    metadata.column_names_to_labels.get(
        "k30",
        "No variable label",
    )
)

print("\n2. K30 value labels")

value_labels = metadata.variable_value_labels.get("k30", {})

if value_labels:
    for value, label in sorted(value_labels.items()):
        print(f"{value}: {label}")
else:
    print("No value labels available.")

print("\n3. K30 observed frequency")

k30_frequency = (
    df.group_by("k30")
    .agg(
        pl.len().alias("observations")
    )
    .sort("k30")
)

print(k30_frequency)

print("\nValid observations:", df["k30"].count())
print("Missing observations:", df["k30"].null_count())

print("\n4. Survey proportion estimation")

weights = ["wmedian", "wweak"]
singleton_methods = ["scale", "center"]

results = []

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
            y="k30",
            drop_nulls=True,
        )

        result_df = estimate.to_polars()

        result_df = result_df.with_columns(
            pl.lit(weight).alias("weight"),
            pl.lit(method).alias("singleton_method"),
        )

        print(result_df)

        results.append(result_df)

print("\n5. Combine all estimation results")

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
    / "k30_survey_weighted_proportions.csv"
)

final_results.write_csv(output_path)

print("\nResults saved to:")
print(output_path)

print("\nInterpretation note:")
print(
    "These are preliminary design-based estimates. "
    "The singleton sensitivity and sampling variance "
    "specification still require methodological review."
)

print("\nEstimation completed.")