from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

variables_path = processed_dir / "selected_model_variables.csv"
frequencies_path = processed_dir / "selected_model_variable_frequencies.csv"

variables = pl.read_csv(variables_path)
frequencies = pl.read_csv(frequencies_path)

print("=" * 78)
print("WBES INDONESIA 2023: CANDIDATE VARIABLE DETAIL AUDIT")
print("=" * 78)

print("\n1. Complete candidate variable inventory")

for row in variables.sort("variable").iter_rows(named=True):
    print("\n" + "-" * 78)
    print(f"Variable: {row['variable']}")
    print(f"Label: {row['label']}")
    print(f"Group: {row['groups']}")
    print(f"Valid observations: {row['valid_observations']}")
    print(f"Missing observations: {row['missing_observations']}")
    print(f"Unique values: {row['unique_values']}")
    print(f"Value labels: {row['value_labels']}")

print("\n2. Categorical distributions")

categorical_variables = (
    variables
    .filter(pl.col("unique_values") <= 15)
    .get_column("variable")
    .to_list()
)

categorical_frequencies = (
    frequencies
    .filter(pl.col("variable").is_in(categorical_variables))
    .sort(["variable", "value"])
)

for variable in categorical_variables:
    subset = categorical_frequencies.filter(
        pl.col("variable") == variable
    )

    print("\n" + "-" * 78)
    print(f"Variable: {variable}")

    for row in subset.iter_rows(named=True):
        print(
            f"Value: {row['value']} | "
            f"Label: {row['value_label']} | "
            f"Count: {row['observations']}"
        )

print("\n3. Candidate variables with substantial missing observations")

missing_candidates = (
    variables
    .with_columns(
        (
            pl.col("missing_observations") / 2955 * 100
        ).round(2).alias("missing_percentage")
    )
    .filter(pl.col("missing_observations") > 0)
    .sort("missing_percentage", descending=True)
)

print(
    missing_candidates.select(
        "variable",
        "label",
        "groups",
        "valid_observations",
        "missing_observations",
        "missing_percentage",
        "unique_values"
    )
)

print("\n4. Output files")

detail_path = processed_dir / "candidate_variable_detail.csv"
missing_path = processed_dir / "candidate_variable_missingness.csv"

variables.sort("variable").write_csv(detail_path)
missing_candidates.write_csv(missing_path)

print(detail_path)
print(missing_path)

print("\nCandidate variable detail audit completed.")