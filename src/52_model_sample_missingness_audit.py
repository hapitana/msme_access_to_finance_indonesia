from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "processed"

input_file = data_dir / "indonesia_2023_candidate_analytic.parquet"

output_missingness = data_dir / "model_missingness_audit.csv"
output_samples = data_dir / "model_sample_comparison.csv"
output_outcomes = data_dir / "model_sample_outcome_comparison.csv"
output_k6 = data_dir / "k6_banking_access_distribution.csv"
output_selection = data_dir / "model_selection_comparison.parquet"


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def weighted_outcome_summary(data, sample_name):
    valid = data.filter(
        pl.col("major_obstacle").is_not_null()
        & pl.col("wmedian").is_not_null()
        & (pl.col("wmedian") > 0)
    )

    if valid.height == 0:
        return {
            "sample": sample_name,
            "valid_outcomes": 0,
            "major_obstacle_cases": 0,
            "unweighted_rate": None,
            "weighted_rate_wmedian": None,
        }

    result = valid.select(
        pl.len().alias("valid_outcomes"),
        pl.col("major_obstacle").sum().alias("major_obstacle_cases"),
        pl.col("major_obstacle").mean().alias("unweighted_rate"),
        (
            (pl.col("major_obstacle") * pl.col("wmedian")).sum()
            / pl.col("wmedian").sum()
        ).alias("weighted_rate_wmedian"),
    ).to_dicts()[0]

    return {
        "sample": sample_name,
        **result,
    }


def model_sample_summary(data, model_name, predictors):
    complete_flag = f"{model_name.lower()}_complete"

    complete_expression = pl.all_horizontal(
        [pl.col(variable).is_not_null() for variable in predictors]
    )

    return data.with_columns(
        complete_expression.alias(complete_flag)
    )


print_section("WBES INDONESIA 2023: MODEL SAMPLE AND MISSINGNESS AUDIT")

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}"
    )

data = pl.read_parquet(input_file)

print(f"Candidate observations: {data.height}")
print(f"Candidate columns: {data.width}")

source_file = data_dir / "indonesia_2023.parquet"

if not source_file.exists():
    raise FileNotFoundError(
        f"Original source dataset not found: {source_file}"
    )

source_data = pl.read_parquet(
    source_file,
    columns=["idstd", "k6"],
)

if source_data.select(
    pl.col("idstd").n_unique()
).item() != source_data.height:
    raise ValueError(
        "Duplicate firm identifiers found in source dataset."
    )

data = data.join(
    source_data,
    on="idstd",
    how="left",
    validate="1:1",
)

print(f"Source observations after join: {data.height}")
print(f"Source columns after join: {data.width}")

required_columns = [
    "idstd",
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "export_share",
    "a4a",
    "a2",
    "strata",
    "wmedian",
    "wweak",
    "k6",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

numeric_columns = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "export_share",
    "wmedian",
    "wweak",
    "k6",
]

data = data.with_columns(
    [
        pl.col(column).cast(pl.Float64, strict=False).alias(column)
        for column in numeric_columns
    ]
)

data = data.with_columns(
    pl.when(pl.col("k6") == 1)
    .then(1)
    .when(pl.col("k6") == 2)
    .then(0)
    .otherwise(None)
    .cast(pl.Int8)
    .alias("bank_account")
)

print_section("1. BANK ACCOUNT VARIABLE AUDIT")

k6_distribution = (
    data.group_by("k6")
    .agg(
        pl.len().alias("observations")
    )
    .sort("k6")
)

print(k6_distribution)

k6_distribution.write_csv(output_k6)

print(f"Output: {output_k6}")

print_section("2. MODEL SPECIFICATIONS")

model_predictors = {
    "Model_1": [
        "major_obstacle",
        "firm_age",
        "log_employees",
        "female_owner",
        "a4a",
        "a2",
        "strata",
        "idstd",
        "wmedian",
    ],
    "Model_2": [
        "major_obstacle",
        "firm_age",
        "log_employees",
        "female_owner",
        "bank_account",
        "a4a",
        "a2",
        "strata",
        "idstd",
        "wmedian",
    ],
    "Model_3": [
        "major_obstacle",
        "firm_age",
        "log_employees",
        "female_owner",
        "bank_account",
        "export_share",
        "a4a",
        "a2",
        "strata",
        "idstd",
        "wmedian",
    ],
}

for model_name, predictors in model_predictors.items():
    data = model_sample_summary(
        data,
        model_name,
        predictors,
    )

sample_records = []

for model_name, predictors in model_predictors.items():
    flag = f"{model_name.lower()}_complete"

    complete_data = data.filter(pl.col(flag))
    excluded_data = data.filter(~pl.col(flag))

    sample_records.append(
        {
            "model": model_name,
            "predictor_count_including_outcome": len(predictors),
            "complete_observations": complete_data.height,
            "excluded_observations": excluded_data.height,
            "complete_sample_percent": (
                complete_data.height / data.height * 100
            ),
            "excluded_sample_percent": (
                excluded_data.height / data.height * 100
            ),
        }
    )

sample_comparison = pl.DataFrame(sample_records)

print(sample_comparison)

sample_comparison.write_csv(output_samples)

print(f"Output: {output_samples}")

print_section("3. VARIABLE MISSINGNESS AUDIT")

audit_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
    "export_share",
    "a4a",
    "a2",
    "strata",
    "idstd",
    "wmedian",
]

missingness_records = []

for variable in audit_variables:
    missing_count = data.select(
        pl.col(variable).is_null().sum()
    ).item()

    missingness_records.append(
        {
            "variable": variable,
            "valid_observations": data.height - missing_count,
            "missing_observations": missing_count,
            "missing_percent": (
                missing_count / data.height * 100
            ),
        }
    )

missingness_audit = (
    pl.DataFrame(missingness_records)
    .sort("missing_observations", descending=True)
)

print(missingness_audit)

missingness_audit.write_csv(output_missingness)

print(f"Output: {output_missingness}")

print_section("4. MISSINGNESS CONTRIBUTION TO MODEL 3 EXCLUSIONS")

model_3_excluded = data.filter(~pl.col("model_3_complete"))

exclusion_records = []

model_3_predictors = model_predictors["Model_3"]

for variable in model_3_predictors:
    missing_count = model_3_excluded.select(
        pl.col(variable).is_null().sum()
    ).item()

    exclusion_records.append(
        {
            "variable": variable,
            "excluded_cases_missing_variable": missing_count,
            "percent_of_model_3_exclusions": (
                missing_count / max(model_3_excluded.height, 1) * 100
            ),
        }
    )

exclusion_audit = (
    pl.DataFrame(exclusion_records)
    .sort(
        "excluded_cases_missing_variable",
        descending=True,
    )
)

print(exclusion_audit)

exclusion_file = data_dir / "model_3_exclusion_contribution.csv"
exclusion_audit.write_csv(exclusion_file)

print(f"Output: {exclusion_file}")

print_section("5. OUTCOME COMPARISON: COMPLETE VS EXCLUDED")

data = data.with_columns(
    pl.when(pl.col("model_3_complete"))
    .then(pl.lit("Complete_Model_3"))
    .otherwise(pl.lit("Excluded_Model_3"))
    .alias("model_3_sample_status")
)

outcome_records = []

for status in [
    "Complete_Model_3",
    "Excluded_Model_3",
    "Full_Valid_Outcome_Sample",
]:
    if status == "Full_Valid_Outcome_Sample":
        subset = data
    else:
        subset = data.filter(
            pl.col("model_3_sample_status") == status
        )

    outcome_records.append(
        weighted_outcome_summary(
            subset,
            status,
        )
    )

outcome_comparison = pl.DataFrame(outcome_records)

print(outcome_comparison)

outcome_comparison.write_csv(output_outcomes)

print(f"Output: {output_outcomes}")

print_section("6. OUTCOME COUNTS BY MODEL")

model_outcome_records = []

for model_name in model_predictors:
    flag = f"{model_name.lower()}_complete"

    subset = data.filter(
        pl.col(flag)
        & pl.col("major_obstacle").is_not_null()
    )

    counts = (
        subset.group_by("major_obstacle")
        .agg(pl.len().alias("observations"))
        .sort("major_obstacle")
    )

    for row in counts.to_dicts():
        model_outcome_records.append(
            {
                "model": model_name,
                "major_obstacle": row["major_obstacle"],
                "observations": row["observations"],
            }
        )

model_outcome_comparison = pl.DataFrame(
    model_outcome_records
)

print(model_outcome_comparison)

outcome_counts_file = (
    data_dir / "model_outcome_counts.csv"
)

model_outcome_comparison.write_csv(
    outcome_counts_file
)

print(f"Output: {outcome_counts_file}")

print_section("7. PRESERVATION AND WEIGHT CHECK")

print(
    "Original observations:",
    data.height,
)

print(
    "Unique firm identifiers:",
    data.select(pl.col("idstd").n_unique()).item(),
)

weight_check = data.select(
    pl.col("wmedian").min().alias("wmedian_min"),
    pl.col("wmedian").max().alias("wmedian_max"),
    pl.col("wmedian").null_count().alias("wmedian_missing"),
    (pl.col("wmedian") <= 0).sum().alias("wmedian_nonpositive"),
    pl.col("wweak").min().alias("wweak_min"),
    pl.col("wweak").max().alias("wweak_max"),
    pl.col("wweak").null_count().alias("wweak_missing"),
    (pl.col("wweak") <= 0).sum().alias("wweak_nonpositive"),
)

print(weight_check)

print_section("8. SAVE MODEL SELECTION DATASET")

data.write_parquet(output_selection)

print(f"Output: {output_selection}")

print_section("AUDIT COMPLETED")

print("All model sample and missingness audits completed.")
print("No observations were deleted from the source dataset.")