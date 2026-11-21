from pathlib import Path

import polars as pl


print("=" * 76)
print("69. MODEL SAMPLE MEMBERSHIP AUDIT")
print("=" * 76)


project_dir = Path(__file__).resolve().parents[1]

data_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023_candidate_analytic.parquet"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "model_specific_design_audit"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

if not data_path.exists():
    raise FileNotFoundError(
        f"Analytic dataset not found: {data_path}"
    )


print()
print("1. LOAD ANALYTIC DATA")

data = pl.read_parquet(data_path)

print(f"Input file: {data_path}")
print(f"Rows: {data.height:,}")
print(f"Columns: {len(data.columns):,}")


print()
print("2. RECONSTRUCT BANK ACCOUNT VARIABLE")

base_file = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

if not base_file.exists():
    raise FileNotFoundError(
        f"Base dataset not found: {base_file}"
    )

base_data = pl.read_parquet(base_file)

required_base_columns = [
    "idstd",
    "k6",
]

missing_base_columns = [
    column
    for column in required_base_columns
    if column not in base_data.columns
]

if missing_base_columns:
    raise KeyError(
        f"Missing base columns: {missing_base_columns}"
    )

k6_lookup = base_data.select(
    ["idstd", "k6"]
)

if k6_lookup["idstd"].n_unique() != k6_lookup.height:
    raise ValueError(
        "Duplicate idstd values in base dataset."
    )

if data["idstd"].n_unique() != data.height:
    raise ValueError(
        "Duplicate idstd values in candidate dataset."
    )

data = data.join(
    k6_lookup,
    on="idstd",
    how="left",
    validate="1:1",
)

if data["k6"].null_count() > 0:
    raise ValueError(
        "Some candidate observations could not be matched to k6."
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

print("Bank account reconstructed from k6.")
print(
    data.group_by("bank_account")
    .len()
    .sort("bank_account")
)

print()
print("2. VALIDATE REQUIRED COLUMNS")

model1_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
]

model2_variables = model1_variables + [
    "bank_account",
]

required_columns = list(
    dict.fromkeys(
        model2_variables
        + ["idstd"]
    )
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise KeyError(
        f"Missing required columns: {missing_columns}"
    )

if data["idstd"].null_count() > 0:
    raise ValueError(
        "Missing idstd values detected."
    )

duplicate_psu = (
    data
    .group_by("idstd")
    .len()
    .filter(pl.col("len") > 1)
)

if duplicate_psu.height > 0:
    raise ValueError(
        "idstd is not unique. "
        "Row-level membership comparison cannot proceed safely."
    )

print("Required columns passed validation.")
print("idstd is complete and unique.")


print()
print("3. RECONSTRUCT ORIGINAL COMPLETE-CASE FLAGS")

model1_complete_expression = pl.all_horizontal(
    [
        pl.col(column).is_not_null()
        for column in model1_variables
    ]
)

model2_complete_expression = pl.all_horizontal(
    [
        pl.col(column).is_not_null()
        for column in model2_variables
    ]
)

data = data.with_columns(
    [
        model1_complete_expression.alias(
            "model1_full_reconstructed"
        ),
        model2_complete_expression.alias(
            "common_sample_reconstructed"
        ),
    ]
)

print(
    "Model 1 full sample:",
    data.filter(
        pl.col("model1_full_reconstructed")
    ).height,
)

print(
    "Common sample:",
    data.filter(
        pl.col("common_sample_reconstructed")
    ).height,
)


print()
print("4. CHECK NESTING OF COMPLETE-CASE SAMPLES")

invalid_nested_rows = data.filter(
    pl.col("common_sample_reconstructed")
    & ~pl.col("model1_full_reconstructed")
)

model1_only_rows = data.filter(
    pl.col("model1_full_reconstructed")
    & ~pl.col("common_sample_reconstructed")
)

common_rows = data.filter(
    pl.col("common_sample_reconstructed")
)

nesting_valid = (
    invalid_nested_rows.height == 0
)

print(
    "Common observations outside Model 1 full:",
    invalid_nested_rows.height,
)

print(
    "Model 1 full observations excluded from common sample:",
    model1_only_rows.height,
)

print(
    "Common sample is a subset of Model 1 full:",
    nesting_valid,
)


print()
print("5. COMPARE MODEL 1 COMMON AND MODEL 2 COMMON")

model1_common_ids = (
    common_rows
    .select("idstd")
    .unique()
    .sort("idstd")
)

model2_common_ids = (
    common_rows
    .select("idstd")
    .unique()
    .sort("idstd")
)

model1_common_not_model2 = (
    model1_common_ids
    .join(
        model2_common_ids,
        on="idstd",
        how="anti",
    )
)

model2_common_not_model1 = (
    model2_common_ids
    .join(
        model1_common_ids,
        on="idstd",
        how="anti",
    )
)

common_membership_identical = (
    model1_common_not_model2.height == 0
    and model2_common_not_model1.height == 0
)

print(
    "Model 1 common observations:",
    model1_common_ids.height,
)

print(
    "Model 2 common observations:",
    model2_common_ids.height,
)

print(
    "Model 1 common but not Model 2 common:",
    model1_common_not_model2.height,
)

print(
    "Model 2 common but not Model 1 common:",
    model2_common_not_model1.height,
)

print(
    "Identical row-level membership:",
    common_membership_identical,
)


print()
print("6. AUDIT BANK ACCOUNT EXCLUSION")

bank_account_missing_in_model1 = (
    data
    .filter(
        pl.col("model1_full_reconstructed")
        & pl.col("bank_account").is_null()
    )
    .height
)

bank_account_missing_outside_model1 = (
    data
    .filter(
        ~pl.col("model1_full_reconstructed")
        & pl.col("bank_account").is_null()
    )
    .height
)

print(
    "Model 1 full observations with missing bank_account:",
    bank_account_missing_in_model1,
)

print(
    "Observations outside Model 1 full with missing bank_account:",
    bank_account_missing_outside_model1,
)

print(
    "Model 1 full minus common sample:",
    model1_only_rows.height,
)

bank_account_exclusion_consistent = (
    model1_only_rows.height
    == bank_account_missing_in_model1
)


print()
print("7. CREATE ROW-LEVEL MEMBERSHIP FILE")

membership = data.select(
    [
        "idstd",
        "model1_full_reconstructed",
        "common_sample_reconstructed",
    ]
).with_columns(
    [
        pl.col(
            "common_sample_reconstructed"
        ).alias("model1_common"),
        pl.col(
            "common_sample_reconstructed"
        ).alias("model2_common"),
    ]
).sort("idstd")

membership_path = (
    output_dir
    / "model_sample_row_level_membership.csv"
)

membership.write_csv(membership_path)

print(f"Saved: {membership_path}")


print()
print("8. SAVE AUDIT SUMMARY")

summary = pl.DataFrame(
    {
        "check": [
            "input_rows",
            "model1_full_n",
            "model1_common_n",
            "model2_common_n",
            "common_outside_model1_full",
            "model1_full_excluded_from_common",
            "model1_common_not_model2_common",
            "model2_common_not_model1_common",
            "bank_account_missing_within_model1_full",
            "bank_account_missing_outside_model1_full",
        ],
        "value": [
            data.height,
            data.filter(
                pl.col("model1_full_reconstructed")
            ).height,
            model1_common_ids.height,
            model2_common_ids.height,
            invalid_nested_rows.height,
            model1_only_rows.height,
            model1_common_not_model2.height,
            model2_common_not_model1.height,
            bank_account_missing_in_model1,
            bank_account_missing_outside_model1,
        ],
    }
)

summary_path = (
    output_dir
    / "model_sample_membership_audit.csv"
)

summary.write_csv(summary_path)

print(summary)
print(f"Saved: {summary_path}")


print()
print("9. FINAL VALIDATION")

checks = {
    "common_sample_nested_in_model1_full": nesting_valid,
    "model1_common_and_model2_common_identical": (
        common_membership_identical
    ),
    "bank_account_exclusion_consistent": (
        bank_account_exclusion_consistent
    ),
    "model1_full_n_matches_expected": (
        data.filter(
            pl.col("model1_full_reconstructed")
        ).height == 2632
    ),
    "common_sample_n_matches_expected": (
        common_rows.height == 2528
    ),
}

for check_name, passed in checks.items():
    print(
        f"{check_name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )

all_checks_passed = all(checks.values())

print()
print("=" * 76)

if all_checks_passed:
    print("69 MODEL SAMPLE MEMBERSHIP AUDIT PASSED")
else:
    print("69 MODEL SAMPLE MEMBERSHIP AUDIT REQUIRES INVESTIGATION")

print("=" * 76)

if not all_checks_passed:
    raise ValueError(
        "One or more sample membership checks failed. "
        "Inspect the saved audit outputs."
    )