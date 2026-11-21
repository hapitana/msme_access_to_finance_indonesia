from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

candidate_path = processed_dir / "indonesia_2023_candidate_analytic.parquet"
source_path = processed_dir / "indonesia_2023.parquet"

output_dir = processed_dir / "script54_common_sample_audit"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("WBES INDONESIA 2023: COMMON SAMPLE AND MISSINGNESS AUDIT")
print("=" * 72)


def require_columns(data, columns, dataset_name):
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {dataset_name}: {missing}"
        )


def save_csv(data, filename):
    data.write_csv(output_dir / filename)


def save_parquet(data, filename):
    data.write_parquet(output_dir / filename)


def outcome_summary(data, sample_name):
    valid_data = data.filter(
        pl.col("major_obstacle").is_not_null()
        & pl.col("wmedian").is_not_null()
        & (pl.col("wmedian") > 0)
    )

    if valid_data.height == 0:
        return {
            "sample": sample_name,
            "n": 0,
            "major_obstacle_count": 0,
            "unweighted_rate": None,
            "weighted_rate_wmedian": None,
        }

    result = valid_data.select(
        pl.len().alias("n"),
        pl.col("major_obstacle").sum().alias(
            "major_obstacle_count"
        ),
        pl.col("major_obstacle").mean().alias(
            "unweighted_rate"
        ),
        (
            (
                pl.col("major_obstacle").cast(pl.Float64)
                * pl.col("wmedian")
            ).sum()
            / pl.col("wmedian").sum()
        ).alias("weighted_rate_wmedian"),
    ).to_dicts()[0]

    return {
        "sample": sample_name,
        **result,
    }


def outcome_distribution(data, sample_name):
    return (
        data.group_by("major_obstacle")
        .agg(
            pl.len().alias("observations"),
            pl.col("wmedian").sum().alias(
                "sum_wmedian"
            ),
        )
        .with_columns(
            pl.lit(sample_name).alias("sample")
        )
        .select(
            "sample",
            "major_obstacle",
            "observations",
            "sum_wmedian",
        )
        .sort("major_obstacle")
    )


def numeric_comparison(data, group_column, variables):
    results = []

    for variable in variables:
        groups = (
            data.group_by(group_column)
            .agg(
                pl.col(variable).mean().alias("mean"),
                pl.col(variable).std().alias("std"),
                pl.col(variable).count().alias("valid_n"),
            )
            .sort(group_column)
        )

        for row in groups.to_dicts():
            results.append(
                {
                    "group": row[group_column],
                    "variable": variable,
                    "valid_n": row["valid_n"],
                    "mean": row["mean"],
                    "std": row["std"],
                }
            )

    return pl.DataFrame(results)


print("\n1. Loading candidate analytic dataset")

if not candidate_path.exists():
    raise FileNotFoundError(
        f"Candidate dataset not found: {candidate_path}"
    )

data = pl.read_parquet(candidate_path)

print(f"Candidate observations: {data.height:,}")
print(f"Candidate columns: {len(data.columns):,}")

require_columns(
    data,
    [
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
    ],
    "candidate analytic dataset",
)


print("\n2. Preparing bank account variable")

if "bank_account" not in data.columns:
    if "k6" not in data.columns:
        source = pl.read_parquet(source_path)

        require_columns(
            source,
            ["idstd", "k6"],
            "source dataset",
        )

        data = data.join(
            source.select(["idstd", "k6"]),
            on="idstd",
            how="left",
            validate="1:1",
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

print(
    data.group_by("bank_account")
    .agg(pl.len().alias("observations"))
    .sort("bank_account")
)

require_columns(
    data,
    ["bank_account"],
    "prepared analytic dataset",
)


print("\n3. Defining model completeness")

model_1_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
]

model_2_variables = model_1_variables + [
    "bank_account",
]

model_3_variables = model_2_variables + [
    "export_share",
]

data = data.with_columns(
    pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in model_1_variables
        ]
    ).alias("model1_complete"),

    pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in model_2_variables
        ]
    ).alias("model2_complete"),

    pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in model_3_variables
        ]
    ).alias("model3_complete"),
)

model1_full = data.filter(
    pl.col("model1_complete")
)

model2_common = data.filter(
    pl.col("model2_complete")
)

model1_common = data.filter(
    pl.col("model1_complete")
    & pl.col("model2_complete")
)

model3_complete = data.filter(
    pl.col("model3_complete")
)

print(f"Model 1 full sample: {model1_full.height:,}")
print(f"Model 1 common sample: {model1_common.height:,}")
print(f"Model 2 common sample: {model2_common.height:,}")
print(f"Model 3 complete sample: {model3_complete.height:,}")

if model1_common.height != model2_common.height:
    raise ValueError(
        "Model 1 and Model 2 common samples do not match."
    )

if model1_common.height != model2_common.height:
    raise ValueError(
        "Common sample validation failed."
    )


print("\n4. Saving consistent model datasets")

save_parquet(
    model1_full,
    "model1_full_sample.parquet",
)

save_parquet(
    model1_common,
    "model1_common_sample.parquet",
)

save_parquet(
    model2_common,
    "model2_common_sample.parquet",
)

save_parquet(
    model3_complete,
    "model3_complete_sample.parquet",
)

print("Model datasets saved.")


print("\n5. Comparing model sample sizes")

sample_comparison = pl.DataFrame(
    [
        {
            "model": "Model 1 full",
            "observations": model1_full.height,
            "excluded_from_source": data.height - model1_full.height,
        },
        {
            "model": "Model 1 common",
            "observations": model1_common.height,
            "excluded_from_source": data.height - model1_common.height,
        },
        {
            "model": "Model 2 common",
            "observations": model2_common.height,
            "excluded_from_source": data.height - model2_common.height,
        },
        {
            "model": "Model 3 complete",
            "observations": model3_complete.height,
            "excluded_from_source": data.height - model3_complete.height,
        },
    ]
)

save_csv(
    sample_comparison,
    "common_sample_size_comparison.csv",
)

print(sample_comparison)


print("\n6. Comparing outcome distributions")

outcome_summaries = pl.DataFrame(
    [
        outcome_summary(model1_full, "Model 1 full"),
        outcome_summary(model1_common, "Model 1 common"),
        outcome_summary(model2_common, "Model 2 common"),
        outcome_summary(model3_complete, "Model 3 complete"),
    ]
)

save_csv(
    outcome_summaries,
    "common_sample_outcome_summary.csv",
)

print(outcome_summaries)


outcome_distributions = pl.concat(
    [
        outcome_distribution(
            model1_full,
            "Model 1 full",
        ),
        outcome_distribution(
            model1_common,
            "Model 1 common",
        ),
        outcome_distribution(
            model2_common,
            "Model 2 common",
        ),
        outcome_distribution(
            model3_complete,
            "Model 3 complete",
        ),
    ],
    how="vertical",
)

save_csv(
    outcome_distributions,
    "common_sample_outcome_distribution.csv",
)

print(outcome_distributions)


print("\n7. Identifying missingness patterns")

pattern_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
    "export_share",
]

pattern_expressions = [
    pl.when(pl.col(column).is_not_null())
    .then(pl.lit("1"))
    .otherwise(pl.lit("0"))
    for column in pattern_variables
]

data = data.with_columns(
    pl.concat_str(
        pattern_expressions,
        separator="",
    ).alias("missingness_pattern")
)

missingness_patterns = (
    data.group_by("missingness_pattern")
    .agg(
        pl.len().alias("observations"),
        pl.col("major_obstacle").is_not_null().sum().alias(
            "valid_outcome"
        ),
        pl.col("export_share").is_not_null().sum().alias(
            "valid_export_share"
        ),
        pl.col("bank_account").is_not_null().sum().alias(
            "valid_bank_account"
        ),
    )
    .sort("observations", descending=True)
)

save_csv(
    missingness_patterns,
    "missingness_pattern_distribution.csv",
)

print("Pattern variable order:")
print(", ".join(pattern_variables))
print(missingness_patterns)


print("\n8. Comparing complete and excluded observations")

data = data.with_columns(
    pl.when(pl.col("model1_complete"))
    .then(pl.lit("Model 1 complete"))
    .otherwise(pl.lit("Model 1 excluded"))
    .alias("model1_status"),

    pl.when(pl.col("model2_complete"))
    .then(pl.lit("Model 2 complete"))
    .otherwise(pl.lit("Model 2 excluded"))
    .alias("model2_status"),
)

comparison_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
    "export_share",
]

model1_characteristics = numeric_comparison(
    data,
    "model1_status",
    comparison_variables,
)

model2_characteristics = numeric_comparison(
    data,
    "model2_status",
    comparison_variables,
)

save_csv(
    model1_characteristics,
    "model1_complete_excluded_characteristics.csv",
)

save_csv(
    model2_characteristics,
    "model2_complete_excluded_characteristics.csv",
)

print("\nModel 1 characteristics:")
print(model1_characteristics)

print("\nModel 2 characteristics:")
print(model2_characteristics)


print("\n9. Checking common sample consistency")

common_ids_1 = set(
    model1_common.get_column("idstd").to_list()
)

common_ids_2 = set(
    model2_common.get_column("idstd").to_list()
)

print(
    f"Model 1 common IDs: {len(common_ids_1):,}"
)
print(
    f"Model 2 common IDs: {len(common_ids_2):,}"
)
print(
    f"Identical common samples: {common_ids_1 == common_ids_2}"
)

if common_ids_1 != common_ids_2:
    raise ValueError(
        "Model 1 and Model 2 common sample IDs differ."
    )


print("\n10. Saving audit dataset")

save_parquet(
    data,
    "script54_full_audit_dataset.parquet",
)

print("\n" + "=" * 72)
print("SCRIPT 54 COMPLETED")
print("=" * 72)

print(f"Output directory: {output_dir}")
print(f"Total source observations: {data.height:,}")
print(f"Model 1 full sample: {model1_full.height:,}")
print(f"Model 1 common sample: {model1_common.height:,}")
print(f"Model 2 common sample: {model2_common.height:,}")
print(f"Model 3 complete sample: {model3_complete.height:,}")