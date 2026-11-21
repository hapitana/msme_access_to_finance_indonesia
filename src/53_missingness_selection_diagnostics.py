from pathlib import Path
import math

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "processed"

input_file = data_dir / "model_selection_comparison.parquet"

output_characteristics = (
    data_dir / "sample_characteristics_comparison.csv"
)
output_smd = data_dir / "sample_standardized_mean_differences.csv"
output_export_profile = (
    data_dir / "export_missingness_profile.csv"
)
output_export_smd = (
    data_dir / "export_missingness_smd.csv"
)
output_outcomes = (
    data_dir / "sample_outcome_diagnostics.csv"
)


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def get_scalar(data, expression):
    return data.select(expression).item()


def numeric_summary(data, variable):
    valid = data.filter(pl.col(variable).is_not_null())

    if valid.height == 0:
        return {
            "valid_n": 0,
            "missing_n": data.height,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }

    result = valid.select(
        pl.len().alias("valid_n"),
        pl.col(variable).mean().alias("mean"),
        pl.col(variable).std(ddof=1).alias("standard_deviation"),
        pl.col(variable).median().alias("median"),
        pl.col(variable).min().alias("minimum"),
        pl.col(variable).max().alias("maximum"),
    ).to_dicts()[0]

    result["missing_n"] = data.height - result["valid_n"]

    return result


def calculate_numeric_smd(group_a, group_b, variable):
    values_a = numeric_summary(group_a, variable)
    values_b = numeric_summary(group_b, variable)

    mean_a = values_a["mean"]
    mean_b = values_b["mean"]
    sd_a = values_a["standard_deviation"]
    sd_b = values_b["standard_deviation"]

    if None in [mean_a, mean_b, sd_a, sd_b]:
        return None

    n_a = values_a["valid_n"]
    n_b = values_b["valid_n"]

    if n_a < 2 or n_b < 2:
        return None

    pooled_variance = (
        ((n_a - 1) * sd_a**2)
        + ((n_b - 1) * sd_b**2)
    ) / (n_a + n_b - 2)

    if pooled_variance <= 0:
        return None

    pooled_sd = math.sqrt(pooled_variance)

    return (mean_a - mean_b) / pooled_sd


def category_distribution(data, variable):
    valid = data.filter(pl.col(variable).is_not_null())

    if valid.height == 0:
        return {}

    counts = (
        valid.group_by(variable)
        .agg(pl.len().alias("count"))
        .sort(variable)
    )

    return {
        str(row[variable]): row["count"] / valid.height
        for row in counts.to_dicts()
    }


def calculate_categorical_smd(group_a, group_b, variable):
    distribution_a = category_distribution(group_a, variable)
    distribution_b = category_distribution(group_b, variable)

    categories = sorted(
        set(distribution_a) | set(distribution_b)
    )

    records = []

    for category in categories:
        p_a = distribution_a.get(category, 0.0)
        p_b = distribution_b.get(category, 0.0)

        pooled_p = (p_a + p_b) / 2

        denominator = math.sqrt(
            pooled_p * (1 - pooled_p)
        )

        if denominator == 0:
            smd = None
        else:
            smd = (p_a - p_b) / denominator

        records.append(
            {
                "variable": variable,
                "category": category,
                "proportion_complete_or_observed": p_a,
                "proportion_excluded_or_missing": p_b,
                "smd": smd,
                "absolute_smd": (
                    abs(smd) if smd is not None else None
                ),
            }
        )

    return records


def build_group_characteristics(
    data,
    comparison_name,
    group_name,
    numeric_variables,
    categorical_variables,
):
    records = []

    for variable in numeric_variables:
        summary = numeric_summary(data, variable)

        records.append(
            {
                "comparison": comparison_name,
                "group": group_name,
                "variable": variable,
                "variable_type": "numeric",
                **summary,
            }
        )

    for variable in categorical_variables:
        valid = data.filter(pl.col(variable).is_not_null())

        records.append(
            {
                "comparison": comparison_name,
                "group": group_name,
                "variable": variable,
                "variable_type": "categorical",
                "valid_n": valid.height,
                "missing_n": data.height - valid.height,
                "mean": None,
                "standard_deviation": None,
                "median": None,
                "minimum": None,
                "maximum": None,
            }
        )

    return records


def build_smd_records(
    group_a,
    group_b,
    comparison_name,
    numeric_variables,
    categorical_variables,
):
    records = []

    for variable in numeric_variables:
        smd = calculate_numeric_smd(
            group_a,
            group_b,
            variable,
        )

        records.append(
            {
                "comparison": comparison_name,
                "variable": variable,
                "category": None,
                "smd": smd,
                "absolute_smd": (
                    abs(smd) if smd is not None else None
                ),
            }
        )

    for variable in categorical_variables:
        category_records = calculate_categorical_smd(
            group_a,
            group_b,
            variable,
        )

        for record in category_records:
            records.append(
                {
                    "comparison": comparison_name,
                    **record,
                }
            )

    return records


print_section(
    "WBES INDONESIA 2023: MISSINGNESS AND SAMPLE SELECTION DIAGNOSTICS"
)

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}"
    )

data = pl.read_parquet(input_file)

print(f"Observations: {data.height}")
print(f"Columns: {data.width}")

required_columns = [
    "idstd",
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
    "export_share",
    "a4a",
    "a2",
    "model_1_complete",
    "model_2_complete",
    "model_3_complete",
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

numeric_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
    "export_share",
]

categorical_variables = [
    "a4a",
    "a2",
]

for variable in categorical_variables:
    data = data.with_columns(
        pl.col(variable).cast(pl.String)
    )

print_section("1. MODEL SAMPLE CHARACTERISTICS")

characteristic_records = []
smd_records = []

for model_name in ["Model_1", "Model_2", "Model_3"]:
    flag = f"{model_name.lower()}_complete"

    complete_group = data.filter(pl.col(flag))
    excluded_group = data.filter(~pl.col(flag))

    print()
    print(f"{model_name}")
    print(f"Complete observations: {complete_group.height}")
    print(f"Excluded observations: {excluded_group.height}")

    characteristic_records.extend(
        build_group_characteristics(
            complete_group,
            model_name,
            "Complete",
            numeric_variables,
            categorical_variables,
        )
    )

    characteristic_records.extend(
        build_group_characteristics(
            excluded_group,
            model_name,
            "Excluded",
            numeric_variables,
            categorical_variables,
        )
    )

    smd_records.extend(
        build_smd_records(
            complete_group,
            excluded_group,
            model_name,
            numeric_variables,
            categorical_variables,
        )
    )

characteristics = pl.DataFrame(characteristic_records)

characteristics.write_csv(
    output_characteristics
)

print(f"Output: {output_characteristics}")

print_section("2. STANDARDIZED MEAN DIFFERENCES")

smd_data = pl.DataFrame(smd_records)

smd_data = smd_data.sort(
    ["comparison", "variable", "category"]
)

print(smd_data)

smd_data.write_csv(output_smd)

print(f"Output: {output_smd}")

print_section("3. EXPORT SHARE MISSINGNESS PROFILE")

observed_export = data.filter(
    pl.col("export_share").is_not_null()
)

missing_export = data.filter(
    pl.col("export_share").is_null()
)

print(f"Export share observed: {observed_export.height}")
print(f"Export share missing: {missing_export.height}")

export_characteristics = []

export_characteristics.extend(
    build_group_characteristics(
        observed_export,
        "Export_Share_Availability",
        "Observed",
        numeric_variables,
        categorical_variables,
    )
)

export_characteristics.extend(
    build_group_characteristics(
        missing_export,
        "Export_Share_Availability",
        "Missing",
        numeric_variables,
        categorical_variables,
    )
)

export_profile = pl.DataFrame(export_characteristics)

export_profile.write_csv(output_export_profile)

print(f"Output: {output_export_profile}")

print_section("4. EXPORT SHARE MISSINGNESS SMD")

export_smd_records = build_smd_records(
    observed_export,
    missing_export,
    "Export_Share_Availability",
    numeric_variables,
    categorical_variables,
)

export_smd = pl.DataFrame(export_smd_records)

export_smd = export_smd.sort(
    ["variable", "category"]
)

print(export_smd)

export_smd.write_csv(output_export_smd)

print(f"Output: {output_export_smd}")

print_section("5. OUTCOME DIAGNOSTICS")

outcome_records = []

comparison_groups = [
    (
        "Model_1",
        data.filter(pl.col("model_1_complete")),
        data.filter(~pl.col("model_1_complete")),
    ),
    (
        "Model_2",
        data.filter(pl.col("model_2_complete")),
        data.filter(~pl.col("model_2_complete")),
    ),
    (
        "Model_3",
        data.filter(pl.col("model_3_complete")),
        data.filter(~pl.col("model_3_complete")),
    ),
    (
        "Export_Share_Availability",
        observed_export,
        missing_export,
    ),
]

for comparison_name, group_a, group_b in comparison_groups:
    for group_name, subset in [
        ("Complete_or_Observed", group_a),
        ("Excluded_or_Missing", group_b),
    ]:
        valid = subset.filter(
            pl.col("major_obstacle").is_not_null()
        )

        if valid.height == 0:
            outcome_records.append(
                {
                    "comparison": comparison_name,
                    "group": group_name,
                    "valid_outcomes": 0,
                    "major_obstacle_cases": None,
                    "unweighted_rate": None,
                    "weighted_rate_wmedian": None,
                }
            )
            continue

        summary = valid.select(
            pl.len().alias("valid_outcomes"),
            pl.col("major_obstacle").sum().alias(
                "major_obstacle_cases"
            ),
            pl.col("major_obstacle").mean().alias(
                "unweighted_rate"
            ),
            (
                (
                    pl.col("major_obstacle")
                    * pl.col("wmedian")
                ).sum()
                / pl.col("wmedian").sum()
            ).alias("weighted_rate_wmedian"),
        ).to_dicts()[0]

        outcome_records.append(
            {
                "comparison": comparison_name,
                "group": group_name,
                **summary,
            }
        )

outcome_diagnostics = pl.DataFrame(outcome_records)

print(outcome_diagnostics)

outcome_diagnostics.write_csv(output_outcomes)

print(f"Output: {output_outcomes}")

print_section("6. SMD INTERPRETATION GUIDE")

print("Absolute SMD below 0.10: small difference")
print("Absolute SMD from 0.10 to below 0.20: moderate difference")
print("Absolute SMD of 0.20 or higher: substantial difference")
print(
    "These are descriptive screening thresholds, "
    "not formal significance tests."
)

print_section("DIAGNOSTICS COMPLETED")

print("No observations were removed from the source dataset.")
print("All comparisons are descriptive.")