from pathlib import Path

import polars as pl
import numpy as np
import svy


project_dir = Path(__file__).resolve().parent.parent

input_file = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023_candidate_analytic.parquet"
)

base_file = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "model_specific_design_audit"
)

output_dir.mkdir(parents=True, exist_ok=True)


def print_section(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def validate_columns(data, columns):
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def prepare_bank_account(data):
    if "bank_account" in data.columns:
        return data

    if not base_file.exists():
        raise FileNotFoundError(
            f"Base dataset not found: {base_file}"
        )

    base_data = pl.read_parquet(base_file)

    validate_columns(
        base_data,
        ["idstd", "k6"],
    )

    lookup = base_data.select(
        ["idstd", "k6"]
    )

    if lookup["idstd"].n_unique() != lookup.height:
        raise ValueError(
            "Duplicate idstd values in base dataset."
        )

    data = data.join(
        lookup,
        on="idstd",
        how="left",
        validate="m:1",
    )

    if data["k6"].null_count() > 0:
        raise ValueError(
            "Some observations could not be matched to k6."
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

    return data


def count_design_structure(sample_data):
    grouped = (
        sample_data
        .group_by("strata")
        .agg(
            pl.col("idstd").n_unique().alias("psu_count"),
            pl.len().alias("observation_count"),
        )
    )

    singleton_count = (
        grouped
        .filter(pl.col("psu_count") == 1)
        .height
    )

    multi_psu_count = (
        grouped
        .filter(pl.col("psu_count") > 1)
        .height
    )

    unique_psu = sample_data["idstd"].n_unique()
    unique_strata = sample_data["strata"].n_unique()

    duplicate_psu_across_strata = (
        sample_data
        .group_by("idstd")
        .agg(
            pl.col("strata").n_unique().alias("strata_count")
        )
        .filter(pl.col("strata_count") > 1)
        .height
    )

    return {
        "observations": sample_data.height,
        "unique_psu": unique_psu,
        "unique_strata": unique_strata,
        "singleton_strata": singleton_count,
        "multi_psu_strata": multi_psu_count,
        "psu_minus_strata": unique_psu - unique_strata,
        "duplicate_psu_across_strata": (
            duplicate_psu_across_strata
        ),
    }


def inspect_subset_singletons(
    sample_data,
    weight_name,
):
    design = svy.Design(
        stratum="strata",
        psu="idstd",
        wgt=weight_name,
        singleton="scale",
    )

    subset_sample = svy.Sample(
        data=sample_data,
        design=design,
    )

    method = getattr(
        subset_sample.design.singleton,
        "method",
        None,
    )

    singleton_table = subset_sample.singletons

    singleton_count = len(singleton_table)

    scale_count = None
    non_scale_count = None

    if isinstance(singleton_table, pl.DataFrame):
        if "handled" in singleton_table.columns:
            scale_count = (
                singleton_table
                .filter(pl.col("handled") == "scale")
                .height
            )

            non_scale_count = (
                singleton_count - scale_count
            )

    return {
        "subset_singleton_count": singleton_count,
        "subset_scale_count": scale_count,
        "subset_non_scale_count": non_scale_count,
        "subset_singleton_method": method,
    }


def inspect_model(
    model,
    model_name,
    weight_name,
    sample_data,
    condition_name,
):
    fitted = model.fitted
    stats = fitted.stats

    covariance = np.asarray(
        fitted.cov_matrix,
        dtype=float,
    )

    coefficients = fitted.coefs

    coefficient_count = len(coefficients)

    standard_errors = np.asarray(
        [float(coef.se) for coef in coefficients],
        dtype=float,
    )

    estimates = np.asarray(
        [float(coef.est) for coef in coefficients],
        dtype=float,
    )

    covariance_shape_valid = (
        covariance.shape
        == (coefficient_count, coefficient_count)
    )

    finite_covariance = bool(
        np.isfinite(covariance).all()
    )

    symmetry_error = float(
        np.max(np.abs(covariance - covariance.T))
    )

    diagonal_error = float(
        np.max(
            np.abs(
                np.diag(covariance)
                - standard_errors ** 2
            )
        )
    )

    symmetric_covariance = (
        covariance + covariance.T
    ) / 2

    eigenvalues = np.linalg.eigvalsh(
        symmetric_covariance
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    maximum_eigenvalue = float(
        np.max(eigenvalues)
    )

    tolerance = (
        1e-10 * max(1.0, abs(maximum_eigenvalue))
    )

    positive_semidefinite = (
        minimum_eigenvalue >= -tolerance
    )

    model_n = int(stats.n)

    model_df_den = float(
        stats.wald_adj.df_den
    )

    model_df_num = float(
        stats.wald_adj.df_num
    )

    sample_structure = count_design_structure(
        sample_data
    )

    coefficient_finite = bool(
        np.isfinite(estimates).all()
        and np.isfinite(standard_errors).all()
    )

    inference_results = []

    for coef in coefficients:
        estimate = float(coef.est)
        standard_error = float(coef.se)

        wald_statistic = float(
            coef.wald.value
        )

        p_value = float(
            coef.wald.p_value
        )

        wald_df = float(
            coef.wald.df
        )

        expected_wald = (
            estimate / standard_error
            if standard_error != 0
            else np.nan
        )

        inference_results.append({
            "model": model_name,
            "weight": weight_name,
            "term": coef.term,
            "estimate": estimate,
            "standard_error": standard_error,
            "wald_statistic": wald_statistic,
            "expected_wald_statistic": expected_wald,
            "p_value": p_value,
            "wald_df": wald_df,
            "wald_consistent": bool(
                np.isclose(
                    wald_statistic,
                    expected_wald,
                    rtol=1e-8,
                    atol=1e-10,
                )
            ),
            "finite_estimate": bool(
                np.isfinite(estimate)
            ),
            "finite_standard_error": bool(
                np.isfinite(standard_error)
            ),
            "finite_p_value": bool(
                np.isfinite(p_value)
            ),
        })

    result = {
        "model": model_name,
        "weight": weight_name,
        "condition": condition_name,
        "sample_n": sample_data.height,
        "model_reported_n": model_n,
        "n_matches": model_n == sample_data.height,
        "unique_psu": sample_structure["unique_psu"],
        "unique_strata": sample_structure["unique_strata"],
        "singleton_strata": (
            sample_structure["singleton_strata"]
        ),
        "multi_psu_strata": (
            sample_structure["multi_psu_strata"]
        ),
        "psu_minus_strata": (
            sample_structure["psu_minus_strata"]
        ),
        "model_df_den": model_df_den,
        "model_df_num": model_df_num,
        "df_difference": (
            sample_structure["psu_minus_strata"]
            - model_df_den
        ),
        "duplicate_psu_across_strata": (
            sample_structure[
                "duplicate_psu_across_strata"
            ]
        ),
        "coefficient_count": coefficient_count,
        "covariance_shape_valid": covariance_shape_valid,
        "finite_covariance": finite_covariance,
        "symmetry_error": symmetry_error,
        "diagonal_error": diagonal_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "positive_semidefinite": positive_semidefinite,
        "finite_coefficients": coefficient_finite,
    }

    return result, inference_results


print_section("1. LOAD ANALYTIC DATA")

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}"
    )

data = pl.read_parquet(input_file)

print(f"Input file: {input_file}")
print(f"Rows: {data.height:,}")
print(f"Columns: {data.width:,}")

data = prepare_bank_account(data)


print_section("2. VALIDATE DESIGN VARIABLES")

required_columns = [
    "idstd",
    "strata",
    "wmedian",
    "wweak",
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
    "bank_account",
]

validate_columns(
    data,
    required_columns,
)

if data["idstd"].null_count() > 0:
    raise ValueError("Missing PSU identifiers.")

if data["strata"].null_count() > 0:
    raise ValueError("Missing stratum identifiers.")

if data["idstd"].n_unique() != data.height:
    raise ValueError(
        "idstd is not unique in the analytic dataset."
    )

for weight_name in ["wmedian", "wweak"]:
    invalid_weights = data.filter(
        pl.col(weight_name).is_null()
        | (~pl.col(weight_name).is_finite())
        | (pl.col(weight_name) <= 0)
    ).height

    if invalid_weights > 0:
        raise ValueError(
            f"Invalid observations in {weight_name}: "
            f"{invalid_weights}"
        )

print("PSU, strata, and both survey weights passed validation.")


print_section("3. RECREATE COMPLETE-CASE FLAGS")

model1_variables = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
]

model2_variables = (
    model1_variables + ["bank_account"]
)

model1_expression = pl.all_horizontal(
    [
        pl.col(column).is_not_null()
        for column in model1_variables
    ]
)

model2_expression = pl.all_horizontal(
    [
        pl.col(column).is_not_null()
        for column in model2_variables
    ]
)

data = data.with_columns([
    model1_expression.alias("model1_complete"),
    model2_expression.alias("model2_complete"),
])

sample_definitions = [
    (
        "model1_full",
        "model1_complete",
    ),
    (
        "model1_common",
        "model2_complete",
    ),
    (
        "model2_common",
        "model2_complete",
    ),
]

for model_name, condition_name in sample_definitions:
    n = data.filter(
        pl.col(condition_name)
    ).height

    print(f"{model_name}: {n:,}")


print_section("4. DEFINE MODEL PREDICTORS")

model1_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
    svy.Cat("a4a"),
    svy.Cat("a2"),
]

model2_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
    svy.Cat("a4a"),
    svy.Cat("a2"),
]


print_section("5. RUN MODEL-SPECIFIC DESIGN AUDIT")

design_results = []
inference_results = []
singleton_results = []

for weight_name in ["wmedian", "wweak"]:

    print_section(
        f"WEIGHT: {weight_name}"
    )

    full_design = svy.Design(
        stratum="strata",
        psu="idstd",
        wgt=weight_name,
        singleton="scale",
    )

    full_sample = svy.Sample(
        data=data,
        design=full_design,
    )

    global_singletons = full_sample.singletons

    global_singleton_count = len(
        global_singletons
    )

    print(
        f"Full-data singleton strata: "
        f"{global_singleton_count}"
    )

    global_method = getattr(
        full_sample.design.singleton,
        "method",
        None,
    )

    print(
        f"Full-data singleton method: "
        f"{global_method}"
    )

    model_objects = {}

    model_objects["model1_full"] = (
        full_sample.glm.fit(
            y="major_obstacle",
            x=model1_predictors,
            family="binomial",
            link="logit",
            where=(
                svy.col("model1_complete") == True
            ),
        )
    )

    model_objects["model1_common"] = (
        full_sample.glm.fit(
            y="major_obstacle",
            x=model1_predictors,
            family="binomial",
            link="logit",
            where=(
                svy.col("model2_complete") == True
            ),
        )
    )

    model_objects["model2_common"] = (
        full_sample.glm.fit(
            y="major_obstacle",
            x=model2_predictors,
            family="binomial",
            link="logit",
            where=(
                svy.col("model2_complete") == True
            ),
        )
    )

    for model_name, condition_name in sample_definitions:

        print_section(
            f"{model_name} | {weight_name}"
        )

        sample_data = data.filter(
            pl.col(condition_name)
        )

        structure = count_design_structure(
            sample_data
        )

        subset_singletons = inspect_subset_singletons(
            sample_data,
            weight_name,
        )

        singleton_results.append({
            "model": model_name,
            "weight": weight_name,
            **structure,
            **subset_singletons,
        })

        print("Sample structure:")
        print(structure)

        print("Subset singleton audit:")
        print(subset_singletons)

        model = model_objects[model_name]

        design_result, coefficient_results = (
            inspect_model(
                model=model,
                model_name=model_name,
                weight_name=weight_name,
                sample_data=sample_data,
                condition_name=condition_name,
            )
        )

        design_result["global_singleton_count"] = (
            global_singleton_count
        )

        design_result["global_singleton_method"] = (
            global_method
        )

        design_results.append(
            design_result
        )

        inference_results.extend(
            coefficient_results
        )

        print()
        print("Model inference diagnostics:")
        print(design_result)


print_section("6. SAVE AUDIT OUTPUTS")

design_table = pl.DataFrame(
    design_results
)

singleton_table = pl.DataFrame(
    singleton_results
)

inference_table = pl.DataFrame(
    inference_results
)

design_file = (
    output_dir
    / "model_specific_design_summary.csv"
)

singleton_file = (
    output_dir
    / "model_specific_singleton_summary.csv"
)

inference_file = (
    output_dir
    / "model_specific_inference_validation.csv"
)

design_table.write_csv(design_file)
singleton_table.write_csv(singleton_file)
inference_table.write_csv(inference_file)

print(f"Saved: {design_file}")
print(f"Saved: {singleton_file}")
print(f"Saved: {inference_file}")


print_section("7. DESIGN AUDIT SUMMARY")

print(
    design_table.select([
        "model",
        "weight",
        "sample_n",
        "model_reported_n",
        "n_matches",
        "unique_psu",
        "unique_strata",
        "singleton_strata",
        "psu_minus_strata",
        "model_df_den",
        "df_difference",
        "coefficient_count",
        "covariance_shape_valid",
        "finite_covariance",
        "diagonal_error",
        "positive_semidefinite",
    ])
)


print_section("8. SINGLETON AUDIT SUMMARY")

print(
    singleton_table.select([
        "model",
        "weight",
        "observations",
        "unique_psu",
        "unique_strata",
        "singleton_strata",
        "multi_psu_strata",
        "subset_singleton_count",
        "subset_scale_count",
        "subset_non_scale_count",
        "subset_singleton_method",
    ])
)


print_section("9. INFERENCE VALIDATION SUMMARY")

print(
    inference_table.group_by(
        ["model", "weight"]
    ).agg([
        pl.len().alias("coefficient_count"),
        pl.col("finite_estimate").all().alias(
            "all_estimates_finite"
        ),
        pl.col("finite_standard_error").all().alias(
            "all_standard_errors_finite"
        ),
        pl.col("finite_p_value").all().alias(
            "all_p_values_finite"
        ),
        pl.col("wald_consistent").all().alias(
            "all_wald_statistics_consistent"
        ),
    ]).sort(["weight", "model"])
)


print_section("10. FINAL AUDIT STATUS")

print(
    "The original full-data survey design was retained "
    "for model estimation."
)

print(
    "Complete-case design structures were audited "
    "separately for each model."
)

print(
    "Subset-only singleton diagnostics are diagnostic "
    "and do not replace the original domain-estimation design."
)

print(
    "No previously validated Wald test results were overwritten."
)

print()
print(f"Output directory: {output_dir}")

print()
print("66 MODEL-SPECIFIC SURVEY DESIGN AUDIT COMPLETED")