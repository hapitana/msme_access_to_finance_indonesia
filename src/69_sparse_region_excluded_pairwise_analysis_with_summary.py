from pathlib import Path
import os
import re
import runpy
import shutil
import sys
import tempfile

import svy
import svy_rs

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import t as t_distribution


project_dir = Path(__file__).resolve().parents[1]

script55_path = (
    project_dir
    / "src"
    / "55_survey_weighted_logistic_regression.py"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "outcome_robustness"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

if not script55_path.exists():
    raise FileNotFoundError(
        f"Script 55 not found: {script55_path}"
    )


print("=" * 78)
print("24C-29. SPARSE-REGION-EXCLUDED REGIONAL PAIRWISE CONTRAST ANALYSIS")
print("=" * 78)


print()
print("STEP 1. VERIFY ENVIRONMENT AND REBUILD THE ORIGINAL GEOGRAPHIC MODELS")

expected_svy_version = "0.35.0"
expected_svy_rs_version = "0.19.1"

if getattr(svy, "__version__", None) != expected_svy_version:
    raise RuntimeError(
        "Unexpected svy version. "
        f"Expected {expected_svy_version}; found {getattr(svy, '__version__', 'unknown')}."
    )

if getattr(svy_rs, "__version__", None) != expected_svy_rs_version:
    raise RuntimeError(
        "Unexpected svy_rs version. "
        f"Expected {expected_svy_rs_version}; found {getattr(svy_rs, '__version__', 'unknown')}."
    )

print(f"Python version: {sys.version.split()[0]}")
print(f"svy version: {svy.__version__}")
print(f"svy_rs version: {svy_rs.__version__}")
print(f"Polars version: {pl.__version__}")
print(f"NumPy version: {np.__version__}")

# Script 55 writes many audited outputs while rebuilding its in-memory
# survey objects. Preserve the existing project outputs and restore them
# after Script 55 finishes, before writing the new 24C-29 results.
processed_dir = project_dir / "data" / "processed"
output_folders = [
    processed_dir / "script55_survey_weighted_logistic_regression",
    processed_dir / "outcome_robustness",
]

original_cwd = Path.cwd()
with tempfile.TemporaryDirectory(prefix="24c29_script55_backup_") as backup_name:
    backup_root = Path(backup_name)
    existing_folders = []

    for folder in output_folders:
        if folder.exists():
            backup_folder = backup_root / folder.name
            shutil.copytree(folder, backup_folder)
            existing_folders.append((folder, backup_folder))

    try:
        os.chdir(project_dir)
        namespace = runpy.run_path(str(script55_path))
    finally:
        os.chdir(original_cwd)
        for folder in output_folders:
            if folder.exists():
                shutil.rmtree(folder)
        for folder, backup_folder in existing_folders:
            shutil.copytree(backup_folder, folder)

required_objects = [
    "data_geo",
    "svy",
    "geo_designs",
    "geo_model1_predictors",
    "geo_model2_predictors",
]

missing_objects = [
    object_name
    for object_name in required_objects
    if object_name not in namespace
]

if missing_objects:
    raise KeyError(
        f"Required objects not found in Script 55 namespace: {missing_objects}"
    )

data_geo = namespace["data_geo"]
svy = namespace["svy"]
geo_designs = namespace["geo_designs"]
geo_model1_predictors = namespace["geo_model1_predictors"]
geo_model2_predictors = namespace["geo_model2_predictors"]

required_geo_columns = [
    "idstd",
    "strata",
    "a3a",
    "a4a",
    "major_obstacle",
    "any_obstacle",
    "model1_complete",
    "model2_complete",
    "wmedian",
    "wweak",
]
missing_geo_columns = [
    column
    for column in required_geo_columns
    if column not in data_geo.columns
]
if missing_geo_columns:
    raise KeyError(
        f"Required columns missing from Script 55 data_geo: {missing_geo_columns}"
    )

if set(geo_designs) != {"wmedian", "wweak"}:
    raise ValueError(
        f"Unexpected survey design keys: {sorted(geo_designs)}"
    )

for weight_name, design_obj in geo_designs.items():
    if getattr(design_obj, "wgt", None) != weight_name:
        raise ValueError(
            f"Survey design weight mismatch for {weight_name}: "
            f"{getattr(design_obj, 'wgt', None)}"
        )

sparse_region_codes = [2, 5, 10, 13, 16, 18, 19]

original_region_counts = (
    data_geo
    .group_by("a3a")
    .agg(pl.len().alias("n"))
    .sort("a3a")
)

data_geo = data_geo.filter(
    ~pl.col("a3a").cast(pl.Int64).is_in(sparse_region_codes)
)

remaining_regions = (
    data_geo
    .get_column("a3a")
    .cast(pl.Int64)
    .unique()
    .sort()
    .to_list()
)

if len(remaining_regions) != 15 or 1 not in remaining_regions:
    raise ValueError(
        "Sparse-region exclusion did not leave the expected 15 regions "
        "including Jakarta. Check the a3a geography codes."
    )

geo_samples = {
    weight_name: svy.Sample(data_geo, design_obj)
    for weight_name, design_obj in geo_designs.items()
}

geo_results = {}

for outcome_name in ["major_obstacle", "any_obstacle"]:
    for weight_name, sample_obj in geo_samples.items():
        geo_results[f"{outcome_name}_{weight_name}"] = {
            f"model1_full_{weight_name}": sample_obj.glm.fit(
                y=outcome_name,
                x=geo_model1_predictors,
                family="binomial",
                link="logit",
                where=(svy.col("model1_complete") == True),
            ),
            f"model1_common_{weight_name}": sample_obj.glm.fit(
                y=outcome_name,
                x=geo_model1_predictors,
                family="binomial",
                link="logit",
                where=(svy.col("model2_complete") == True),
            ),
            f"model2_common_{weight_name}": sample_obj.glm.fit(
                y=outcome_name,
                x=geo_model2_predictors,
                family="binomial",
                link="logit",
                where=(svy.col("model2_complete") == True),
            ),
        }

print("Sparse-region-excluded geographic models re-estimated.")
print(f"Excluded a3a codes: {sparse_region_codes}")
print(f"Remaining a3a codes: {remaining_regions}")
print(f"Geographic dataset shape after exclusion: {data_geo.shape}")


print()
print("STEP 2. DEFINE ANALYSIS SETTINGS")

region_reference = 1

expected_regions = remaining_regions

outcomes = [
    "any_obstacle",
    "major_obstacle",
]

weights = [
    "wmedian",
    "wweak",
]

model_names = [
    "model1_full",
    "model1_common",
    "model2_common",
]

model_sample_flags = {
    "model1_full": "model1_complete",
    "model1_common": "model2_complete",
    "model2_common": "model2_complete",
}

region_values = (
    data_geo
    .get_column("a3a")
    .cast(pl.Float64)
    .unique()
    .sort()
    .to_list()
)

if region_values != [
    float(value)
    for value in expected_regions
]:
    raise ValueError(
        "Unexpected establishment-region categories. "
        f"Found: {region_values}"
    )

if region_reference not in expected_regions:
    raise ValueError(
        "The reference region is not present."
    )

print("Reference region: Jakarta (a3a = 1)")
print(f"Number of retained regions: {len(expected_regions)}")
print(f"Pairwise contrasts per family: {len(expected_regions) * (len(expected_regions) - 1) // 2}")


print()
print("STEP 2A. AUDIT THE a3a-a4a GEOGRAPHIC CROSSWALK")

if "a4a" not in data_geo.columns:
    raise KeyError("a4a is required for the geographic crosswalk audit.")

geographic_crosswalk = (
    data_geo
    .select(
        pl.col("a3a").cast(pl.Int64, strict=False),
        pl.col("a4a").cast(pl.Float64, strict=False),
    )
    .drop_nulls()
    .group_by("a4a")
    .agg(
        pl.col("a3a").n_unique().alias("number_of_regions"),
        pl.col("a3a").unique().sort().alias("region_codes_list"),
        pl.len().alias("observations"),
    )
    .with_columns(
        pl.col("region_codes_list")
        .list.eval(pl.element().cast(pl.String))
        .list.join(";")
        .alias("region_codes")
    )
    .drop("region_codes_list")
    .sort("a4a")
)

nested_a4a_codes = geographic_crosswalk.filter(
    pl.col("number_of_regions") == 1
).height
cross_region_a4a_codes = geographic_crosswalk.filter(
    pl.col("number_of_regions") > 1
).height

crosswalk_file = output_dir / "24c29_a3a_a4a_geographic_crosswalk_audit.csv"
geographic_crosswalk.write_csv(crosswalk_file)

print(f"Distinct non-missing a4a codes: {geographic_crosswalk.height}")
print(f"a4a codes observed in exactly one a3a region: {nested_a4a_codes}")
print(f"a4a codes observed in multiple a3a regions: {cross_region_a4a_codes}")
print(f"Crosswalk audit saved: {crosswalk_file}")

# The a3a-a4a crosswalk is retained as a diagnostic. Regional predictions
# below use each region's own observed estimation-sample covariate distribution.
# This avoids constructing impossible combinations by assigning a different
# a3a while retaining a4a values from other regions. The resulting estimand is
# a within-region, survey-weighted average predicted probability, not a
# common-covariate-distribution marginal standardization.


print()
print("STEP 3. DEFINE DESIGN MATRIX CONSTRUCTION")


def build_design_matrix(
    sample_frame,
    feature_names,
    counterfactual_region,
):
    matrix_columns = []

    intercept_names = {
        "_intercept_",
        "intercept",
        "const",
        "(intercept)",
    }

    for feature_name in feature_names:

        feature_name = str(feature_name)

        feature_lower = feature_name.lower()

        if feature_lower in intercept_names:

            column_values = np.ones(
                sample_frame.height,
                dtype=np.float64,
            )

        else:

            categorical_match = re.fullmatch(
                r"(a3a|a4a)_(.+)",
                feature_name,
            )

            if categorical_match is not None:

                variable_name = (
                    categorical_match.group(1)
                )

                category_value = float(
                    categorical_match.group(2)
                )

                if variable_name == "a3a":

                    column_values = np.full(
                        sample_frame.height,
                        float(counterfactual_region),
                        dtype=np.float64,
                    )

                else:

                    if "a4a" not in sample_frame.columns:
                        raise KeyError(
                            "Categorical variable a4a "
                            "not found in estimation data."
                        )

                    column_values = np.asarray(
                        sample_frame
                        .get_column("a4a")
                        .cast(pl.Float64)
                        .to_list(),
                        dtype=np.float64,
                    )

                column_values = (
                    np.isclose(
                        column_values,
                        category_value,
                    )
                ).astype(np.float64)

            else:

                if feature_name not in sample_frame.columns:
                    raise KeyError(
                        "Feature not found in estimation data: "
                        f"{feature_name}"
                    )

                column_values = np.asarray(
                    sample_frame
                    .get_column(feature_name)
                    .cast(pl.Float64)
                    .to_list(),
                    dtype=np.float64,
                )

        if column_values.shape != (
            sample_frame.height,
        ):
            raise ValueError(
                "Unexpected feature column dimensions: "
                f"{feature_name}"
            )

        if not np.isfinite(column_values).all():
            raise ValueError(
                f"Non-finite values in feature: {feature_name}"
            )

        matrix_columns.append(
            column_values
        )

    matrix = np.column_stack(
        matrix_columns
    )

    if matrix.shape != (
        sample_frame.height,
        len(feature_names),
    ):
        raise ValueError(
            "Unexpected design matrix dimensions."
        )

    return matrix


print("Design matrix construction function ready.")
print("Intercept handling includes _intercept_.")


print()
print("STEP 4. DEFINE WEIGHTED PREDICTION FUNCTION")


def calculate_weighted_prediction(
    matrix,
    beta,
    survey_weights,
    covariance,
):
    linear_predictor = matrix @ beta

    predicted_probability = expit(
        linear_predictor
    )

    weighted_probability = float(
        np.average(
            predicted_probability,
            weights=survey_weights,
        )
    )

    individual_gradient = (
        predicted_probability
        * (1.0 - predicted_probability)
    )[:, None] * matrix

    gradient = np.average(
        individual_gradient,
        axis=0,
        weights=survey_weights,
    )

    variance = float(
        gradient
        @ covariance
        @ gradient
    )

    if variance < -1e-10:
        raise ValueError(
            "Negative prediction variance detected."
        )

    standard_error = float(
        np.sqrt(
            max(variance, 0.0)
        )
    )

    return (
        weighted_probability,
        standard_error,
        gradient,
    )


print("Weighted prediction function ready.")
print("Estimand: within-region weighted average predicted probability.")


print()
print("STEP 5. ESTIMATE REGIONAL EFFECTS")

prediction_rows = []
odds_ratio_rows = []
sample_rows = []
pairwise_rows = []

for outcome_name in outcomes:

    for weight_name in weights:

        model_key = (
            f"{outcome_name}_{weight_name}"
        )

        if model_key not in geo_results:
            raise KeyError(
                f"Model group not found: {model_key}"
            )

        fitted_models = geo_results[
            model_key
        ]

        for model_name in model_names:

            fitted_model_key = (
                f"{model_name}_{weight_name}"
            )

            if fitted_model_key not in fitted_models:
                raise KeyError(
                    "Fitted model not found: "
                    f"{fitted_model_key}"
                )

            fitted_model = fitted_models[
                fitted_model_key
            ]

            fitted = fitted_model.fitted

            sample_flag = model_sample_flags[
                model_name
            ]

            required_sample_columns = [
                sample_flag,
                outcome_name,
                weight_name,
                "a3a",
                "a4a",
            ]

            missing_sample_columns = [
                column
                for column in required_sample_columns
                if column not in data_geo.columns
            ]

            if missing_sample_columns:
                raise KeyError(
                    "Required sample columns not found: "
                    f"{missing_sample_columns}"
                )

            sample_frame = (
                data_geo
                .filter(
                    pl.col(sample_flag)
                    & pl.col(outcome_name).is_not_null()
                    & pl.col(weight_name).is_not_null()
                    & pl.col(weight_name).is_finite()
                    & (pl.col(weight_name) > 0)
                )
            )

            if sample_frame.height == 0:
                raise ValueError(
                    "Empty estimation sample: "
                    f"{model_key}, {model_name}"
                )

            survey_weights = np.asarray(
                sample_frame
                .get_column(weight_name)
                .cast(pl.Float64)
                .to_list(),
                dtype=np.float64,
            )

            if not np.isfinite(survey_weights).all():
                raise ValueError(
                    f"Non-finite survey weights: {model_key}"
                )

            if (survey_weights <= 0).any():
                raise ValueError(
                    f"Non-positive survey weights: {model_key}"
                )

            weight_sum = float(
                survey_weights.sum()
            )

            kish_effective_n = float(
                weight_sum ** 2
                / np.sum(survey_weights ** 2)
            )

            feature_names = [
                str(feature)
                for feature in fitted.feature_names
            ]

            if not feature_names:
                raise ValueError(
                    f"No model features found: {model_name}"
                )

            coefficients = fitted.coefs

            coefficient_lookup = {
                str(coefficient.term): coefficient
                for coefficient in coefficients
            }

            missing_features = [
                feature
                for feature in feature_names
                if feature not in coefficient_lookup
            ]

            if missing_features:
                raise ValueError(
                    "Missing model coefficients: "
                    f"{missing_features}"
                )

            beta = np.asarray(
                [
                    coefficient_lookup[
                        feature
                    ].est
                    for feature in feature_names
                ],
                dtype=np.float64,
            )

            covariance = np.asarray(
                fitted.cov_matrix,
                dtype=np.float64,
            )

            if covariance.shape != (
                len(feature_names),
                len(feature_names),
            ):
                raise ValueError(
                    "Covariance matrix dimensions do not match "
                    f"the feature list in {model_name}."
                )

            covariance = (
                covariance + covariance.T
            ) / 2.0

            if not np.isfinite(beta).all():
                raise ValueError(
                    f"Non-finite coefficients in {model_name}."
                )

            if not np.isfinite(covariance).all():
                raise ValueError(
                    f"Non-finite covariance matrix in {model_name}."
                )

            df_den = float(
                fitted._design_df()
            )

            if df_den <= 0:
                raise ValueError(
                    f"Invalid design degrees of freedom: {df_den}"
                )

            critical_value = float(
                t_distribution.ppf(
                    0.975,
                    df_den,
                )
            )

            sample_rows.append(
                {
                    "outcome": outcome_name,
                    "weight": weight_name,
                    "model": model_name,
                    "sample_n": sample_frame.height,
                    "kish_effective_n": kish_effective_n,
                    "weight_sum": weight_sum,
                    "design_df": df_den,
                }
            )

            print()
            print("-" * 72)
            print(
                f"OUTCOME: {outcome_name} | "
                f"WEIGHT: {weight_name} | "
                f"MODEL: {model_name}"
            )
            print(
                f"Sample N: {sample_frame.height}"
            )
            print(
                f"Kish effective N: {kish_effective_n:.2f}"
            )
            print(
                f"Design df: {df_den:.0f}"
            )

            model_prediction_results = {}

            for region_code in expected_regions:

                regional_sample = sample_frame.filter(
                    pl.col("a3a").cast(pl.Int64) == region_code
                )

                if regional_sample.height == 0:
                    raise ValueError(
                        "No estimation-sample observations for region "
                        f"{region_code} in {model_key}/{model_name}."
                    )

                regional_weights = np.asarray(
                    regional_sample
                    .get_column(weight_name)
                    .cast(pl.Float64)
                    .to_list(),
                    dtype=np.float64,
                )

                regional_weight_sum = float(regional_weights.sum())
                regional_kish_n = float(
                    regional_weight_sum ** 2
                    / np.sum(regional_weights ** 2)
                )

                matrix = build_design_matrix(
                    sample_frame=regional_sample,
                    feature_names=feature_names,
                    counterfactual_region=region_code,
                )

                (
                    predicted_probability,
                    prediction_se,
                    gradient,
                ) = calculate_weighted_prediction(
                    matrix=matrix,
                    beta=beta,
                    survey_weights=regional_weights,
                    covariance=covariance,
                )

                ci_low = max(
                    0.0,
                    predicted_probability
                    - critical_value * prediction_se,
                )

                ci_high = min(
                    1.0,
                    predicted_probability
                    + critical_value * prediction_se,
                )

                model_prediction_results[
                    region_code
                ] = {
                    "probability": predicted_probability,
                    "standard_error": prediction_se,
                    "gradient": gradient,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "regional_sample_n": regional_sample.height,
                    "regional_kish_effective_n": regional_kish_n,
                    "regional_weight_sum": regional_weight_sum,
                }

            reference_result = (
                model_prediction_results[
                    region_reference
                ]
            )

            # All unordered pairs among the 15 retained regions are tested within each
            # outcome x weight x model family.
            for first_index, first_region in enumerate(expected_regions):
                for second_region in expected_regions[first_index + 1:]:
                    first_result = model_prediction_results[first_region]
                    second_result = model_prediction_results[second_region]

                    pair_difference = (
                        first_result["probability"]
                        - second_result["probability"]
                    )
                    pair_gradient = (
                        first_result["gradient"]
                        - second_result["gradient"]
                    )
                    pair_variance = float(
                        pair_gradient @ covariance @ pair_gradient
                    )

                    if pair_variance < -1e-10:
                        raise ValueError(
                            "Negative pairwise contrast variance detected."
                        )

                    pair_se = float(np.sqrt(max(pair_variance, 0.0)))
                    pair_ci_low = pair_difference - critical_value * pair_se
                    pair_ci_high = pair_difference + critical_value * pair_se

                    if pair_se > 0:
                        pair_t = pair_difference / pair_se
                        pair_p = float(
                            2.0 * t_distribution.sf(abs(pair_t), df_den)
                        )
                    else:
                        pair_p = float("nan")

                    pairwise_rows.append(
                        {
                            "analysis": "excluding_7_sparse_regions",
                            "excluded_region_codes": ";".join(str(x) for x in sparse_region_codes),
                            "retained_region_count": len(expected_regions),
                            "outcome": outcome_name,
                            "weight": weight_name,
                            "model": model_name,
                            "region_1_code": first_region,
                            "region_2_code": second_region,
                            "difference_probability": pair_difference,
                            "difference_percentage_points": pair_difference * 100.0,
                            "standard_error": pair_se,
                            "ci_low": pair_ci_low,
                            "ci_high": pair_ci_high,
                            "ci_low_percentage_points": pair_ci_low * 100.0,
                            "ci_high_percentage_points": pair_ci_high * 100.0,
                            "p_value_raw": pair_p,
                            "design_df": df_den,
                            "ci_method": "conditional_coefficient_covariance_delta_method",
                            "ci_scale": "probability; percentage-point bounds also provided",
                            "estimand": "difference_between_within_region_weighted_average_predicted_probabilities",
                            "inference_scope": "conditional_on_region_specific_observed_covariate_distributions",
                        }
                    )

            for region_code in expected_regions:

                prediction_result = (
                    model_prediction_results[
                        region_code
                    ]
                )

                probability = (
                    prediction_result[
                        "probability"
                    ]
                )

                probability_se = (
                    prediction_result[
                        "standard_error"
                    ]
                )

                probability_gradient = (
                    prediction_result[
                        "gradient"
                    ]
                )

                probability_difference = (
                    probability
                    - reference_result["probability"]
                )

                difference_gradient = (
                    probability_gradient
                    - reference_result["gradient"]
                )

                difference_variance = float(
                    difference_gradient
                    @ covariance
                    @ difference_gradient
                )

                if difference_variance < -1e-10:
                    raise ValueError(
                        "Negative regional contrast variance detected."
                    )

                difference_se = float(
                    np.sqrt(
                        max(
                            difference_variance,
                            0.0,
                        )
                    )
                )

                difference_ci_low = (
                    probability_difference
                    - critical_value * difference_se
                )

                difference_ci_high = (
                    probability_difference
                    + critical_value * difference_se
                )

                if difference_se > 0:

                    difference_t = (
                        probability_difference
                        / difference_se
                    )

                    difference_p = float(
                        2.0
                        * t_distribution.sf(
                            abs(difference_t),
                            df_den,
                        )
                    )

                else:
                    difference_t = float("nan")
                    difference_p = float("nan")

                prediction_rows.append(
                    {
                        "outcome": outcome_name,
                        "weight": weight_name,
                        "model": model_name,
                        "region_code": region_code,
                        "reference_region": region_reference,
                        "sample_n": sample_frame.height,
                        "kish_effective_n": kish_effective_n,
                        "regional_sample_n": prediction_result["regional_sample_n"],
                        "regional_kish_effective_n": prediction_result["regional_kish_effective_n"],
                        "regional_weight_sum": prediction_result["regional_weight_sum"],
                        "predicted_probability": probability,
                        "predicted_percentage": (
                            probability * 100.0
                        ),
                        "prediction_se": probability_se,
                        "ci_low": prediction_result["ci_low"],
                        "ci_high": prediction_result["ci_high"],
                        "ci_low_percentage": prediction_result["ci_low"] * 100.0,
                        "ci_high_percentage": prediction_result["ci_high"] * 100.0,
                        "difference_vs_jakarta": probability_difference,
                        "difference_vs_jakarta_pp": (
                            probability_difference * 100.0
                        ),
                        "difference_se": difference_se,
                        "difference_ci_low": difference_ci_low,
                        "difference_ci_high": difference_ci_high,
                        "difference_ci_low_pp": difference_ci_low * 100.0,
                        "difference_ci_high_pp": difference_ci_high * 100.0,
                        "difference_p_value": difference_p,
                        "design_df": df_den,
                        "ci_method": "conditional_coefficient_covariance_delta_method",
                        "estimand": "within_region_weighted_average_predicted_probability",
                        "inference_scope": "conditional_on_region_specific_observed_covariate_distribution",
                    }
                )

                if region_code == region_reference:

                    odds_ratio_rows.append(
                        {
                            "outcome": outcome_name,
                            "weight": weight_name,
                            "model": model_name,
                            "region_code": region_code,
                            "reference_region": region_reference,
                            "log_odds_ratio": 0.0,
                            "odds_ratio": 1.0,
                            "or_ci_low": 1.0,
                            "or_ci_high": 1.0,
                            "p_value": None,
                            "comparison": "Reference",
                        }
                    )

                else:

                    matching_terms = [
                        term
                        for term in feature_names
                        if re.fullmatch(
                            rf"a3a_{re.escape(str(float(region_code)))}",
                            term,
                        )
                    ]

                    if len(matching_terms) != 1:
                        raise ValueError(
                            "Could not uniquely identify the regional "
                            f"coefficient for region {region_code}: "
                            f"{matching_terms}"
                        )

                    regional_coefficient = (
                        coefficient_lookup[
                            matching_terms[0]
                        ]
                    )

                    log_or = float(
                        regional_coefficient.est
                    )

                    log_or_low = float(
                        regional_coefficient.lci
                    )

                    log_or_high = float(
                        regional_coefficient.uci
                    )

                    odds_ratio_rows.append(
                        {
                            "outcome": outcome_name,
                            "weight": weight_name,
                            "model": model_name,
                            "region_code": region_code,
                            "reference_region": region_reference,
                            "log_odds_ratio": log_or,
                            "odds_ratio": float(
                                np.exp(log_or)
                            ),
                            "or_ci_low": float(
                                np.exp(log_or_low)
                            ),
                            "or_ci_high": float(
                                np.exp(log_or_high)
                            ),
                            "p_value": float(
                                regional_coefficient.wald.p_value
                            ),
                            "comparison": "Compared with Jakarta",
                        }
                    )

            print(
                "Regional predictions completed:",
                len(model_prediction_results),
            )


print()
print("STEP 6. BUILD OUTPUT DATAFRAMES")

expected_pair_count = len(expected_regions) * (len(expected_regions) - 1) // 2
if len(pairwise_rows) != len(outcomes) * len(weights) * len(model_names) * expected_pair_count:
    raise ValueError(
        "Unexpected number of pairwise regional contrasts: "
        f"{len(pairwise_rows)}; expected "
        f"{len(outcomes) * len(weights) * len(model_names) * expected_pair_count}"
    )

# Holm step-down adjustment, applied separately within each
# outcome x weight x model family (105 hypotheses per family).
for group_key in sorted({
    (row["outcome"], row["weight"], row["model"])
    for row in pairwise_rows
}):
    family = [
        row for row in pairwise_rows
        if (row["outcome"], row["weight"], row["model"]) == group_key
        and np.isfinite(row["p_value_raw"])
    ]
    if len(family) != expected_pair_count:
        raise ValueError(
            f"A Holm family must contain {expected_pair_count} finite pairwise p-values; "
            f"found {len(family)} for {group_key}."
        )
    family_sorted = sorted(family, key=lambda row: row["p_value_raw"])
    running_max = 0.0
    family_size = len(family_sorted)
    for rank, row in enumerate(family_sorted):
        adjusted = min(1.0, (family_size - rank) * row["p_value_raw"])
        running_max = max(running_max, adjusted)
        row["p_value_holm"] = running_max
    for row in pairwise_rows:
        if (row["outcome"], row["weight"], row["model"]) == group_key:
            row.setdefault("p_value_holm", float("nan"))
            row["holm_family_size"] = family_size

pairwise_df = pl.DataFrame(pairwise_rows).sort(
    ["outcome", "model", "weight", "p_value_holm"]
)


predictions_df = (
    pl.DataFrame(prediction_rows)
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "region_code",
        ]
    )
)

odds_ratios_df = (
    pl.DataFrame(odds_ratio_rows)
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "region_code",
        ]
    )
)

samples_df = (
    pl.DataFrame(sample_rows)
    .sort(
        [
            "outcome",
            "model",
            "weight",
        ]
    )
)

expected_prediction_rows = (
    len(outcomes)
    * len(weights)
    * len(model_names)
    * len(expected_regions)
)

expected_or_rows = (
    len(outcomes)
    * len(weights)
    * len(model_names)
    * len(expected_regions)
)

if predictions_df.height != expected_prediction_rows:
    raise ValueError(
        "Unexpected number of predicted probability rows: "
        f"{predictions_df.height}"
    )

if odds_ratios_df.height != expected_or_rows:
    raise ValueError(
        "Unexpected number of odds ratio rows: "
        f"{odds_ratios_df.height}"
    )

if predictions_df.select(
    pl.col("predicted_probability").is_finite().all()
).item() is not True:
    raise ValueError(
        "Non-finite predicted probabilities detected."
    )

if predictions_df.filter(
    (pl.col("predicted_probability") < 0)
    | (pl.col("predicted_probability") > 1)
).height > 0:
    raise ValueError(
        "Predicted probabilities outside the [0, 1] range."
    )


print()
print("STEP 7. WEIGHT SENSITIVITY COMPARISON")

median_predictions = (
    predictions_df
    .filter(
        pl.col("weight") == "wmedian"
    )
    .select(
        "outcome",
        "model",
        "region_code",
        pl.col(
            "predicted_probability"
        ).alias("probability_wmedian"),
    )
)

weak_predictions = (
    predictions_df
    .filter(
        pl.col("weight") == "wweak"
    )
    .select(
        "outcome",
        "model",
        "region_code",
        pl.col(
            "predicted_probability"
        ).alias("probability_wweak"),
    )
)

weight_comparison_df = (
    median_predictions
    .join(
        weak_predictions,
        on=[
            "outcome",
            "model",
            "region_code",
        ],
        how="inner",
        validate="1:1",
    )
    .with_columns(
        (
            pl.col("probability_wmedian")
            - pl.col("probability_wweak")
        ).alias("difference"),
        (
            (
                pl.col("probability_wmedian")
                - pl.col("probability_wweak")
            ).abs()
            * 100.0
        ).alias("absolute_difference_pp"),
    )
    .sort(
        [
            "outcome",
            "model",
            "absolute_difference_pp",
        ],
        descending=[
            False,
            False,
            True,
        ],
    )
)

expected_weight_rows = (
    len(outcomes)
    * len(model_names)
    * len(expected_regions)
)

if weight_comparison_df.height != expected_weight_rows:
    raise ValueError(
        "Unexpected number of weight comparison rows."
    )


print()
print("STEP 8. CREATE REGIONAL RANKINGS")

ranking_df = (
    predictions_df
    .with_columns(
        pl.col(
            "predicted_probability"
        )
        .rank(
            method="ordinal",
            descending=True,
        )
        .over(
            [
                "outcome",
                "model",
                "weight",
            ]
        )
        .alias("risk_rank")
    )
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "risk_rank",
        ]
    )
)


print()
print("STEP 8A. SUMMARIZE HOLM-SIGNIFICANT PAIRWISE CONTRASTS")

significance_threshold = 0.05
expected_model_weight_combinations = len(model_names) * len(weights)

significant_pairwise_df = (
    pairwise_df
    .filter(
        pl.col("p_value_holm").is_finite()
        & (pl.col("p_value_holm") < significance_threshold)
    )
    .with_columns(
        pl.col("difference_percentage_points")
        .abs()
        .alias("absolute_difference_percentage_points"),
        pl.concat_str(
            [pl.col("model"), pl.col("weight")],
            separator="|",
        ).alias("model_weight_combination"),
    )
)

# Count appearances in significant pairs; each pair contributes to both regions.
region_1_appearances = significant_pairwise_df.select(
    "outcome",
    pl.col("region_1_code").alias("region_code"),
    pl.col("region_2_code").alias("counterpart_region_code"),
    "model_weight_combination",
)
region_2_appearances = significant_pairwise_df.select(
    "outcome",
    pl.col("region_2_code").alias("region_code"),
    pl.col("region_1_code").alias("counterpart_region_code"),
    "model_weight_combination",
)

region_significance_frequency_df = (
    pl.concat([region_1_appearances, region_2_appearances], how="vertical")
    .group_by(["outcome", "region_code"])
    .agg(
        pl.len().alias("significant_pair_appearances"),
        pl.col("counterpart_region_code").n_unique().alias("unique_significant_counterparts"),
        pl.col("model_weight_combination").n_unique().alias("model_weight_combinations_with_significance"),
    )
    .sort(
        ["outcome", "significant_pair_appearances", "region_code"],
        descending=[False, True, False],
    )
)

# A robust pair must be Holm-significant in all six combinations and retain
# the same direction in every combination.
pairwise_robustness_df = (
    pairwise_df
    .group_by(["outcome", "region_1_code", "region_2_code"])
    .agg(
        pl.len().alias("model_weight_combinations_tested"),
        (pl.col("p_value_holm") < significance_threshold).sum().alias("holm_significant_combinations"),
        pl.col("difference_percentage_points").mean().alias("mean_difference_percentage_points"),
        pl.col("difference_percentage_points").min().alias("minimum_signed_difference_percentage_points"),
        pl.col("difference_percentage_points").max().alias("maximum_signed_difference_percentage_points"),
        pl.col("difference_percentage_points").abs().mean().alias("mean_absolute_difference_percentage_points"),
        pl.col("difference_percentage_points").abs().min().alias("minimum_absolute_difference_percentage_points"),
        pl.col("difference_percentage_points").abs().max().alias("maximum_absolute_difference_percentage_points"),
        (pl.col("difference_percentage_points") > 0).sum().alias("positive_direction_combinations"),
        (pl.col("difference_percentage_points") < 0).sum().alias("negative_direction_combinations"),
    )
    .with_columns(
        (
            (pl.col("model_weight_combinations_tested") == expected_model_weight_combinations)
            & (pl.col("holm_significant_combinations") == expected_model_weight_combinations)
            & (
                (pl.col("positive_direction_combinations") == expected_model_weight_combinations)
                | (pl.col("negative_direction_combinations") == expected_model_weight_combinations)
            )
        ).alias("robust_significant_all_six")
    )
    .sort(
        ["outcome", "robust_significant_all_six", "mean_absolute_difference_percentage_points"],
        descending=[False, True, True],
    )
)

robust_pairwise_df = pairwise_robustness_df.filter(pl.col("robust_significant_all_six"))

# Ten largest Holm-significant contrasts within each outcome x weight x model family.
largest_significant_pairwise_df = (
    significant_pairwise_df
    .sort(
        ["outcome", "weight", "model", "absolute_difference_percentage_points"],
        descending=[False, False, False, True],
    )
    .group_by(["outcome", "weight", "model"], maintain_order=True)
    .head(10)
    .sort(
        ["outcome", "weight", "model", "absolute_difference_percentage_points"],
        descending=[False, False, False, True],
    )
)

region_frequency_file = output_dir / "24c29_region_holm_significance_frequency.csv"
robust_pairs_file = output_dir / "24c29_pairwise_robust_all_six_combinations.csv"
largest_pairs_file = output_dir / "24c29_largest_holm_significant_pairwise_contrasts.csv"

region_significance_frequency_df.write_csv(region_frequency_file)
pairwise_robustness_df.write_csv(robust_pairs_file)
largest_significant_pairwise_df.write_csv(largest_pairs_file)

print(f"Holm-significant pairwise rows: {significant_pairwise_df.height}")
print(f"Robust pairs significant in all six combinations: {robust_pairwise_df.height}")
print("Region frequencies count appearances in significant pairs, not unique regions.")

print()
print("ROBUST PAIRWISE CONTRASTS BY OUTCOME")
for outcome_name in outcomes:
    robust_outcome = robust_pairwise_df.filter(pl.col("outcome") == outcome_name).sort(
        "mean_absolute_difference_percentage_points", descending=True
    )
    print(f"{outcome_name}: {robust_outcome.height} robust pairs")
    print(robust_outcome.select(
        "region_1_code", "region_2_code", "mean_difference_percentage_points",
        "minimum_absolute_difference_percentage_points", "maximum_absolute_difference_percentage_points",
    ).head(10))

print()
print("REGIONS MOST FREQUENTLY INVOLVED IN HOLM-SIGNIFICANT PAIRS")
for outcome_name in outcomes:
    print(outcome_name)
    print(region_significance_frequency_df.filter(pl.col("outcome") == outcome_name).head(10))

print()
print("MANUSCRIPT-READY SUMMARY TEXT (REGION CODES REQUIRE LABEL CROSSWALK)")
for outcome_name in outcomes:
    robust_outcome = robust_pairwise_df.filter(pl.col("outcome") == outcome_name).sort(
        "mean_absolute_difference_percentage_points", descending=True
    )
    if robust_outcome.height == 0:
        print(
            f"For {outcome_name.replace('_', ' ')}, no regional pair remained "
            "statistically significant after Holm adjustment in all six model-weight combinations."
        )
    else:
        example_text = "; ".join(
            f"a3a {int(row['region_1_code'])} versus a3a {int(row['region_2_code'])} "
            f"(mean absolute difference {row['mean_absolute_difference_percentage_points']:.2f} percentage points)"
            for row in robust_outcome.head(3).iter_rows(named=True)
        )
        print(
            f"For {outcome_name.replace('_', ' ')}, {robust_outcome.height} regional pairs "
            "remained Holm-significant in all six model-weight combinations. "
            f"The largest robust contrasts included: {example_text}."
        )


print()
print("STEP 9. SAVE OUTPUT FILES")

prediction_file = (
    output_dir
    / "24c29_regional_adjusted_predicted_probabilities.csv"
)

odds_ratio_file = (
    output_dir
    / "24c29_regional_odds_ratios.csv"
)

sample_file = (
    output_dir
    / "24c29_regional_prediction_sample_summary.csv"
)

weight_comparison_file = (
    output_dir
    / "24c29_regional_weight_sensitivity.csv"
)

ranking_file = (
    output_dir
    / "24c29_regional_risk_rankings.csv"
)

pairwise_file = (
    output_dir
    / "24c29_regional_all_pairwise_contrasts_holm.csv"
)

predictions_df.write_csv(
    prediction_file
)

odds_ratios_df.write_csv(
    odds_ratio_file
)

samples_df.write_csv(
    sample_file
)

weight_comparison_df.write_csv(
    weight_comparison_file
)

ranking_df.write_csv(
    ranking_file
)

pairwise_df.write_csv(
    pairwise_file
)

# Descriptive forest plot for the prespecified primary display:
# major_obstacle, wmedian, model1_full. Each regional prediction uses its
# own observed covariate distribution. Error bars are conditional
# coefficient-covariance delta-method intervals, not full design CIs.
forest_data = (
    predictions_df
    .filter(
        (pl.col("outcome") == "major_obstacle")
        & (pl.col("weight") == "wmedian")
        & (pl.col("model") == "model1_full")
    )
    .sort("predicted_probability")
)

forest_codes = forest_data.get_column("region_code").to_list()
forest_estimates = (
    forest_data.get_column("predicted_percentage").to_numpy()
)
forest_low = (
    forest_data.get_column("ci_low_percentage").to_numpy()
)
forest_high = (
    forest_data.get_column("ci_high_percentage").to_numpy()
)

fig, ax = plt.subplots(figsize=(9, 9))
y_positions = np.arange(len(forest_codes))
ax.errorbar(
    forest_estimates,
    y_positions,
    xerr=[forest_estimates - forest_low, forest_high - forest_estimates],
    fmt="o",
    capsize=3,
)
ax.set_yticks(y_positions)
ax.set_yticklabels([str(code) for code in forest_codes])
ax.set_xlabel("Adjusted predicted probability (%)")
ax.set_ylabel("Establishment region code (a3a)")
ax.set_title(
    "Major financing obstacle by establishment region\n"
    "wmedian; model1_full; within-region predictions; conditional 95% CIs"
)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
forest_plot_file = output_dir / "24c29_major_obstacle_regional_forest_plot.png"
fig.savefig(forest_plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)


print()
print("STEP 10. FINAL SUMMARY")

print()
print("PAIRWISE CONTRAST SUMMARY")
print(f"Total pairwise rows: {pairwise_df.height}")
print(f"Holm families: outcome x weight x model; {expected_pair_count} tests per family")
print("Confidence intervals: conditional coefficient-covariance delta method")
print("Regional predictions use each region's own observed covariate distribution.")
print("Probability CIs are stored in both 0-1 and 0-100 scales.")
print("Pairwise CIs are stored in probability and percentage-point scales.")
print("These are not full design-based intervals; covariate-distribution uncertainty is not included.")

print()
print("MODEL SAMPLE SUMMARY")
print(samples_df)

print()
print("TOP FIVE REGIONS BY PREDICTED PROBABILITY")

for outcome_name in outcomes:

    for weight_name in weights:

        for model_name in model_names:

            top_regions = (
                ranking_df
                .filter(
                    (pl.col("outcome") == outcome_name)
                    & (pl.col("weight") == weight_name)
                    & (pl.col("model") == model_name)
                )
                .head(5)
                .select(
                    "region_code",
                    "predicted_percentage",
                    "ci_low_percentage",
                    "ci_high_percentage",
                    "risk_rank",
                )
            )

            print()
            print(
                f"{outcome_name} | "
                f"{weight_name} | "
                f"{model_name}"
            )

            print(top_regions)


print()
print("OUTPUT FILES")

for output_file in [
    prediction_file,
    odds_ratio_file,
    sample_file,
    weight_comparison_file,
    ranking_file,
    pairwise_file,
    region_frequency_file,
    robust_pairs_file,
    largest_pairs_file,
    forest_plot_file,
    crosswalk_file,
]:
    print(output_file)


print()
print("=" * 78)
print("24C-29 COMPLETED")
print("=" * 78)

region_exclusion_audit = original_region_counts.with_columns(
    pl.col("a3a").cast(pl.Int64).alias("region_code")
).with_columns(
    pl.col("region_code").is_in(sparse_region_codes).alias("excluded")
).drop("a3a")

region_exclusion_audit.write_csv(
    output_dir / "24c29_sparse_region_exclusion_audit.csv"
)

print("Sparse-region exclusion audit saved.")
