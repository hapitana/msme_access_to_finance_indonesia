from pathlib import Path
import os
import re
import runpy

import numpy as np
import polars as pl
from scipy.stats import f as f_distribution


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
print("24C-21. INDEPENDENT GEOGRAPHIC JOINT WALD TEST AUDIT")
print("=" * 78)

print()
print("STEP 1. REBUILD GEOGRAPHIC MODELS")

original_cwd = Path.cwd()

try:
    os.chdir(project_dir)

    script55_namespace = runpy.run_path(
        str(script55_path)
    )

finally:
    os.chdir(original_cwd)


if "geo_results" not in script55_namespace:
    raise KeyError(
        "geo_results was not found after executing Script 55."
    )

geo_results = script55_namespace["geo_results"]

expected_model_keys = {
    f"{outcome}_{weight}"
    for outcome in [
        "major_obstacle",
        "any_obstacle",
    ]
    for weight in [
        "wmedian",
        "wweak",
    ]
}

if set(geo_results.keys()) != expected_model_keys:
    raise ValueError(
        "Unexpected geographic model keys detected."
    )

print("Geographic model objects retrieved.")
print(f"Model groups: {len(geo_results)}")


print()
print("STEP 2. INDEPENDENT WALD CALCULATIONS")

summary_rows = []
coefficient_rows = []
audit_text = []

for outcome_name in [
    "major_obstacle",
    "any_obstacle",
]:

    for weight_name in [
        "wmedian",
        "wweak",
    ]:

        model_key = f"{outcome_name}_{weight_name}"

        fitted_models = geo_results[model_key]

        for model_name, model_object in fitted_models.items():

            fitted = model_object.fitted

            feature_names = list(
                fitted.feature_names
            )

            covariance = np.asarray(
                fitted.cov_matrix,
                dtype=np.float64,
            )

            region_terms = [
                name
                for name in feature_names
                if str(name).startswith("a3a_")
            ]

            if len(region_terms) != 21:
                raise ValueError(
                    f"{model_name}: expected 21 region terms, "
                    f"found {len(region_terms)}."
                )

            if covariance.shape != (
                len(feature_names),
                len(feature_names),
            ):
                raise ValueError(
                    f"{model_name}: covariance dimensions "
                    "do not match feature names."
                )

            coefficient_lookup = {
                coefficient.term: float(
                    coefficient.est
                )
                for coefficient in fitted.coefs
            }

            missing_coefficients = [
                term
                for term in region_terms
                if term not in coefficient_lookup
            ]

            if missing_coefficients:
                raise ValueError(
                    f"{model_name}: missing coefficients: "
                    f"{missing_coefficients}"
                )

            beta = np.asarray(
                [
                    coefficient_lookup[term]
                    for term in region_terms
                ],
                dtype=np.float64,
            )

            region_indices = [
                feature_names.index(term)
                for term in region_terms
            ]

            covariance_region = covariance[
                np.ix_(
                    region_indices,
                    region_indices,
                )
            ]

            covariance_region = (
                covariance_region
                + covariance_region.T
            ) / 2.0

            if not np.isfinite(beta).all():
                raise ValueError(
                    f"{model_name}: non-finite regional coefficients."
                )

            if not np.isfinite(covariance_region).all():
                raise ValueError(
                    f"{model_name}: non-finite covariance values."
                )

            q = len(beta)

            df_den = float(
                fitted._design_df()
            )

            matrix_rank = int(
                np.linalg.matrix_rank(
                    covariance_region
                )
            )

            eigenvalues = np.linalg.eigvalsh(
                covariance_region
            )

            minimum_eigenvalue = float(
                eigenvalues.min()
            )

            maximum_eigenvalue = float(
                eigenvalues.max()
            )

            condition_number = float(
                np.linalg.cond(
                    covariance_region
                )
            )

            try:
                solved_beta = np.linalg.solve(
                    covariance_region,
                    beta,
                )

                wald_chi_square = float(
                    beta @ solved_beta
                )

                independent_f = (
                    wald_chi_square / q
                )

                independent_p = float(
                    f_distribution.sf(
                        independent_f,
                        q,
                        df_den,
                    )
                )

                calculation_status = "SUCCESS"

            except np.linalg.LinAlgError:
                wald_chi_square = float("nan")
                independent_f = float("nan")
                independent_p = float("nan")
                calculation_status = "SINGULAR COVARIANCE MATRIX"

            package_test = str(
                fitted.term_test("a3a")
            )

            package_match = re.search(
                r"F\((\d+),\s*([\d.]+)\)"
                r"\s*=\s*([\d.eE+-]+),\s*"
                r"p\s*(<|=)\s*([\d.eE+-]+)",
                package_test,
            )

            if package_match is None:
                package_df_num = float("nan")
                package_df_den = float("nan")
                package_f = float("nan")
                package_p_operator = "UNPARSED"
                package_p_threshold = float("nan")
                package_comparison = "UNPARSED"

            else:
                package_df_num = int(
                    package_match.group(1)
                )

                package_df_den = float(
                    package_match.group(2)
                )

                package_f = float(
                    package_match.group(3)
                )

                package_p_operator = (
                    package_match.group(4)
                )

                package_p_threshold = float(
                    package_match.group(5)
                )

                f_matches = (
                    abs(
                        independent_f - package_f
                    ) <= 0.0051
                )

                df_matches = (
                    q == package_df_num
                    and abs(
                        df_den - package_df_den
                    ) < 1e-8
                )

                if package_p_operator == "<":
                    p_matches = (
                        independent_p
                        < package_p_threshold
                    )
                else:
                    p_matches = (
                        abs(
                            independent_p
                            - package_p_threshold
                        ) < 1e-8
                    )

                if (
                    calculation_status == "SUCCESS"
                    and f_matches
                    and df_matches
                    and p_matches
                ):
                    package_comparison = "PASS"
                else:
                    package_comparison = "CHECK"

            covariance_full_rank = (
                matrix_rank == q
            )

            covariance_positive_definite = (
                minimum_eigenvalue > 0
            )

            summary_rows.append(
                {
                    "outcome": outcome_name,
                    "weight": weight_name,
                    "model": model_name,
                    "region_coefficient_count": q,
                    "df_numerator": q,
                    "df_denominator": df_den,
                    "wald_chi_square": wald_chi_square,
                    "independent_wald_f": independent_f,
                    "independent_p_value": independent_p,
                    "package_wald_f": package_f,
                    "package_p_operator": package_p_operator,
                    "package_p_threshold": package_p_threshold,
                    "covariance_rank": matrix_rank,
                    "covariance_full_rank": covariance_full_rank,
                    "minimum_eigenvalue": minimum_eigenvalue,
                    "maximum_eigenvalue": maximum_eigenvalue,
                    "condition_number": condition_number,
                    "positive_definite": covariance_positive_definite,
                    "calculation_status": calculation_status,
                    "package_comparison": package_comparison,
                    "package_test": package_test,
                }
            )

            for index, term in enumerate(region_terms):
                coefficient_rows.append(
                    {
                        "outcome": outcome_name,
                        "weight": weight_name,
                        "model": model_name,
                        "term": term,
                        "estimate": float(beta[index]),
                        "covariance_diagonal": float(
                            covariance_region[index, index]
                        ),
                    }
                )

            audit_text.extend(
                [
                    "",
                    "=" * 78,
                    f"MODEL: {model_name}",
                    f"OUTCOME: {outcome_name}",
                    f"WEIGHT: {weight_name}",
                    "=" * 78,
                    f"Region coefficients: {q}",
                    f"Numerator df: {q}",
                    f"Denominator df: {df_den}",
                    f"Wald chi-square: {wald_chi_square:.10f}",
                    f"Independent Wald F: {independent_f:.10f}",
                    f"Independent p-value: {independent_p:.12g}",
                    f"Package result: {package_test}",
                    f"Covariance rank: {matrix_rank}",
                    f"Minimum eigenvalue: {minimum_eigenvalue:.12g}",
                    f"Maximum eigenvalue: {maximum_eigenvalue:.12g}",
                    f"Condition number: {condition_number:.12g}",
                    f"Calculation status: {calculation_status}",
                    f"Package comparison: {package_comparison}",
                ]
            )

            print()
            print(f"MODEL: {model_name}")
            print(
                f"Independent F({q}, {df_den:.0f}): "
                f"{independent_f:.6f}"
            )
            print(
                f"Independent p-value: "
                f"{independent_p:.12g}"
            )
            print(f"Package result: {package_test}")
            print(
                f"Covariance rank: {matrix_rank}/{q}"
            )
            print(
                f"Condition number: "
                f"{condition_number:.6g}"
            )
            print(
                f"Comparison: {package_comparison}"
            )


print()
print("STEP 3. SAVE AUDIT RESULTS")

if len(summary_rows) != 12:
    raise ValueError(
        f"Expected 12 model results, found {len(summary_rows)}."
    )

summary_df = pl.DataFrame(
    summary_rows
).sort(
    [
        "outcome",
        "weight",
        "model",
    ]
)

coefficient_df = pl.DataFrame(
    coefficient_rows
).sort(
    [
        "outcome",
        "weight",
        "model",
        "term",
    ]
)

summary_file = (
    output_dir
    / "24c21_geographic_joint_wald_audit.csv"
)

coefficient_file = (
    output_dir
    / "24c21_geographic_joint_wald_coefficients.csv"
)

text_file = (
    output_dir
    / "24c21_geographic_joint_wald_audit.txt"
)

summary_df.write_csv(
    summary_file
)

coefficient_df.write_csv(
    coefficient_file
)

audit_text.insert(
    0,
    "24C-21. INDEPENDENT GEOGRAPHIC JOINT WALD TEST AUDIT",
)

audit_text.extend(
    [
        "",
        "=" * 78,
        "FINAL VALIDATION",
        "=" * 78,
        f"Models audited: {summary_df.height}",
        f"Regional coefficients audited: {coefficient_df.height}",
        (
            "Successful calculations: "
            f"{summary_df.filter(pl.col('calculation_status') == 'SUCCESS').height}"
        ),
        (
            "Package comparisons passed: "
            f"{summary_df.filter(pl.col('package_comparison') == 'PASS').height}"
        ),
        (
            "Package comparisons requiring checks: "
            f"{summary_df.filter(pl.col('package_comparison') != 'PASS').height}"
        ),
        "",
        "Interpretation:",
        "The independent Wald calculation tests the joint null hypothesis",
        "that all 21 establishment-region coefficients are zero.",
        "Statistical significance indicates regional association, not causation.",
    ]
)

text_file.write_text(
    "\n".join(audit_text),
    encoding="utf-8",
)

print()
print("FINAL AUDIT SUMMARY")
print(summary_df)

print()
print("OUTPUT FILES")
print(summary_file)
print(coefficient_file)
print(text_file)

print()
print("24C-21 COMPLETED")