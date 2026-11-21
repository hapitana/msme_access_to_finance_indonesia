from pathlib import Path
import polars as pl

import svy
import numpy as np
import re



project_dir = Path(__file__).resolve().parent.parent

input_file = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023_candidate_analytic.parquet"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "script55_survey_weighted_logistic_regression"
)

output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("SURVEY-WEIGHTED LOGISTIC REGRESSION")
print("=" * 72)


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def save_text(filename, content):
    output_file = output_dir / filename
    output_file.write_text(content, encoding="utf-8")
    print(f"Saved: {output_file}")


def inspect_glm_object(model, model_name):
    print_section(f"OBJECT INSPECTION: {model_name}")

    inspection = []

    inspection.append(f"Model name: {model_name}")
    inspection.append(f"Model type: {type(model)}")
    inspection.append("")
    inspection.append("Model public attributes:")
    inspection.append(", ".join(
        name for name in dir(model)
        if not name.startswith("_")
    ))

    print(f"Model type: {type(model)}")
    print("Model public attributes:")
    print([
        name for name in dir(model)
        if not name.startswith("_")
    ])

    for attribute_name in ["fitted", "coefs", "stats"]:
        inspection.append("")
        inspection.append(f"Attribute: {attribute_name}")

        try:
            attribute_value = getattr(model, attribute_name)

            inspection.append(
                f"Type: {type(attribute_value)}"
            )
            inspection.append(
                f"Value: {repr(attribute_value)}"
            )

            print()
            print(f"{attribute_name} type:")
            print(type(attribute_value))

            print(f"{attribute_name} value:")
            print(repr(attribute_value))

            inspection.append("Public attributes:")
            inspection.append(", ".join(
                name for name in dir(attribute_value)
                if not name.startswith("_")
            ))

            print(f"{attribute_name} public attributes:")
            print([
                name for name in dir(attribute_value)
                if not name.startswith("_")
            ])

        except Exception as error:
            inspection.append(
                f"Inspection error: {type(error).__name__}: {error}"
            )
            print(
                f"Could not inspect {attribute_name}: "
                f"{type(error).__name__}: {error}"
            )

    save_text(
        f"{model_name.lower()}_object_inspection.txt",
        "\n".join(inspection)
    )


def save_model_results(model, model_name):
    print_section(f"SAVING RESULTS: {model_name}")

    coefficients = model.to_polars()

    coefficient_file = (
        output_dir
        / f"{model_name.lower()}_coefficients.csv"
    )

    coefficients.write_csv(coefficient_file)

    print(f"Saved coefficients: {coefficient_file}")
    print(coefficients)

    expected_columns = [
        "term",
        "estimate",
        "std_err",
        "conf_low",
        "conf_high",
        "statistic",
        "p_value",
        "df",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in coefficients.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unexpected coefficient output. "
            f"Missing columns: {missing_columns}. "
            f"Available columns: {coefficients.columns}"
        )

    odds_ratios = coefficients.with_columns(
        [
            pl.col("estimate").exp().alias("odds_ratio"),
            pl.col("conf_low").exp().alias("or_conf_low"),
            pl.col("conf_high").exp().alias("or_conf_high"),
        ]
    )

    odds_ratio_file = (
        output_dir
        / f"{model_name.lower()}_odds_ratios.csv"
    )

    odds_ratios.write_csv(odds_ratio_file)

    print(f"Saved odds ratios: {odds_ratio_file}")
    print(odds_ratios)

    return coefficients, odds_ratios


def save_model_statistics(model, model_name):
    print_section(f"MODEL STATISTICS: {model_name}")

    statistics = model.fitted.stats

    print(statistics)

    statistics_file = (
        output_dir
        / f"{model_name.lower()}_statistics.txt"
    )

    save_text(
        f"{model_name.lower()}_statistics.txt",
        repr(statistics)
    )

    return statistics


def validate_required_columns(data, required_columns):
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


print_section("1. LOAD ANALYTIC DATA")

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}"
    )

data = pl.read_parquet(input_file)

print(f"Input file: {input_file}")
print(f"Rows: {data.height:,}")
print(f"Columns: {data.width:,}")


print_section("2. PREPARE BANK ACCOUNT VARIABLE")

if "bank_account" not in data.columns:

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

    print(f"Loading base dataset: {base_file}")

    base_data = pl.read_parquet(base_file)

    validate_required_columns(
        base_data,
        ["idstd", "k6"],
    )

    k6_lookup = base_data.select(
        ["idstd", "k6"]
    )

    if k6_lookup["idstd"].n_unique() != k6_lookup.height:
        raise ValueError(
            "Duplicate idstd values found in base dataset."
        )

    data = data.join(
        k6_lookup,
        on="idstd",
        how="left",
        validate="m:1",
    )

    if data["k6"].null_count() > 0:
        raise ValueError(
            "Some candidate observations could not be matched "
            "with k6 from the base dataset."
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

else:
    print("Using existing bank_account variable.")

print("Bank account frequency:")
print(
    data.group_by("bank_account")
    .len()
    .sort("bank_account")
)

print(f"Rows after join: {data.height:,}")

print(
    "Mapping assumption: k6 = 1 means Yes; "
    "k6 = 2 means No. Verify against the WBES codebook."
)


print_section("3. VALIDATE SURVEY DESIGN VARIABLES")

design_columns = [
    "idstd",
    "strata",
    "wmedian",
]

validate_required_columns(data, design_columns)

design_missing = data.select(
    [
        pl.col("idstd").is_null().sum().alias("missing_idstd"),
        pl.col("strata").is_null().sum().alias("missing_strata"),
        pl.col("wmedian").is_null().sum().alias("missing_weight"),
        (pl.col("wmedian") <= 0).sum().alias("nonpositive_weight"),
    ]
)

print(design_missing)

if design_missing.row(0)[0] > 0:
    raise ValueError("Missing PSU identifiers detected.")

if design_missing.row(0)[1] > 0:
    raise ValueError("Missing strata identifiers detected.")

if design_missing.row(0)[2] > 0:
    raise ValueError("Missing survey weights detected.")

if design_missing.row(0)[3] > 0:
    raise ValueError("Nonpositive survey weights detected.")

duplicate_psu = (
    data.group_by("idstd")
    .len()
    .filter(pl.col("len") > 1)
)

if duplicate_psu.height > 0:
    raise ValueError(
        "PSU identifiers are not unique. "
        "Check idstd before fitting the survey models."
    )

print("PSU identifiers are unique.")
print("Survey design variables passed validation.")


print_section("4. DEFINE MODEL VARIABLES")

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

required_model_columns = list(
    dict.fromkeys(
        model2_variables
        + design_columns
    )
)

validate_required_columns(
    data,
    required_model_columns
)

print("Model 1 variables:")
print(model1_variables)

print()
print("Model 2 variables:")
print(model2_variables)


print_section("5. CREATE COMPLETE-CASE FLAGS")

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
            "model1_complete"
        ),
        model2_complete_expression.alias(
            "model2_complete"
        ),
    ]
)

model1_full_n = data.filter(
    pl.col("model1_complete")
).height

model1_common_n = data.filter(
    pl.col("model2_complete")
).height

model2_common_n = model1_common_n

print(f"Model 1 full sample: {model1_full_n:,}")
print(f"Model 1 common sample: {model1_common_n:,}")
print(f"Model 2 common sample: {model2_common_n:,}")

print()
print("Complete-case sample summary:")

sample_summary = pl.DataFrame(
    {
        "model": [
            "Model 1 full",
            "Model 1 common",
            "Model 2 common",
        ],
        "observations": [
            model1_full_n,
            model1_common_n,
            model2_common_n,
        ],
    }
)

print(sample_summary)

sample_summary.write_csv(
    output_dir / "sample_size_summary.csv"
)


print_section("6. DEFINE SURVEY DESIGN")

design = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wmedian",
    singleton="scale",
)

sample = svy.Sample(
    data=data,
    design=design,
)

print("Survey design created.")
print(f"Total observations: {data.height:,}")
print("Stratum variable: strata")
print("PSU variable: idstd")
print("Weight variable: wmedian")
print("Singleton adjustment: scale")
print("Variance estimation: package default")


print_section("7. DEFINE MODEL PREDICTORS")

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

print("Model 1 predictors:")
print(model1_predictors)

print()
print("Model 2 predictors:")
print(model2_predictors)





print_section("8. FIT MODEL 1: FULL SAMPLE")

model1_full = sample.glm.fit(
    y="major_obstacle",
    x=model1_predictors,
    family="binomial",
    link="logit",
    where=(
        svy.col("model1_complete") == True
    ),
)

print("Model 1 full sample estimation completed.")

save_model_results(
    model1_full,
    "model1_full",
)

save_model_statistics(
    model1_full,
    "model1_full",
)


print_section("9. FIT MODEL 1: COMMON SAMPLE")

model1_common = sample.glm.fit(
    y="major_obstacle",
    x=model1_predictors,
    family="binomial",
    link="logit",
    where=(
        svy.col("model2_complete") == True
    ),
)

print("Model 1 common sample estimation completed.")

save_model_results(
    model1_common,
    "model1_common",
)

save_model_statistics(
    model1_common,
    "model1_common",
)


print_section("10. FIT MODEL 2: COMMON SAMPLE")

model2_common = sample.glm.fit(
    y="major_obstacle",
    x=model2_predictors,
    family="binomial",
    link="logit",
    where=(
        svy.col("model2_complete") == True
    ),
)

print("Model 2 common sample estimation completed.")

save_model_results(
    model2_common,
    "model2_common",
)

save_model_statistics(
    model2_common,
    "model2_common",
)


def save_compact_inspection(model, model_name):
    lines = []

    lines.append(f"MODEL: {model_name}")
    lines.append(f"MODEL TYPE: {type(model)}")

    for attribute_name in ["fitted", "coefs", "stats"]:
        lines.append("")
        lines.append(f"ATTRIBUTE: {attribute_name}")

        try:
            value = getattr(model, attribute_name)

            lines.append(f"TYPE: {type(value)}")
            lines.append(
                "PUBLIC ATTRIBUTES: "
                + ", ".join(
                    name for name in dir(value)
                    if not name.startswith("_")
                )
            )

            if attribute_name == "coefs":
                lines.append(f"VALUE: {repr(value)[:1500]}")

        except Exception as error:
            lines.append(
                f"ERROR: {type(error)._name_}: {error}"
            )

    save_text(
        f"{model_name.lower()}_compact_inspection.txt",
        "\n".join(lines),
    )

    print(f"Compact inspection completed: {model_name}")


print_section("12. SAVE MODEL COMPARISON SUMMARY")

model_comparison = pl.DataFrame(
    {
        "model": [
            "Model 1 full",
            "Model 1 common",
            "Model 2 common",
        ],
        "sample_size": [
            model1_full_n,
            model1_common_n,
            model2_common_n,
        ],
        "predictor_set": [
            "Baseline predictors",
            "Baseline predictors",
            "Baseline + bank account",
        ],
    }
)

comparison_file = (
    output_dir
    / "model_comparison_summary.csv"
)

model_comparison.write_csv(comparison_file)

print(model_comparison)
print(f"Saved: {comparison_file}")


print_section("13. FINAL OUTPUT")

print(f"Output directory: {output_dir}")

print()
print("Generated files include:")
for output_file in sorted(output_dir.iterdir()):
    if output_file.is_file():
        print(f"- {output_file.name}")

print_section("14. COVARIANCE MATRIX INSPECTION")

for model_name, model in [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]:

    print()
    print(f"MODEL: {model_name}")

    fitted = model.fitted
    covariance = fitted.cov_matrix

    print(f"Fitted type: {type(fitted)}")
    print(f"Covariance type: {type(covariance)}")

    print("Covariance representation:")
    print(repr(covariance)[:1500])

    print("Coefficient count:")
    print(len(fitted.coefs))

    print("Covariance public attributes:")
    print([
        name for name in dir(covariance)
        if not name.startswith("_")
    ])

print_section("15. COVARIANCE MATRIX NUMERICAL VALIDATION")

validation_results = []

for model_name, model in [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]:
    print()
    print(f"MODEL: {model_name}")

    fitted = model.fitted
    covariance = fitted.cov_matrix
    coefficients = fitted.coefs

    coefficient_count = len(coefficients)

    standard_errors = np.array(
        [float(coef.se) for coef in coefficients],
        dtype=float,
    )

    covariance_diagonal = np.diag(covariance)
    expected_diagonal = standard_errors ** 2

    symmetry_error = float(
        np.max(np.abs(covariance - covariance.T))
    )

    diagonal_error = float(
        np.max(np.abs(covariance_diagonal - expected_diagonal))
    )

    finite_values = bool(np.isfinite(covariance).all())

    eigenvalues = np.linalg.eigvalsh(
        (covariance + covariance.T) / 2
    )

    minimum_eigenvalue = float(np.min(eigenvalues))
    maximum_eigenvalue = float(np.max(eigenvalues))

    numerical_tolerance = (
        1e-10 * max(1.0, maximum_eigenvalue)
    )

    positive_semidefinite = (
        minimum_eigenvalue >= -numerical_tolerance
    )

    print(f"Covariance shape: {covariance.shape}")
    print(f"Coefficient count: {coefficient_count}")
    print(f"All covariance values finite: {finite_values}")
    print(f"Maximum symmetry error: {symmetry_error:.12e}")
    print(f"Maximum diagonal vs SE squared error: {diagonal_error:.12e}")
    print(f"Minimum eigenvalue: {minimum_eigenvalue:.12e}")
    print(f"Maximum eigenvalue: {maximum_eigenvalue:.12e}")
    print(f"Positive semidefinite: {positive_semidefinite}")

    print("Coefficient order:")
    print([coef.term for coef in coefficients])

    validation_results.append({
        "model": model_name,
        "coefficient_count": coefficient_count,
        "finite_values": finite_values,
        "symmetry_error": symmetry_error,
        "diagonal_error": diagonal_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "positive_semidefinite": positive_semidefinite,
    })

validation_table = pl.DataFrame(validation_results)

validation_table.write_csv(
    output_dir / "covariance_matrix_validation.csv"
)

print()
print("COVARIANCE VALIDATION SUMMARY")
print(validation_table)

print_section("16. COEFFICIENT AND INFERENCE VALIDATION")

inference_results = []

for model_name, model in [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]:
    print()
    print(f"MODEL: {model_name}")

    fitted = model.fitted

    for coef in fitted.coefs:
        estimate = float(coef.est)
        standard_error = float(coef.se)
        lower_ci = float(coef.lci)
        upper_ci = float(coef.uci)

        wald_value = float(coef.wald.value)
        p_value = float(coef.wald.p_value)
        degrees_freedom = float(coef.wald.df)

        expected_wald = estimate / standard_error

        odds_ratio = float(np.exp(estimate))
        odds_ratio_lower = float(np.exp(lower_ci))
        odds_ratio_upper = float(np.exp(upper_ci))

        inference_results.append({
            "model": model_name,
            "term": coef.term,
            "estimate": estimate,
            "standard_error": standard_error,
            "wald_statistic": wald_value,
            "expected_wald_statistic": expected_wald,
            "p_value": p_value,
            "degrees_freedom": degrees_freedom,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "odds_ratio": odds_ratio,
            "odds_ratio_lower": odds_ratio_lower,
            "odds_ratio_upper": odds_ratio_upper,
            "wald_consistent": np.isclose(
                wald_value,
                expected_wald,
                rtol=1e-8,
                atol=1e-10,
            ),
            "ci_contains_estimate": (
                lower_ci <= estimate <= upper_ci
            ),
        })

inference_table = pl.DataFrame(inference_results)

inference_table.write_csv(
    output_dir / "coefficient_inference_validation.csv"
)

print("INFERENCE VALIDATION SUMMARY")

print(
    inference_table.group_by("model").agg([
        pl.len().alias("coefficient_count"),
        pl.col("wald_consistent").all().alias("all_wald_consistent"),
        pl.col("ci_contains_estimate").all().alias(
            "all_ci_contain_estimate"
        ),
        pl.col("p_value").is_finite().all().alias(
            "all_p_values_finite"
        ),
    ])
)

print()
print("SELECTED COEFFICIENT RESULTS")

print(
    inference_table.filter(
        pl.col("term").is_in([
            "firm_age",
            "log_employees",
            "female_owner",
            "bank_account",
        ])
    )
)

print_section("17. MODEL STATISTICS AND INFERENCE OBJECT INSPECTION")

for model_name, model in [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]:
    print()
    print(f"MODEL: {model_name}")

    fitted = model.fitted

    print("Model statistics:")
    print(repr(fitted.stats))

    print()
    print("Model statistics type:")
    print(type(fitted.stats))

    print()
    print("Model statistics public attributes:")
    print([
        name
        for name in dir(fitted.stats)
        if not name.startswith("_")
    ])

    print()
    print("Model-level public attributes:")
    print([
        name
        for name in dir(model)
        if not name.startswith("_")
    ])

    print()
    print("Fitted-level public attributes:")
    print([
        name
        for name in dir(fitted)
        if not name.startswith("_")
    ])

    print()
    print("First coefficient Wald object:")
    print(repr(fitted.coefs[0].wald))

    print()
    print("First coefficient Wald object attributes:")
    print([
        name
        for name in dir(fitted.coefs[0].wald)
        if not name.startswith("_")
    ])

print_section("18. DESIGN DEGREES OF FREEDOM INSPECTION")

for model_name, model in [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]:
    fitted = model.fitted
    stats = fitted.stats

    denominator_df = stats.wald_adj.df_den
    numerator_df = stats.wald_adj.df_num

    print()
    print(f"MODEL: {model_name}")
    print(f"Estimation sample size: {stats.n}")
    print(f"Denominator degrees of freedom: {denominator_df}")
    print(f"Numerator degrees of freedom: {numerator_df}")

    implied_strata = stats.n - denominator_df

    print(f"Implied number of strata: {implied_strata}")
    print(f"Coefficient count: {len(fitted.coefs)}")
    print(f"Non-intercept coefficient count: {len(fitted.coefs) - 1}")

    print()
    print("Model estimation condition:")
    print(repr(fitted.where_clause))

    print()
    print("Term information:")
    print(repr(fitted.term_info))

print()
print("SURVEY DESIGN OBJECT")

print(f"Design type: {type(design)}")

print("Design public attributes:")
print([
    name for name in dir(design)
    if not name.startswith("_")
])

print("Design representation:")
print(repr(design))

print_section("19. ACTUAL PSU AND STRATA COUNT BY ESTIMATION SAMPLE")

sample_conditions = [
    ("model1_full", "model1_complete", model1_full),
    ("model1_common", "model2_complete", model1_common),
    ("model2_common", "model2_complete", model2_common),
]

design_audit_results = []

for model_name, condition_name, model in sample_conditions:
    print()
    print(f"MODEL: {model_name}")

    sample_data = data.filter(
        pl.col(condition_name) == True
    )

    row_count = sample_data.height

    unique_psu = sample_data["idstd"].n_unique()
    unique_strata = sample_data["strata"].n_unique()

    missing_psu = sample_data["idstd"].null_count()
    missing_strata = sample_data["strata"].null_count()

    psu_strata_table = (
        sample_data
        .group_by("strata")
        .agg(
            pl.col("idstd").n_unique().alias("psu_count"),
            pl.len().alias("observation_count"),
        )
    )

    singleton_strata_count = (
        psu_strata_table
        .filter(pl.col("psu_count") == 1)
        .height
    )

    calculated_design_df = unique_psu - unique_strata

    model_df = model.fitted.stats.wald_adj.df_den

    print(f"Estimation condition: {condition_name}")
    print(f"Observations: {row_count}")
    print(f"Unique PSU: {unique_psu}")
    print(f"Unique strata: {unique_strata}")
    print(f"Singleton strata: {singleton_strata_count}")
    print(f"Missing PSU: {missing_psu}")
    print(f"Missing strata: {missing_strata}")
    print(f"Calculated PSU minus strata: {calculated_design_df}")
    print(f"Model denominator df: {model_df}")
    print(f"Difference: {calculated_design_df - model_df}")

    print()
    print("PSU uniqueness check:")

    duplicate_psu_count = (
        sample_data
        .group_by("idstd")
        .agg(
            pl.col("strata").n_unique().alias("strata_count")
        )
        .filter(pl.col("strata_count") > 1)
        .height
    )

    print(f"PSUs appearing in multiple strata: {duplicate_psu_count}")

    design_audit_results.append({
        "model": model_name,
        "observations": row_count,
        "unique_psu": unique_psu,
        "unique_strata": unique_strata,
        "singleton_strata": singleton_strata_count,
        "calculated_design_df": calculated_design_df,
        "model_denominator_df": model_df,
        "difference": calculated_design_df - model_df,
        "duplicate_psu_across_strata": duplicate_psu_count,
    })

design_audit_table = pl.DataFrame(design_audit_results)

design_audit_table.write_csv(
    output_dir / "design_degrees_of_freedom_audit.csv"
)

print()
print("DESIGN DEGREES OF FREEDOM AUDIT SUMMARY")
print(design_audit_table)

print("=" * 72)
print("20. SURVEY PACKAGE SOURCE AUDIT")
print("=" * 72)

module_path = Path(svy.__file__).resolve()

if module_path.name == "_init_.py":
    package_root = module_path.parent
elif module_path.is_file():
    package_root = module_path.parent
else:
    package_root = module_path

search_pattern = re.compile(
    r"df_den|df_num|denominator|df_resid|"
    r"n_psu|n_strata|n_clusters|"
    r"degrees.of.freedom|singleton",
    re.IGNORECASE,
)

source_files = sorted(package_root.rglob("*.py"))

print(f"Package location: {package_root}")
print(f"Python source files found: {len(source_files)}")

match_count = 0
max_matches = 60

for source_file in source_files:
    try:
        lines = source_file.read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeDecodeError):
        continue

    for line_number, line in enumerate(lines):
        if search_pattern.search(line):
            match_count += 1

            if match_count > max_matches:
                break

            start = max(0, line_number - 2)
            end = min(len(lines), line_number + 3)

            print()
            print(
                f"FILE: {source_file.relative_to(package_root)}"
            )
            print(f"MATCH LINE: {line_number + 1}")

            for index in range(start, end):
                marker = ">" if index == line_number else " "
                print(
                    f"{marker} {index + 1:5d}: {lines[index]}"
                )

    if match_count > max_matches:
        break

print()
print(f"Total matching lines displayed: {min(match_count, max_matches)}")

if match_count > max_matches:
    print("Additional matches were omitted.")

print("=" * 72)
print("21. TARGETED GLM DEGREES OF FREEDOM SOURCE AUDIT")
print("=" * 72)

package_root = Path(svy.__file__).resolve().parent

source_files = sorted(package_root.rglob("*.py"))

path_keywords = (
    "glm",
    "logit",
    "regression",
    "model",
    "linear",
    "fit",
)

relevant_files = [
    path
    for path in source_files
    if any(
        keyword in str(path.relative_to(package_root)).lower()
        for keyword in path_keywords
    )
]

print()
print("RELEVANT SOURCE FILES")

for path in relevant_files:
    print(path.relative_to(package_root))

search_pattern = re.compile(
    r"df_den|df2|df_resid|"
    r"n_psus|n_strata|"
    r"denominator|degrees.of.freedom|"
    r"wald_adj|wald_adj|"
    r"FDist|FDist\(",
    re.IGNORECASE,
)

print()
print("MATCHING SOURCE CODE")

match_count = 0
max_matches = 80

for source_file in relevant_files:
    try:
        lines = source_file.read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeDecodeError):
        continue

    matching_lines = [
        index
        for index, line in enumerate(lines)
        if search_pattern.search(line)
    ]

    if not matching_lines:
        continue

    print()
    print(f"FILE: {source_file.relative_to(package_root)}")

    for line_number in matching_lines:
        match_count += 1

        if match_count > max_matches:
            break

        start = max(0, line_number - 4)
        end = min(len(lines), line_number + 5)

        print()
        print(f"MATCH LINE: {line_number + 1}")

        for index in range(start, end):
            marker = ">" if index == line_number else " "
            print(
                f"{marker} {index + 1:5d}: {lines[index]}"
            )

    if match_count > max_matches:
        break

print()
print(f"Total matching lines displayed: {min(match_count, max_matches)}")

if match_count > max_matches:
    print("Additional matches were omitted.")

import polars as pl
from scipy.stats import f


print("=" * 72)
print("22. ADJUSTED WALD F AND P-VALUE VERIFICATION")
print("=" * 72)

models = [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]

results = []

for model_name, model in models:
    stats = model.fitted.stats

    ordinary = stats.wald
    adjusted = stats.wald_adj

    df_num = float(adjusted.df_num)
    df_reported = float(adjusted.df_den)

    df_adjusted = df_reported - df_num + 1

    f_adjusted = float(adjusted.value)
    p_reported = float(adjusted.p_value)

    p_using_reported_df = float(
        f.sf(
            f_adjusted,
            df_num,
            df_reported,
        )
    )

    p_using_adjusted_df = float(
        f.sf(
            f_adjusted,
            df_num,
            df_adjusted,
        )
    )

    results.append({
        "model": model_name,
        "f_adjusted": f_adjusted,
        "df_num": df_num,
        "df_reported": df_reported,
        "df_adjusted_formula": df_adjusted,
        "p_reported": p_reported,
        "p_recalculated_reported_df": p_using_reported_df,
        "p_recalculated_adjusted_df": p_using_adjusted_df,
        "difference_adjusted": abs(
            p_reported - p_using_adjusted_df
        ),
        "difference_reported": abs(
            p_reported - p_using_reported_df
        ),
    })

    print()
    print(f"MODEL: {model_name}")
    print(f"Ordinary Wald F: {ordinary.value}")
    print(f"Adjusted Wald F: {f_adjusted}")
    print(f"Numerator df: {df_num}")
    print(f"Reported denominator df: {df_reported}")
    print(f"Adjusted denominator df formula: {df_adjusted}")
    print(f"Reported adjusted p-value: {p_reported}")
    print(f"P-value using reported denominator df: {p_using_reported_df}")
    print(f"P-value using adjusted denominator df: {p_using_adjusted_df}")

verification_table = pl.DataFrame(results)

print()
print("ADJUSTED WALD VERIFICATION SUMMARY")
print(verification_table)

print("=" * 72)
print("23A. GLMFIT OBJECT STRUCTURE INSPECTION")
print("=" * 72)

models = [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]

for model_name, model in models:
    fitted = model.fitted

    print()
    print(f"MODEL: {model_name}")
    print(f"Fitted object type: {type(fitted)}")

    print()
    print("Available fitted attributes:")

    for attribute in dir(fitted):
        if not attribute.startswith("_"):
            try:
                value = getattr(fitted, attribute)
                print(
                    f"{attribute}: "
                    f"{type(value).__name__}"
                )
            except Exception as error:
                print(
                    f"{attribute}: "
                    f"Unable to inspect ({type(error).__name__})"
                )

    print()
    print("Fitted object representation:")
    print(fitted)

    print()
    print("Available stats attributes:")

    stats = fitted.stats

    for attribute in dir(stats):
        if not attribute.startswith("_"):
            try:
                value = getattr(stats, attribute)
                print(
                    f"{attribute}: "
                    f"{type(value).__name__}"
                )
            except Exception as error:
                print(
                    f"{attribute}: "
                    f"Unable to inspect ({type(error).__name__})"
                )

print()
print("23A INSPECTION COMPLETED")

print()
print("=" * 72)
print("23B. COEFFICIENT OBJECT INSPECTION")
print("=" * 72)

models = [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]

for model_name, model in models:
    fitted = model.fitted

    print()
    print(f"MODEL: {model_name}")
    print(f"Number of coefficients: {len(fitted.coefs)}")
    print(f"Feature names: {fitted.feature_names}")

    if fitted.coefs:
        coefficient = fitted.coefs[0]

        print()
        print("First coefficient object:")
        print(f"Type: {type(coefficient)}")
        print(repr(coefficient))

        print()
        print("Available coefficient attributes:")

        for attribute in dir(coefficient):
            if not attribute.startswith("_"):
                try:
                    value = getattr(coefficient, attribute)
                    print(
                        f"{attribute}: "
                        f"{type(value).__name__} = {value}"
                    )
                except Exception as error:
                    print(
                        f"{attribute}: "
                        f"Unable to inspect ({type(error).__name__})"
                    )

    print()
    print("POLARS OUTPUT:")

    coefficient_table = fitted.to_polars()
    print(coefficient_table)
    print("Schema:", coefficient_table.schema)

print()
print("23B INSPECTION COMPLETED")
print()

import math

import polars as pl
from scipy.stats import t


print("=" * 72)
print("23C. INDEPENDENT COEFFICIENT INFERENCE VERIFICATION")
print("=" * 72)

models = [
    ("model1_full", model1_full),
    ("model1_common", model1_common),
    ("model2_common", model2_common),
]

results = []

for model_name, model in models:
    fitted = model.fitted
    alpha = float(fitted.alpha)

    coefficient_table = fitted.to_polars()

    df_lookup = {
        row["term"]: row["df"]
        for row in coefficient_table.iter_rows(named=True)
    }

    for coefficient in fitted.coefs:
        term = coefficient.term
        beta = float(coefficient.est)
        se = float(coefficient.se)

        reported_t = float(coefficient.wald.value)
        reported_p = float(coefficient.wald.p_value)

        reported_ci_low = float(coefficient.lci)
        reported_ci_high = float(coefficient.uci)

        df_den = int(coefficient.wald.df)

        if se <= 0 or not math.isfinite(se):
            raise ValueError(
                f"Invalid standard error: {model_name}, {term}"
            )

        t_recalculated = beta / se

        p_recalculated = float(
            2 * t.sf(abs(t_recalculated), df_den)
        )

        critical_value = float(
            t.ppf(1 - alpha / 2, df_den)
        )

        ci_low = beta - critical_value * se
        ci_high = beta + critical_value * se

        or_value = math.exp(beta)
        or_ci_low = math.exp(ci_low)
        or_ci_high = math.exp(ci_high)

        results.append({
            "model": model_name,
            "term": term,
            "beta": beta,
            "se": se,
            "df": df_den,
            "reported_t": reported_t,
            "recalculated_t": t_recalculated,
            "reported_p": reported_p,
            "recalculated_p": p_recalculated,
            "reported_ci_low": reported_ci_low,
            "recalculated_ci_low": ci_low,
            "reported_ci_high": reported_ci_high,
            "recalculated_ci_high": ci_high,
            "odds_ratio": or_value,
            "or_ci_low": or_ci_low,
            "or_ci_high": or_ci_high,
            "t_difference": abs(
                reported_t - t_recalculated
            ),
            "p_difference": abs(
                reported_p - p_recalculated
            ),
            "ci_low_difference": abs(
                reported_ci_low - ci_low
            ),
            "ci_high_difference": abs(
                reported_ci_high - ci_high
            ),
            "df_matches_table": (
                df_den == df_lookup[term]
            ),
        })

verification = pl.DataFrame(results)

print()
print("VERIFICATION SUMMARY")

summary = verification.group_by("model").agg(
    pl.len().alias("coefficient_count"),
    pl.col("t_difference").max().alias("max_t_difference"),
    pl.col("p_difference").max().alias("max_p_difference"),
    pl.col("ci_low_difference").max().alias(
        "max_ci_low_difference"
    ),
    pl.col("ci_high_difference").max().alias(
        "max_ci_high_difference"
    ),
    pl.col("df_matches_table").all().alias(
        "all_df_match_table"
    ),
)

print(summary.sort("model"))

print()
print("OVERALL VERIFICATION")

print(
    verification.select(
        pl.len().alias("total_coefficients"),
        (pl.col("t_difference") < 1e-10)
        .sum()
        .alias("t_statistics_matching"),
        (pl.col("p_difference") < 1e-10)
        .sum()
        .alias("p_values_matching"),
        (pl.col("ci_low_difference") < 1e-8)
        .sum()
        .alias("lower_ci_matching"),
        (pl.col("ci_high_difference") < 1e-8)
        .sum()
        .alias("upper_ci_matching"),
        pl.col("df_matches_table").sum().alias(
            "df_values_matching"
        ),
    )
)

print()
print("PRIMARY PREDICTORS")

primary_predictors = verification.filter(
    pl.col("term").is_in([
        "firm_age",
        "log_employees",
        "female_owner",
        "bank_account",
    ])
)

print(
    primary_predictors.select(
        "model",
        "term",
        "beta",
        "se",
        "df",
        "recalculated_t",
        "recalculated_p",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    ).sort(["model", "term"])
)

print()
print("23C VERIFICATION COMPLETED")
print()

from pathlib import Path

import polars as pl


print("=" * 72)
print("24A. SURVEY WEIGHT AND SAMPLE EQUIVALENCE AUDIT")
print("=" * 72)

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

candidate_path = (
    processed_dir
    / "indonesia_2023_candidate_analytic.parquet"
)

source_path = processed_dir / "indonesia_2023.parquet"

candidate = pl.read_parquet(candidate_path)

source = (
    pl.read_parquet(source_path)
    .select(["idstd", "k6"])
)

data = candidate.join(
    source,
    on="idstd",
    how="left",
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

required_model1 = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
]

required_model2 = required_model1 + ["bank_account"]

data = data.with_columns(
    pl.all_horizontal([
        pl.col(column).is_not_null()
        for column in required_model1
    ]).alias("model1_complete"),

    pl.all_horizontal([
        pl.col(column).is_not_null()
        for column in required_model2
    ]).alias("model2_complete"),

    (
        pl.col("wmedian").is_not_null()
        & pl.col("wmedian").is_finite()
        & (pl.col("wmedian") > 0)
    ).alias("valid_wmedian"),

    (
        pl.col("wweak").is_not_null()
        & pl.col("wweak").is_finite()
        & (pl.col("wweak") > 0)
    ).alias("valid_wweak"),
)

print()
print("DATASET DIMENSIONS")
print(data.shape)

print()
print("WEIGHT QUALITY")

weight_quality = data.select(
    pl.len().alias("total_rows"),

    pl.col("valid_wmedian")
    .sum()
    .alias("valid_wmedian_rows"),

    pl.col("valid_wweak")
    .sum()
    .alias("valid_wweak_rows"),

    (~pl.col("valid_wmedian"))
    .sum()
    .alias("invalid_wmedian_rows"),

    (~pl.col("valid_wweak"))
    .sum()
    .alias("invalid_wweak_rows"),

    pl.col("wmedian")
    .filter(pl.col("valid_wmedian"))
    .min()
    .alias("wmedian_min"),

    pl.col("wmedian")
    .filter(pl.col("valid_wmedian"))
    .max()
    .alias("wmedian_max"),

    pl.col("wweak")
    .filter(pl.col("valid_wweak"))
    .min()
    .alias("wweak_min"),

    pl.col("wweak")
    .filter(pl.col("valid_wweak"))
    .max()
    .alias("wweak_max"),
)

print(weight_quality)

print()
print("SAMPLE COUNTS")

sample_counts = data.select(
    pl.col("model1_complete")
    .sum()
    .alias("model1_complete_rows"),

    pl.col("model2_complete")
    .sum()
    .alias("model2_complete_rows"),

    (
        pl.col("model1_complete")
        & pl.col("valid_wmedian")
    )
    .sum()
    .alias("model1_wmedian_rows"),

    (
        pl.col("model1_complete")
        & pl.col("valid_wweak")
    )
    .sum()
    .alias("model1_wweak_rows"),

    (
        pl.col("model2_complete")
        & pl.col("valid_wmedian")
    )
    .sum()
    .alias("model2_wmedian_rows"),

    (
        pl.col("model2_complete")
        & pl.col("valid_wweak")
    )
    .sum()
    .alias("model2_wweak_rows"),

    (
        pl.col("model1_complete")
        & pl.col("valid_wmedian")
        & pl.col("valid_wweak")
    )
    .sum()
    .alias("model1_common_weight_rows"),

    (
        pl.col("model2_complete")
        & pl.col("valid_wmedian")
        & pl.col("valid_wweak")
    )
    .sum()
    .alias("model2_common_weight_rows"),
)

print(sample_counts)

print()
print("WEIGHT OVERLAP")

print(
    data.select(
        (
            pl.col("valid_wmedian")
            & pl.col("valid_wweak")
        )
        .sum()
        .alias("both_weights_valid"),

        (
            pl.col("valid_wmedian")
            & ~pl.col("valid_wweak")
        )
        .sum()
        .alias("only_wmedian_valid"),

        (
            ~pl.col("valid_wmedian")
            & pl.col("valid_wweak")
        )
        .sum()
        .alias("only_wweak_valid"),

        (
            ~pl.col("valid_wmedian")
            & ~pl.col("valid_wweak")
        )
        .sum()
        .alias("neither_weight_valid"),
    )
)

print()
print("24A AUDIT COMPLETED")
print()

import math

import polars as pl
import svy


print("=" * 72)
print("24B. SURVEY WEIGHT ROBUSTNESS ANALYSIS")
print("=" * 72)

model1_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
    svy.Cat("a4a"),
    svy.Cat("a2"),
]

model2_predictors = model1_predictors + [
    "bank_account",
]

design_median = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wmedian",
    singleton="scale",
)

design_weak = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wweak",
    singleton="scale",
)

sample_median = svy.Sample(
    data=data,
    design=design_median,
)

sample_weak = svy.Sample(
    data=data,
    design=design_weak,
)


def fit_three_models(sample_obj, weight_name):
    fitted_models = {}

    fitted_models[f"model1_full_{weight_name}"] = (
        sample_obj.glm.fit(
            y="major_obstacle",
            x=model1_predictors,
            family="binomial",
            link="logit",
            where=(svy.col("model1_complete") == True),
        )
    )

    fitted_models[f"model1_common_{weight_name}"] = (
        sample_obj.glm.fit(
            y="major_obstacle",
            x=model1_predictors,
            family="binomial",
            link="logit",
            where=(svy.col("model2_complete") == True),
        )
    )

    fitted_models[f"model2_common_{weight_name}"] = (
        sample_obj.glm.fit(
            y="major_obstacle",
            x=model2_predictors,
            family="binomial",
            link="logit",
            where=(svy.col("model2_complete") == True),
        )
    )

    return fitted_models


models_median = fit_three_models(
    sample_median,
    "wmedian",
)

models_weak = fit_three_models(
    sample_weak,
    "wweak",
)

all_models = {
    **models_median,
    **models_weak,
}

print()
print("MODEL SAMPLE SIZES")

sample_rows = [
    {
        "model": "model1_full_wmedian",
        "n": int(
            data.filter(
                pl.col("model1_complete")
            ).height
        ),
        "expected_n": 2632,
    },
    {
        "model": "model1_common_wmedian",
        "n": int(
            data.filter(
                pl.col("model2_complete")
            ).height
        ),
        "expected_n": 2528,
    },
    {
        "model": "model2_common_wmedian",
        "n": int(
            data.filter(
                pl.col("model2_complete")
            ).height
        ),
        "expected_n": 2528,
    },
    {
        "model": "model1_full_wweak",
        "n": int(
            data.filter(
                pl.col("model1_complete")
            ).height
        ),
        "expected_n": 2632,
    },
    {
        "model": "model1_common_wweak",
        "n": int(
            data.filter(
                pl.col("model2_complete")
            ).height
        ),
        "expected_n": 2528,
    },
    {
        "model": "model2_common_wweak",
        "n": int(
            data.filter(
                pl.col("model2_complete")
            ).height
        ),
        "expected_n": 2528,
    },
]

sample_summary = pl.DataFrame(sample_rows)

sample_summary = sample_summary.with_columns(
    (
        pl.col("n") == pl.col("expected_n")
    ).alias("sample_size_matches")
)

print(sample_summary)

if not sample_summary["sample_size_matches"].all():
    raise ValueError(
        "Unexpected sample size detected."
    )

print()
print("SAMPLE SIZE VERIFICATION PASSED")

coefficient_rows = []

for model_name, fitted_model in all_models.items():
    fitted = fitted_model.fitted

    for coefficient in fitted.coefs:
        beta = float(coefficient.est)
        se = float(coefficient.se)
        ci_low = float(coefficient.lci)
        ci_high = float(coefficient.uci)
        p_value = float(coefficient.wald.p_value)

        coefficient_rows.append({
            "model": model_name,
            "term": coefficient.term,
            "beta": beta,
            "se": se,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(ci_low),
            "or_ci_high": math.exp(ci_high),
        })

coefficient_results = pl.DataFrame(coefficient_rows)

print()
print("COEFFICIENT EXTRACTION SUMMARY")

print(
    coefficient_results.group_by("model").agg(
        pl.len().alias("coefficient_count"),
        pl.col("beta").is_finite().all().alias(
            "all_beta_finite"
        ),
        pl.col("se").is_finite().all().alias(
            "all_se_finite"
        ),
        pl.col("p_value").is_finite().all().alias(
            "all_p_values_finite"
        ),
    ).sort("model")
)

print()
print("WEIGHT SENSITIVITY COMPARISON")

comparison_rows = []

for model_base in [
    "model1_full",
    "model1_common",
    "model2_common",
]:
    median_name = f"{model_base}_wmedian"
    weak_name = f"{model_base}_wweak"

    median_table = (
        coefficient_results
        .filter(pl.col("model") == median_name)
        .drop("model")
    )

    weak_table = (
        coefficient_results
        .filter(pl.col("model") == weak_name)
        .drop("model")
    )

    paired = median_table.join(
        weak_table,
        on="term",
        how="full",
        suffix="_weak",
        coalesce=True,
    )

    paired = paired.with_columns(
        pl.lit(model_base).alias("model")
    )

    comparison_rows.append(paired)


comparison = pl.concat(
    comparison_rows,
    how="diagonal_relaxed",
)

comparison = comparison.with_columns(
    (
        pl.col("beta_weak") - pl.col("beta")
    ).alias("beta_difference"),

    (
        pl.col("odds_ratio_weak")
        - pl.col("odds_ratio")
    ).alias("or_difference"),

    (
        pl.col("p_value_weak")
        - pl.col("p_value")
    ).alias("p_value_difference"),

    (
        (pl.col("beta") * pl.col("beta_weak")) >= 0
    ).alias("same_coefficient_direction"),

    (
        (pl.col("p_value") < 0.05)
        == (pl.col("p_value_weak") < 0.05)
    ).alias("same_significance_status"),
)

print()
print("PAIRED MODEL COMPARISON")

print(
    comparison.select(
        "model",
        "term",
        "beta",
        "beta_weak",
        "beta_difference",
        "odds_ratio",
        "odds_ratio_weak",
        "p_value",
        "p_value_weak",
        "same_coefficient_direction",
        "same_significance_status",
    ).sort(["model", "term"])
)

print()
print("ROBUSTNESS SUMMARY")

robustness_summary = comparison.group_by("model").agg(
    pl.len().alias("matched_coefficients"),

    pl.col("same_coefficient_direction")
    .sum()
    .alias("same_direction_count"),

    pl.col("same_significance_status")
    .sum()
    .alias("same_significance_count"),

    pl.col("beta_difference")
    .abs()
    .max()
    .alias("maximum_absolute_beta_difference"),

    pl.col("or_difference")
    .abs()
    .max()
    .alias("maximum_absolute_or_difference"),

    pl.col("p_value_difference")
    .abs()
    .max()
    .alias("maximum_absolute_p_difference"),
)

print(robustness_summary.sort("model"))

print()
print("PRIMARY PREDICTOR COMPARISON")

primary_comparison = comparison.filter(
    pl.col("term").is_in([
        "firm_age",
        "log_employees",
        "female_owner",
        "bank_account",
    ])
)

print(
    primary_comparison.select(
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "beta_weak",
        "se_weak",
        "p_value_weak",
        "odds_ratio_weak",
        "or_ci_low_weak",
        "or_ci_high_weak",
        "same_coefficient_direction",
        "same_significance_status",
    ).sort(["model", "term"])
)

print()
print("24B ROBUSTNESS ANALYSIS COMPLETED")
print()

print("=" * 72)
print("24B-1. IDENTIFICATION OF WEIGHT SENSITIVITY CHANGES")
print("=" * 72)

print()
print("COEFFICIENTS WITH CHANGED SIGNIFICANCE STATUS")

changed_significance = comparison.filter(
    ~pl.col("same_significance_status")
)

print(
    changed_significance.select(
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
        "odds_ratio",
        "beta_weak",
        "se_weak",
        "p_value_weak",
        "ci_low_weak",
        "ci_high_weak",
        "odds_ratio_weak",
    ).sort(["model", "term"])
)

print()
print("COEFFICIENTS WITH CHANGED DIRECTION")

changed_direction = comparison.filter(
    ~pl.col("same_coefficient_direction")
)

print(
    changed_direction.select(
        "model",
        "term",
        "beta",
        "beta_weak",
        "p_value",
        "p_value_weak",
        "odds_ratio",
        "odds_ratio_weak",
    ).sort(["model", "term"])
)

print()
print("COEFFICIENTS NEAR THE 5 PERCENT THRESHOLD")

near_threshold = comparison.filter(
    (
        (pl.col("p_value") - 0.05).abs() <= 0.02
    )
    |
    (
        (pl.col("p_value_weak") - 0.05).abs() <= 0.02
    )
)

print(
    near_threshold.select(
        "model",
        "term",
        "p_value",
        "p_value_weak",
        "odds_ratio",
        "odds_ratio_weak",
        "same_significance_status",
    ).sort(["model", "term"])
)

print()
print("24B-1 COMPLETED")

print("=" * 72)
print("24C-1. ALTERNATIVE OUTCOME DEFINITIONS")
print("=" * 72)

source_outcome = (
    pl.read_parquet(source_path)
    .select(
        "idstd",
        "k30",
    )
)

data_outcome = data.join(
    source_outcome,
    on="idstd",
    how="left",
)

data_outcome = data_outcome.with_columns(
    pl.when(
        pl.col("k30").is_between(0, 4, closed="both")
    )
    .then(
        (pl.col("k30") >= 3).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("major_obstacle_rebuilt"),

    pl.when(
        pl.col("k30").is_between(0, 4, closed="both")
    )
    .then(
        (pl.col("k30") >= 1).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("any_obstacle"),

    pl.when(
        pl.col("k30").is_between(0, 4, closed="both")
    )
    .then(
        (pl.col("k30") == 4).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("severe_obstacle"),
)

print()
print("SOURCE OUTCOME DISTRIBUTION")

print(
    data_outcome
    .group_by("k30")
    .agg(
        pl.len().alias("observations")
    )
    .sort("k30")
)

print()
print("OUTCOME DATA QUALITY AUDIT")

outcome_audit = data_outcome.select(
    pl.len().alias("total_observations"),

    pl.col("k30")
    .is_null()
    .sum()
    .alias("missing_k30"),

    (
        pl.col("k30").is_not_null()
        & ~pl.col("k30").is_between(
            0,
            4,
            closed="both",
        )
    )
    .sum()
    .alias("invalid_k30"),

    pl.col("major_obstacle")
    .is_not_null()
    .sum()
    .alias("original_major_valid"),

    pl.col("major_obstacle_rebuilt")
    .is_not_null()
    .sum()
    .alias("rebuilt_major_valid"),

    (
        pl.col("major_obstacle")
        != pl.col("major_obstacle_rebuilt")
    )
    .sum()
    .alias("major_definition_mismatches"),
)

print(outcome_audit)

print()
print("ALTERNATIVE OUTCOME COUNTS")

outcome_counts = data_outcome.select(
    pl.len().alias("total_observations"),

    pl.col("major_obstacle_rebuilt")
    .is_not_null()
    .sum()
    .alias("major_valid"),

    (pl.col("major_obstacle_rebuilt") == 1)
    .sum()
    .alias("major_positive"),

    (pl.col("major_obstacle_rebuilt") == 0)
    .sum()
    .alias("major_negative"),

    pl.col("any_obstacle")
    .is_not_null()
    .sum()
    .alias("any_valid"),

    (pl.col("any_obstacle") == 1)
    .sum()
    .alias("any_positive"),

    (pl.col("any_obstacle") == 0)
    .sum()
    .alias("any_negative"),

    pl.col("severe_obstacle")
    .is_not_null()
    .sum()
    .alias("severe_valid"),

    (pl.col("severe_obstacle") == 1)
    .sum()
    .alias("severe_positive"),

    (pl.col("severe_obstacle") == 0)
    .sum()
    .alias("severe_negative"),
)

print(outcome_counts)

print()
print("24C-1 COMPLETED")

print("=" * 72)
print("24C-2. ALTERNATIVE OUTCOME SAMPLE AUDIT")
print("=" * 72)

model1_variables = [
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
    "wmedian",
]

model2_variables = model1_variables + [
    "bank_account",
]

data_outcome = data_outcome.with_columns(
    pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in model1_variables
        ]
    ).alias("model1_full_complete"),

    pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in model2_variables
        ]
    ).alias("common_complete"),
)

outcome_variables = [
    "major_obstacle_rebuilt",
    "any_obstacle",
    "severe_obstacle",
]

sample_definitions = [
    ("model1_full", "model1_full_complete"),
    ("model1_common", "common_complete"),
    ("model2_common", "common_complete"),
]

audit_rows = []

for model_name, sample_column in sample_definitions:
    for outcome in outcome_variables:
        result = (
            data_outcome
            .filter(
                pl.col(sample_column)
                & pl.col(outcome).is_not_null()
            )
            .select(
                pl.len().alias("valid_n"),
                (pl.col(outcome) == 1)
                .sum()
                .alias("positive_n"),
                (pl.col(outcome) == 0)
                .sum()
                .alias("negative_n"),
            )
            .with_columns(
                pl.lit(model_name).alias("model"),
                pl.lit(outcome).alias("outcome"),
                (
                    pl.col("positive_n")
                    / pl.col("valid_n")
                    * 100
                ).alias("positive_pct"),
            )
        )

        audit_rows.append(result)

sample_audit = pl.concat(audit_rows).select(
    "model",
    "outcome",
    "valid_n",
    "positive_n",
    "negative_n",
    "positive_pct",
)

print()
print("SAMPLE SIZE VALIDATION")

print(
    data_outcome.select(
        pl.col("model1_full_complete")
        .sum()
        .alias("model1_full_n"),

        pl.col("common_complete")
        .sum()
        .alias("common_sample_n"),
    )
)

print()
print("OUTCOME DISTRIBUTION BY MODEL SAMPLE")

print(
    sample_audit.sort(
        ["model", "outcome"]
    )
)

print()
print("24C-2 COMPLETED")

import math

import polars as pl
import svy


print("=" * 72)
print("24C-3. ALTERNATIVE OUTCOME REGRESSION")
print("=" * 72)

model1_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
    svy.Cat("a4a"),
    svy.Cat("a2"),
]

model2_predictors = model1_predictors + [
    "bank_account",
]

outcome_variables = [
    "any_obstacle",
    "major_obstacle",
    "severe_obstacle",
]

model_specs = {
    "model1_full": {
        "predictors": model1_predictors,
        "sample_flag": "model1_full_complete",
        "expected_n": 2632,
    },
    "model1_common": {
        "predictors": model1_predictors,
        "sample_flag": "common_complete",
        "expected_n": 2528,
    },
    "model2_common": {
        "predictors": model2_predictors,
        "sample_flag": "common_complete",
        "expected_n": 2528,
    },
}

design_median_outcome = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wmedian",
    singleton="scale",
)

sample_median_outcome = svy.Sample(
    data=data_outcome,
    design=design_median_outcome,
)


print()
print("OUTCOME SAMPLE SIZE VERIFICATION")

sample_rows = []

for model_name, model_spec in model_specs.items():
    sample_flag = model_spec["sample_flag"]
    expected_n = model_spec["expected_n"]

    for outcome in outcome_variables:
        actual_n = data_outcome.filter(
            pl.col(sample_flag)
            & pl.col(outcome).is_not_null()
        ).height

        sample_rows.append({
            "model": model_name,
            "outcome": outcome,
            "n": actual_n,
            "expected_n": expected_n,
            "sample_size_matches": actual_n == expected_n,
        })

sample_summary = pl.DataFrame(sample_rows)

print(sample_summary.sort(["model", "outcome"]))

if not sample_summary["sample_size_matches"].all():
    raise ValueError(
        "Unexpected sample size detected in alternative outcome analysis."
    )

print()
print("ALL OUTCOME SAMPLE SIZES VERIFIED")


def fit_alternative_outcomes(
    sample_obj,
    model_specifications,
    outcomes,
):
    fitted_models = {}

    for model_name, model_spec in model_specifications.items():
        sample_flag = model_spec["sample_flag"]
        predictors = model_spec["predictors"]

        for outcome in outcomes:
            result_name = f"{model_name}_{outcome}"

            fitted_models[result_name] = (
                sample_obj.glm.fit(
                    y=outcome,
                    x=predictors,
                    family="binomial",
                    link="logit",
                    where=(
                        (svy.col(sample_flag) == True)
                        & (svy.col(outcome).is_not_null())
                    ),
                )
            )

    return fitted_models


alternative_models = fit_alternative_outcomes(
    sample_obj=sample_median_outcome,
    model_specifications=model_specs,
    outcomes=outcome_variables,
)


print()
print("ALTERNATIVE OUTCOME REGRESSIONS COMPLETED")


coefficient_rows = []

for model_name, fitted_model in alternative_models.items():
    fitted = fitted_model.fitted

    for coefficient in fitted.coefs:
        beta = float(coefficient.est)
        se = float(coefficient.se)
        ci_low = float(coefficient.lci)
        ci_high = float(coefficient.uci)
        p_value = float(coefficient.wald.p_value)

        coefficient_rows.append({
            "model": model_name,
            "term": coefficient.term,
            "beta": beta,
            "se": se,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(ci_low),
            "or_ci_high": math.exp(ci_high),
        })

alternative_coefficient_results = pl.DataFrame(
    coefficient_rows
)


print()
print("COEFFICIENT EXTRACTION SUMMARY")

coefficient_summary = (
    alternative_coefficient_results
    .group_by("model")
    .agg(
        pl.len().alias("coefficient_count"),
        pl.col("beta").is_finite().all().alias(
            "all_beta_finite"
        ),
        pl.col("se").is_finite().all().alias(
            "all_se_finite"
        ),
        pl.col("p_value").is_finite().all().alias(
            "all_p_values_finite"
        ),
        pl.col("ci_low").is_finite().all().alias(
            "all_ci_low_finite"
        ),
        pl.col("ci_high").is_finite().all().alias(
            "all_ci_high_finite"
        ),
    )
    .sort("model")
)

print(coefficient_summary)


print()
print("PRIMARY PREDICTOR RESULTS")

primary_terms = [
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
]

primary_results = (
    alternative_coefficient_results
    .filter(
        pl.col("term").is_in(primary_terms)
    )
    .sort(["model", "term"])
)

print(
    primary_results.select(
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    )
)


print()
print("OUTCOME-LEVEL MODEL SUMMARY")

outcome_model_summary = (
    alternative_coefficient_results
    .with_columns(
        pl.col("model")
        .str.extract(
            r"_(any_obstacle|major_obstacle|severe_obstacle)$",
            group_index=1,
        )
        .alias("outcome"),

        pl.col("model")
        .str.replace(
            r"_(any_obstacle|major_obstacle|severe_obstacle)$",
            "",
        )
        .alias("model_specification"),
    )
    .group_by(["outcome", "model_specification"])
    .agg(
        pl.len().alias("coefficient_count"),
        pl.col("beta").is_finite().all().alias(
            "all_beta_finite"
        ),
        pl.col("se").is_finite().all().alias(
            "all_se_finite"
        ),
        pl.col("p_value").is_finite().all().alias(
            "all_p_values_finite"
        ),
    )
    .sort(["outcome", "model_specification"])
)

print(outcome_model_summary)


print()
print("24C-3 ALTERNATIVE OUTCOME REGRESSION COMPLETED")
print()

import polars as pl


print("=" * 72)
print("24C-4. PRIMARY PREDICTOR AND SEPARATION DIAGNOSTICS")
print("=" * 72)


primary_terms = [
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
]


print()
print("PRIMARY PREDICTOR COMPARISON")

primary_comparison = (
    alternative_coefficient_results
    .filter(
        pl.col("term").is_in(primary_terms)
        & (
            pl.col("model").str.contains("any_obstacle")
            | pl.col("model").str.contains("major_obstacle")
        )
    )
    .with_columns(
        pl.col("model")
        .str.extract(
            r"^(model1_full|model1_common|model2_common)",
            group_index=1,
        )
        .alias("model_specification"),

        pl.col("model")
        .str.extract(
            r"_(any_obstacle|major_obstacle)$",
            group_index=1,
        )
        .alias("outcome"),
    )
    .select(
        "model_specification",
        "outcome",
        "term",
        "beta",
        "se",
        "p_value",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    )
    .sort([
        "model_specification",
        "term",
        "outcome",
    ])
)

print(primary_comparison)


print()
print("PRIMARY PREDICTOR SIGNIFICANCE SUMMARY")

primary_significance = (
    primary_comparison
    .with_columns(
        (pl.col("p_value") < 0.05).alias(
            "significant_at_5pct"
        )
    )
    .select(
        "model_specification",
        "outcome",
        "term",
        "p_value",
        "significant_at_5pct",
    )
    .sort([
        "model_specification",
        "term",
        "outcome",
    ])
)

print(primary_significance)


print()
print("MAJOR OUTCOME CONSISTENCY CHECK")

major_consistency = (
    alternative_coefficient_results
    .filter(
        pl.col("model").is_in([
            "model1_full_major_obstacle",
            "model1_common_major_obstacle",
            "model2_common_major_obstacle",
        ])
    )
    .with_columns(
        pl.col("model").str.replace(
            r"_major_obstacle$",
            "",
        ).alias("model_specification")
    )
    .select(
        "model_specification",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
    )
    .sort([
        "model_specification",
        "term",
    ])
)

print(major_consistency)


print()
print("SEVERE OUTCOME DISTRIBUTION BY A2 CATEGORY")

severe_a2_distribution = (
    data_outcome
    .filter(
        pl.col("model1_full_complete")
        & pl.col("severe_obstacle").is_not_null()
        & pl.col("a2").is_not_null()
    )
    .group_by("a2")
    .agg(
        pl.len().alias("observations"),
        pl.col("severe_obstacle")
        .sum()
        .alias("severe_cases"),
        (
            pl.col("severe_obstacle").mean() * 100
        ).alias("severe_percentage"),
    )
    .sort("a2")
)

print(severe_a2_distribution)


print()
print("SEVERE OUTCOME DISTRIBUTION BY A2 CATEGORY - COMMON SAMPLE")

severe_a2_common = (
    data_outcome
    .filter(
        pl.col("common_complete")
        & pl.col("severe_obstacle").is_not_null()
        & pl.col("a2").is_not_null()
    )
    .group_by("a2")
    .agg(
        pl.len().alias("observations"),
        pl.col("severe_obstacle")
        .sum()
        .alias("severe_cases"),
        (
            pl.col("severe_obstacle").mean() * 100
        ).alias("severe_percentage"),
    )
    .sort("a2")
)

print(severe_a2_common)


print()
print("24C-4 DIAGNOSTICS COMPLETED")

import polars as pl


print("=" * 72)
print("24C-5. MAJOR OUTCOME REPLICATION AND SEPARATION CHECK")
print("=" * 72)


print()
print("MAJOR OUTCOME REPLICATION CHECK")


major_alternative = (
    alternative_coefficient_results
    .filter(
        pl.col("model").str.ends_with("major_obstacle")
    )
    .with_columns(
        pl.col("model")
        .str.replace(
            r"_major_obstacle$",
            "",
        )
        .alias("model_base")
    )
    .select(
        "model_base",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
    )
)

major_original = (
    coefficient_results
    .filter(
        pl.col("model").is_in([
            "model1_full_wmedian",
            "model1_common_wmedian",
            "model2_common_wmedian",
        ])
    )
    .with_columns(
        pl.col("model")
        .str.replace(
            r"_wmedian$",
            "",
        )
        .alias("model_base")
    )
    .select(
        "model_base",
        "term",
        pl.col("beta").alias("beta_original"),
        pl.col("se").alias("se_original"),
        pl.col("p_value").alias("p_value_original"),
        pl.col("ci_low").alias("ci_low_original"),
        pl.col("ci_high").alias("ci_high_original"),
    )
)

replication = major_alternative.join(
    major_original,
    on=["model_base", "term"],
    how="full",
    coalesce=True,
)

replication = replication.with_columns(
    (pl.col("beta") - pl.col("beta_original"))
    .abs()
    .alias("beta_absolute_difference"),

    (pl.col("se") - pl.col("se_original"))
    .abs()
    .alias("se_absolute_difference"),

    (pl.col("p_value") - pl.col("p_value_original"))
    .abs()
    .alias("p_value_absolute_difference"),

    (pl.col("ci_low") - pl.col("ci_low_original"))
    .abs()
    .alias("ci_low_absolute_difference"),

    (pl.col("ci_high") - pl.col("ci_high_original"))
    .abs()
    .alias("ci_high_absolute_difference"),
)

replication_summary = replication.group_by(
    "model_base"
).agg(
    pl.len().alias("matched_rows"),

    pl.col("beta_absolute_difference")
    .max()
    .alias("max_beta_difference"),

    pl.col("se_absolute_difference")
    .max()
    .alias("max_se_difference"),

    pl.col("p_value_absolute_difference")
    .max()
    .alias("max_p_value_difference"),

    pl.col("ci_low_absolute_difference")
    .max()
    .alias("max_ci_low_difference"),

    pl.col("ci_high_absolute_difference")
    .max()
    .alias("max_ci_high_difference"),
).sort("model_base")

print(replication_summary)


print()
print("NON-IDENTICAL REPLICATION ROWS")

non_identical = replication.filter(
    (
        pl.col("beta_absolute_difference") > 1e-10
    )
    |
    (
        pl.col("se_absolute_difference") > 1e-10
    )
    |
    (
        pl.col("p_value_absolute_difference") > 1e-10
    )
    |
    (
        pl.col("ci_low_absolute_difference") > 1e-10
    )
    |
    (
        pl.col("ci_high_absolute_difference") > 1e-10
    )
)

print(non_identical)


print()
print("ALL A2 CATEGORIES IN SEVERE OUTCOME SAMPLE")

severe_a2_all = (
    data_outcome
    .filter(
        pl.col("model1_full_complete")
        & pl.col("severe_obstacle").is_not_null()
        & pl.col("a2").is_not_null()
    )
    .group_by("a2")
    .agg(
        pl.len().alias("observations"),
        pl.col("severe_obstacle")
        .sum()
        .alias("severe_cases"),
        (
            pl.col("severe_obstacle").mean() * 100
        ).alias("severe_percentage"),
    )
    .sort("a2")
)

for row in severe_a2_all.to_dicts():
    print(row)


print()
print("SEPARATION CATEGORY CHECK")

separation_categories = (
    severe_a2_all
    .filter(
        pl.col("a2").is_in([2, 7, 10, 18])
    )
)

print(separation_categories)


print()
print("24C-5 COMPLETED")

import polars as pl


print()
print("=" * 72)
print("24C-6. SEVERE OUTCOME SEPARATION DIAGNOSTICS")
print("=" * 72)


diagnostic_specs = [
    ("model1_full", "model1_full_complete"),
    ("model1_common", "common_complete"),
    ("model2_common", "common_complete"),
]


for model_name, sample_flag in diagnostic_specs:

    print()
    print(f"MODEL: {model_name}")
    print("-" * 72)

    model_data = (
        data_outcome
        .filter(
            pl.col(sample_flag)
            & pl.col("severe_obstacle").is_not_null()
        )
    )

    print("Observations:", model_data.height)

    for variable in ["a2", "a4a"]:

        print()
        print(f"SEVERE OUTCOME DISTRIBUTION BY {variable.upper()}")

        distribution = (
            model_data
            .filter(pl.col(variable).is_not_null())
            .group_by(variable)
            .agg(
                pl.len().alias("observations"),

                pl.col("severe_obstacle")
                .sum()
                .alias("severe_cases"),

                (
                    pl.col("severe_obstacle").mean() * 100
                )
                .alias("severe_percentage"),

                pl.col("wmedian")
                .filter(pl.col("severe_obstacle") == 1)
                .sum()
                .alias("weighted_severe_cases"),

                pl.col("wmedian")
                .sum()
                .alias("total_weight"),
            )
            .with_columns(
                (
                    pl.col("weighted_severe_cases")
                    / pl.col("total_weight")
                    * 100
                ).alias("weighted_severe_percentage"),

                (
                    pl.col("severe_cases") == 0
                ).alias("zero_event_category"),
            )
            .sort(variable)
        )

        print(distribution)

        zero_event_categories = distribution.filter(
            pl.col("zero_event_category")
        )

        print()
        print(f"ZERO-EVENT CATEGORIES IN {variable.upper()}")
        print(zero_event_categories)


print()
print("24C-6 COMPLETED")

print()
print("=" * 72)
print("24C-7. SEVERE OUTCOME REPORTING AND MODEL DISPOSITION")
print("=" * 72)


severe_summary_frames = []
severe_zero_event_frames = []
severe_disposition_records = []


for model_name, sample_flag in diagnostic_specs:

    print()
    print(f"MODEL: {model_name}")
    print("-" * 72)

    model_data = (
        data_outcome
        .filter(
            pl.col(sample_flag)
            & pl.col("severe_obstacle").is_not_null()
        )
    )

    total_observations = model_data.height

    severe_cases = int(
        model_data["severe_obstacle"].sum() or 0
    )

    total_weight = model_data["wmedian"].sum() or 0

    weighted_severe_cases = (
        model_data
        .filter(pl.col("severe_obstacle") == 1)
        ["wmedian"]
        .sum()
        or 0
    )

    severe_percentage = (
        severe_cases / total_observations * 100
        if total_observations > 0
        else None
    )

    weighted_severe_percentage = (
        weighted_severe_cases / total_weight * 100
        if total_weight > 0
        else None
    )

    summary_record = {
        "model": model_name,
        "sample_flag": sample_flag,
        "observations": total_observations,
        "severe_cases": severe_cases,
        "non_severe_cases": total_observations - severe_cases,
        "severe_percentage": severe_percentage,
        "weighted_severe_cases": weighted_severe_cases,
        "total_weight": total_weight,
        "weighted_severe_percentage": weighted_severe_percentage,
    }

    severe_summary_frames.append(
        pl.DataFrame([summary_record])
    )

    print("Observations:", total_observations)
    print("Severe cases:", severe_cases)
    print("Severe prevalence:", severe_percentage)
    print("Weighted severe prevalence:", weighted_severe_percentage)

    for variable in ["a2", "a4a"]:

        distribution = (
            model_data
            .filter(pl.col(variable).is_not_null())
            .group_by(variable)
            .agg(
                pl.len().alias("observations"),

                pl.col("severe_obstacle")
                .sum()
                .alias("severe_cases"),

                pl.col("wmedian")
                .filter(pl.col("severe_obstacle") == 1)
                .sum()
                .alias("weighted_severe_cases"),

                pl.col("wmedian")
                .sum()
                .alias("total_weight"),
            )
            .with_columns(
                (
                    pl.col("severe_cases")
                    / pl.col("observations")
                    * 100
                ).alias("severe_percentage"),

                pl.when(pl.col("total_weight") > 0)
                .then(
                    pl.col("weighted_severe_cases")
                    / pl.col("total_weight")
                    * 100
                )
                .otherwise(None)
                .alias("weighted_severe_percentage"),

                (
                    pl.col("severe_cases") == 0
                ).alias("zero_event_category"),

                pl.lit(model_name).alias("model"),

                pl.lit(variable).alias("predictor"),
            )
            .sort(variable)
        )

        zero_events = distribution.filter(
            pl.col("zero_event_category")
        )

        if zero_events.height > 0:
            severe_zero_event_frames.append(
                zero_events
            )

        print()
        print(f"ZERO-EVENT CATEGORIES: {variable.upper()}")
        print(zero_events)

    severe_disposition_records.append(
        {
            "model": model_name,
            "outcome": "severe_obstacle",
            "disposition": "Descriptive and exploratory only",
            "inferential_use": False,
            "reason": (
                "Quasi-complete separation and zero-event "
                "predictor categories"
            ),
            "reporting_recommendation": (
                "Report prevalence and category distributions; "
                "do not interpret unstable logistic coefficients"
            ),
        }
    )


print()
print("=" * 72)
print("24C-7A. SEVERE OUTCOME SUMMARY")
print("=" * 72)

severe_summary = pl.concat(
    severe_summary_frames,
    how="diagonal_relaxed"
)

print(severe_summary)


print()
print("=" * 72)
print("24C-7B. ZERO-EVENT CATEGORY SUMMARY")
print("=" * 72)

if severe_zero_event_frames:

    severe_zero_event_summary = pl.concat(
        severe_zero_event_frames,
        how="diagonal_relaxed"
    )

    severe_zero_event_summary = (
        pl.concat(
            severe_zero_event_frames,
            how="diagonal_relaxed"
        )
        .select(
            "model",
            "predictor",
            "observations",
            "severe_cases",
            "severe_percentage",
            "weighted_severe_cases",
            "total_weight",
            "weighted_severe_percentage",
            "zero_event_category",
            *[
                variable
                for variable in ["a2", "a4a"]
                if variable in pl.concat(
                    severe_zero_event_frames,
                    how="diagonal_relaxed"
                ).columns
            ]
        )
    )

    print(severe_zero_event_summary)

else:

    severe_zero_event_summary = pl.DataFrame()

    print("No zero-event categories identified.")


print()
print("=" * 72)
print("24C-7C. MODEL DISPOSITION")
print("=" * 72)

severe_model_disposition = pl.DataFrame(
    severe_disposition_records
)

print(severe_model_disposition)


print()
print("=" * 72)
print("24C-7D. EXPORT REPORTING FILES")
print("=" * 72)

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

processed_dir = project_root / "data" / "processed"

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)

severe_summary_path = (
    processed_dir / "24c7_severe_outcome_summary.csv"
)

severe_zero_event_path = (
    processed_dir / "24c7_severe_zero_event_categories.csv"
)

severe_disposition_path = (
    processed_dir / "24c7_severe_model_disposition.csv"
)

severe_summary.write_csv(severe_summary_path)

severe_zero_event_summary.write_csv(
    severe_zero_event_path
)

severe_model_disposition.write_csv(
    severe_disposition_path
)

print("Saved:", severe_summary_path)
print("Saved:", severe_zero_event_path)
print("Saved:", severe_disposition_path)


print()
print("=" * 72)
print("24C-7 COMPLETED")
print("=" * 72)

print("Severe outcome reporting completed.")
print("Severe regression is retained for descriptive purposes only.")
print("No unstable severe-outcome coefficients are used for inference.")


print()
print()
print("=" * 72)
print("24C-8. FINAL ROBUSTNESS ASSESSMENT AND MODEL SELECTION")
print("=" * 72)


outcome_specs = [
    (
        "major_obstacle",
        "Primary inferential outcome",
        True,
        "Primary measure of substantial financial obstacles"
    ),
    (
        "any_obstacle",
        "Inferential robustness outcome",
        True,
        "Broader alternative outcome definition"
    ),
    (
        "severe_obstacle",
        "Descriptive and exploratory outcome",
        False,
        "Low prevalence and zero-event categories raise separation concerns"
    ),
]


outcome_comparison_records = []
model_selection_records = []


for model_name, sample_flag in diagnostic_specs:

    print()
    print(f"MODEL: {model_name}")
    print("-" * 72)

    model_data = data_outcome.filter(
        pl.col(sample_flag)
    )

    for outcome, role, inferential_use, rationale in outcome_specs:

        if outcome not in model_data.columns:
            raise ValueError(
                f"Missing outcome column: {outcome}"
            )

        outcome_data = (
            model_data
            .filter(pl.col(outcome).is_not_null())
        )

        observed_values = set(
            outcome_data
            .select(pl.col(outcome).unique())
            .to_series()
            .to_list()
        )

        if not observed_values.issubset({0, 1}):
            raise ValueError(
                f"{outcome} must contain only binary values 0 and 1. "
                f"Found: {observed_values}"
            )

        observations = outcome_data.height

        outcome_cases = int(
            outcome_data[outcome].sum() or 0
        )

        non_outcome_cases = (
            observations - outcome_cases
        )

        total_weight = (
            outcome_data["wmedian"].sum() or 0
        )

        weighted_outcome_cases = (
            outcome_data
            .filter(pl.col(outcome) == 1)
            ["wmedian"]
            .sum()
            or 0
        )

        unweighted_prevalence = (
            outcome_cases / observations * 100
            if observations > 0
            else None
        )

        weighted_prevalence = (
            weighted_outcome_cases / total_weight * 100
            if total_weight > 0
            else None
        )

        outcome_comparison_records.append(
            {
                "model": model_name,
                "sample_flag": sample_flag,
                "outcome": outcome,
                "observations": observations,
                "outcome_cases": outcome_cases,
                "non_outcome_cases": non_outcome_cases,
                "unweighted_prevalence": unweighted_prevalence,
                "weighted_outcome_weight_sum": weighted_outcome_cases,
                "total_weight": total_weight,
                "weighted_prevalence": weighted_prevalence,
            }
        )

        model_selection_records.append(
            {
                "model": model_name,
                "sample_flag": sample_flag,
                "outcome": outcome,
                "model_role": role,
                "inferential_use": inferential_use,
                "decision": (
                    "Retain for inference"
                    if inferential_use
                    else "Descriptive and exploratory only"
                ),
                "rationale": rationale,
            }
        )

        print()
        print(f"OUTCOME: {outcome}")
        print("Observations:", observations)
        print("Outcome cases:", outcome_cases)
        print("Unweighted prevalence:", unweighted_prevalence)
        print("Weighted prevalence:", weighted_prevalence)


print()
print("=" * 72)
print("24C-8A. OUTCOME COMPARISON")
print("=" * 72)

outcome_comparison = pl.DataFrame(
    outcome_comparison_records
)

print(outcome_comparison)


print()
print("=" * 72)
print("24C-8B. FINAL INFERENTIAL MODEL SELECTION")
print("=" * 72)

inferential_model_selection = pl.DataFrame(
    model_selection_records
)

print(inferential_model_selection)


print()
print("=" * 72)
print("24C-8C. MODEL SELECTION VALIDATION")
print("=" * 72)

primary_outcome = (
    inferential_model_selection
    .filter(
        pl.col("outcome") == "major_obstacle"
    )
)

robustness_outcome = (
    inferential_model_selection
    .filter(
        pl.col("outcome") == "any_obstacle"
    )
)

exploratory_outcome = (
    inferential_model_selection
    .filter(
        pl.col("outcome") == "severe_obstacle"
    )
)

assert primary_outcome.height == len(diagnostic_specs)
assert robustness_outcome.height == len(diagnostic_specs)
assert exploratory_outcome.height == len(diagnostic_specs)

assert primary_outcome["inferential_use"].all()
assert robustness_outcome["inferential_use"].all()
assert not exploratory_outcome["inferential_use"].any()

print("Primary outcome validation: PASSED")
print("Robustness outcome validation: PASSED")
print("Severe outcome exclusion from inference: PASSED")


print()
print("=" * 72)
print("24C-8D. EXPORT FINAL MODEL SELECTION")
print("=" * 72)

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

processed_dir = (
    project_root / "data" / "processed"
)

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)

comparison_path = (
    processed_dir / "24c8_outcome_comparison.csv"
)

selection_path = (
    processed_dir / "24c8_inferential_model_selection.csv"
)

outcome_comparison.write_csv(
    comparison_path
)

inferential_model_selection.write_csv(
    selection_path
)

print("Saved:", comparison_path)
print("Saved:", selection_path)


print()
print("=" * 72)
print("24C-8 COMPLETED")
print("=" * 72)

print("Primary outcome: major_obstacle")
print("Robustness outcome: any_obstacle")
print("Exploratory outcome: severe_obstacle")
print("No regression models were re-estimated.")

print()
print("=" * 72)
print("24C-9. OUTCOME DATASET DIAGNOSTICS")
print("=" * 72)

print("data shape:", data.shape)
print("data_outcome shape:", data_outcome.shape)

print()
print("OUTCOME COLUMN AVAILABILITY")

for column in [
    "major_obstacle",
    "any_obstacle",
    "severe_obstacle",
]:
    print(
        f"{column}: "
        f"data={column in data.columns}, "
        f"data_outcome={column in data_outcome.columns}"
    )

print()
print("POTENTIAL OBSERVATION IDENTIFIERS")

for column in data.columns:
    if column in data_outcome.columns:
        if any(
            term in column.lower()
            for term in ["id", "case", "firm", "estab", "company"]
        ):
            print(column)

print()
print("=" * 72)
print("24C-9. DATASET AND OUTCOME COLUMN DIAGNOSTICS")
print("=" * 72)

print()
print("DATA COLUMNS")

for column in [
    "major_obstacle",
    "any_obstacle",
    "severe_obstacle",
    "model1_complete",
    "model2_complete",
]:
    print(
        column,
        "data:", column in data.columns,
        "data_outcome:", column in data_outcome.columns,
    )

print()
print("DATA DIMENSIONS")
print("data:", data.shape)
print("data_outcome:", data_outcome.shape)

print()
print("DATA OUTCOME SAMPLE")

print(
    data_outcome.select(
        [
            column
            for column in [
                "major_obstacle",
                "any_obstacle",
                "severe_obstacle",
            ]
            if column in data_outcome.columns
        ]
    ).head()
)

print()
print("=" * 72)
print("24C-9. ALTERNATIVE OUTCOME REGRESSION")
print("=" * 72)

from pathlib import Path

required_columns = [
    "any_obstacle",
    "major_obstacle",
    "model1_complete",
    "model2_complete",
    "strata",
    "idstd",
    "wmedian",
    "wweak",
]

"""missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from data: {missing_columns}. "
        "Ensure all outcome variables are included in the analysis dataset."
    )"""

print()
print("=" * 72)
print("24C-9. OUTCOME DATASET DIAGNOSTICS")
print("=" * 72)

print("data shape:", data.shape)
print("data_outcome shape:", data_outcome.shape)

print()
print("OUTCOME COLUMN AVAILABILITY")

for column in [
    "major_obstacle",
    "any_obstacle",
    "severe_obstacle",
]:
    print(
        f"{column}: "
        f"data={column in data.columns}, "
        f"data_outcome={column in data_outcome.columns}"
    )

print()
print("COMMON COLUMNS")

common_columns = [
    column
    for column in data.columns
    if column in data_outcome.columns
]

print(common_columns)

print()
print("POTENTIAL IDENTIFIER COLUMNS")

for column in common_columns:
    if any(
        term in column.lower()
        for term in ["id", "case", "firm", "estab", "company"]
    ):
        print(column)

print()
print("24C-9 DIAGNOSTICS COMPLETED")


if "coefficient_results" not in globals():
    raise ValueError(
        "The 24B coefficient_results object was not found. "
        "Run the original 24B analysis before continuing."
    )

major_coefficient_results = coefficient_results.clone()

design_median = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wmedian",
    singleton="scale",
)

design_weak = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wweak",
    singleton="scale",
)

sample_median = svy.Sample(
    data=data,
    design=design_median,
)

sample_weak = svy.Sample(
    data=data,
    design=design_weak,
)

print()
print("=" * 72)
print("24C-9. FINAL DATASET INSPECTION")
print("=" * 72)

print()
print("DATA")
print("Shape:", data.shape)
print("Obstacle columns:", [
    column
    for column in data.columns
    if "obstacle" in column.lower()
])

print()
print("DATA_OUTCOME")
print("Shape:", data_outcome.shape)
print("Obstacle columns:", [
    column
    for column in data_outcome.columns
    if "obstacle" in column.lower()
])

print()
print("OUTCOME VARIABLES IN DATA_OUTCOME")

available_outcomes = [
    column
    for column in [
        "major_obstacle",
        "any_obstacle",
        "severe_obstacle",
    ]
    if column in data_outcome.columns
]

if available_outcomes:
    print(
        data_outcome.select(available_outcomes).head(10)
    )
else:
    print("No expected outcome variables found.")

print()
print("COMMON IDENTIFIER CHECK")

for column in ["idstd"]:
    print(
        column,
        "in data:", column in data.columns,
        "in data_outcome:", column in data_outcome.columns,
    )

    if (
        column in data.columns
        and column in data_outcome.columns
    ):
        print(
            "data unique IDs:",
            data[column].n_unique(),
        )
        print(
            "data_outcome unique IDs:",
            data_outcome[column].n_unique(),
        )

print()
print("24C-9 DATASET INSPECTION COMPLETED")

print()
print("=" * 72)
print("24C-9. ALTERNATIVE OUTCOME REGRESSION")
print("=" * 72)

print()
print("24C-9A. VALIDATE OUTCOME IDENTIFIERS")

if data["idstd"].null_count() > 0:
    raise ValueError(
        "Null identifiers detected in the regression dataset."
    )

if data_outcome["idstd"].null_count() > 0:
    raise ValueError(
        "Null identifiers detected in the outcome dataset."
    )

if data["idstd"].n_unique() != data.height:
    raise ValueError(
        "Duplicate identifiers detected in the regression dataset."
    )

if data_outcome["idstd"].n_unique() != data_outcome.height:
    raise ValueError(
        "Duplicate identifiers detected in the outcome dataset."
    )

missing_outcomes = (
    data
    .select("idstd")
    .join(
        data_outcome.select("idstd"),
        on="idstd",
        how="anti",
    )
)

unexpected_outcomes = (
    data_outcome
    .select("idstd")
    .join(
        data.select("idstd"),
        on="idstd",
        how="anti",
    )
)

if missing_outcomes.height > 0:
    raise ValueError(
        "Some regression observations have no matching outcome ID."
    )

if unexpected_outcomes.height > 0:
    raise ValueError(
        "Some outcome observations have no matching regression ID."
    )

print("Regression IDs:", data.height)
print("Outcome IDs:", data_outcome.height)
print("Unique identifiers verified.")
print("Identifier matching verified.")

print()
print("24C-9B. MERGE ALTERNATIVE OUTCOME")

outcome_lookup = data_outcome.select(
    "idstd",
    "any_obstacle",
)

data_24c9 = data.join(
    outcome_lookup,
    on="idstd",
    how="left",
    validate="1:1",
)

if data_24c9.height != data.height:
    raise ValueError(
        "Unexpected observation count after outcome merge."
    )

print()
print("=" * 72)
print("24C-9. ALTERNATIVE OUTCOME MISSING VALUE DIAGNOSTICS")
print("=" * 72)

print()
print("SOURCE OUTCOME MISSING VALUES")

print(
    data_outcome.select(
        pl.len().alias("total_observations"),
        pl.col("any_obstacle").null_count().alias(
            "missing_any_obstacle"
        ),
        pl.col("major_obstacle").null_count().alias(
            "missing_major_obstacle"
        ),
        pl.col("severe_obstacle").null_count().alias(
            "missing_severe_obstacle"
        ),
    )
)

print()
print("OUTCOME DISTRIBUTION IN SOURCE DATA")

print(
    data_outcome.group_by("any_obstacle").agg(
        pl.len().alias("observations")
    ).sort("any_obstacle")
)

print()
print("MISSING OUTCOME BY MODEL SAMPLE")

missing_rows = []

for model_name, sample_flag in [
    ("model1_full", "model1_complete"),
    ("model1_common", "model2_complete"),
    ("model2_common", "model2_complete"),
]:
    selected = data_24c9.filter(
        pl.col(sample_flag) == True
    )

    missing_rows.append({
        "model": model_name,
        "total_sample": selected.height,
        "valid_outcome": selected.filter(
            pl.col("any_obstacle").is_not_null()
        ).height,
        "missing_outcome": selected.filter(
            pl.col("any_obstacle").is_null()
        ).height,
    })

missing_summary = pl.DataFrame(missing_rows)

print(missing_summary)

print()
print("MISSING OUTCOME BY MAJOR OUTCOME")

print(
    data_24c9.group_by(
        "major_obstacle",
        "any_obstacle",
    ).agg(
        pl.len().alias("observations")
    ).sort(
        ["major_obstacle", "any_obstacle"]
    )
)

print()
print("24C-9 MISSING VALUE DIAGNOSTICS COMPLETED")

print()
print("=" * 72)
print("24C-9. PREPARE ALTERNATIVE OUTCOME DATASET")
print("=" * 72)

outcome_lookup = data_outcome.select(
    "idstd",
    "any_obstacle",
)

data_24c9 = data.join(
    outcome_lookup,
    on="idstd",
    how="left",
    validate="1:1",
)

if data_24c9.height != data.height:
    raise ValueError("Row count changed after merging the alternative outcome.")

if data_24c9["idstd"].n_unique() != data_24c9.height:
    raise ValueError("Duplicate identifiers detected after merging.")

if data_24c9["any_obstacle"].null_count() != 122:
    raise ValueError("Unexpected number of missing alternative outcome values.")

print("Merged dataset shape:", data_24c9.shape)
print(
    "Missing any_obstacle values:",
    data_24c9["any_obstacle"].null_count(),
)

print()
print("ALTERNATIVE OUTCOME DISTRIBUTION")

print(
    data_24c9.group_by("any_obstacle")
    .agg(
        pl.len().alias("observations")
    )
    .sort("any_obstacle", nulls_last=True)
)

sample_diagnostics = pl.DataFrame(
    {
        "model": [
            "model1_full",
            "model1_common",
            "model2_common",
        ],
        "sample_flag": [
            "model1_complete",
            "model2_complete",
            "model2_complete",
        ],
    }
)

sample_checks = []

for row in sample_diagnostics.iter_rows(named=True):
    sample_flag = row["sample_flag"]

    sample = data_24c9.filter(
        pl.col(sample_flag)
    )

    sample_checks.append(
        {
            "model": row["model"],
            "total_sample": sample.height,
            "valid_outcome": sample.filter(
                pl.col("any_obstacle").is_not_null()
            ).height,
            "missing_outcome": sample["any_obstacle"].null_count(),
        }
    )

sample_check_results = pl.DataFrame(sample_checks)

print()
print("FINAL REGRESSION SAMPLE VALIDATION")
print(sample_check_results)

if sample_check_results["missing_outcome"].sum() != 0:
    raise ValueError(
        "Missing alternative outcome values detected in regression samples."
    )

print()
print("24C-9 DATASET PREPARATION COMPLETED")

invalid_outcomes = data_24c9.filter(
    ~pl.col("any_obstacle").is_in([0, 1])
)

if invalid_outcomes.height > 0:
    raise ValueError(
        "Invalid any_obstacle values detected. "
        "Expected binary values 0 and 1."
    )

print("Merged dataset shape:", data_24c9.shape)
print("Alternative outcome successfully merged.")
print("All outcome values are valid.")

print()
print("ALTERNATIVE OUTCOME DISTRIBUTION")

print(
    data_24c9.group_by("any_obstacle").agg(
        pl.len().alias("observations")
    ).sort("any_obstacle")
)

print()
print("=" * 78)
print("24C-10J. ESTIMATE ACTUAL ALTERNATIVE OUTCOME MODELS")
print("=" * 78)

def fit_three_any_models(sample_obj, weight_name):
    fitted_models = {}

    fitted_models[f"model1_full_{weight_name}"] = (
        sample_obj.glm.fit(
            y="any_obstacle",
            x=model1_predictors,
            family="binomial",
            link="logit",
            where=(svy.col("model1_complete") == True),
        )
    )

    fitted_models[f"model1_common_{weight_name}"] = (
        sample_obj.glm.fit(
            y="any_obstacle",
            x=model1_predictors,
            family="binomial",
            link="logit",
            where=(svy.col("model2_complete") == True),
        )
    )

    fitted_models[f"model2_common_{weight_name}"] = (
        sample_obj.glm.fit(
            y="any_obstacle",
            x=model2_predictors,
            family="binomial",
            link="logit",
            where=(svy.col("model2_complete") == True),
        )
    )

    return fitted_models


print()
print("REBUILD ALTERNATIVE OUTCOME SURVEY SAMPLES")

sample_median_24c10 = svy.Sample(
    data=data_24c9,
    design=design_median,
)

sample_weak_24c10 = svy.Sample(
    data=data_24c9,
    design=design_weak,
)

sample_flags_24c10 = [
    ("model1_full", "model1_complete", 2632),
    ("model1_common", "model2_complete", 2528),
    ("model2_common", "model2_complete", 2528),
]

for sample_name, flag_name, expected_n in sample_flags_24c10:
    sample_data = data_24c9.filter(
        pl.col(flag_name) == True
    )

    actual_n = sample_data.height

    missing_outcome = sample_data.filter(
        pl.col("any_obstacle").is_null()
    ).height

    print(
        f"{sample_name}: "
        f"N={actual_n}, "
        f"missing_any_obstacle={missing_outcome}"
    )

    if actual_n != expected_n:
        raise ValueError(
            f"Unexpected sample size for {sample_name}: "
            f"expected {expected_n}, received {actual_n}."
        )

    if missing_outcome > 0:
        raise ValueError(
            f"Missing any_obstacle values remain in {sample_name}."
        )

print()
print("All regression samples have complete alternative outcomes.")


print()
print("ESTIMATE ALTERNATIVE OUTCOME MODELS")

any_models_median_24c10 = fit_three_any_models(
    sample_median_24c10,
    "wmedian",
)

any_models_weak_24c10 = fit_three_any_models(
    sample_weak_24c10,
    "wweak",
)

any_models_24c10 = {
    **any_models_median_24c10,
    **any_models_weak_24c10,
}

expected_any_models_24c10 = {
    "model1_full_wmedian",
    "model1_common_wmedian",
    "model2_common_wmedian",
    "model1_full_wweak",
    "model1_common_wweak",
    "model2_common_wweak",
}

if set(any_models_24c10.keys()) != expected_any_models_24c10:
    raise ValueError(
        "The alternative outcome model names do not match "
        "the expected six models."
    )

print(
    "Alternative outcome models estimated:",
    len(any_models_24c10),
)

print()
print("24C-10J COMPLETED")


print()
print("=" * 78)
print("24C-10K. EXTRACT ALTERNATIVE OUTCOME COEFFICIENTS")
print("=" * 78)

any_coefficient_rows_24c10 = []

for model_name, fitted_model in any_models_24c10.items():
    fitted = fitted_model.fitted

    for coefficient in fitted.coefs:
        beta = float(coefficient.est)
        se = float(coefficient.se)
        ci_low = float(coefficient.lci)
        ci_high = float(coefficient.uci)
        p_value = float(coefficient.wald.p_value)

        any_coefficient_rows_24c10.append({
            "model": model_name,
            "term": coefficient.term,
            "beta": beta,
            "se": se,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(ci_low),
            "or_ci_high": math.exp(ci_high),
        })

any_coefficient_results_24c10 = pl.DataFrame(
    any_coefficient_rows_24c10
)

if any_coefficient_results_24c10.is_empty():
    raise ValueError(
        "Alternative outcome coefficient extraction returned no rows."
    )

required_any_columns_24c10 = [
    "model",
    "term",
    "beta",
    "se",
    "p_value",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "or_ci_low",
    "or_ci_high",
]

missing_any_columns_24c10 = [
    column
    for column in required_any_columns_24c10
    if column not in any_coefficient_results_24c10.columns
]

if missing_any_columns_24c10:
    raise ValueError(
        "Missing alternative coefficient columns: "
        f"{missing_any_columns_24c10}"
    )

finite_any_check_24c10 = any_coefficient_results_24c10.filter(
    ~(
        pl.col("beta").is_finite()
        & pl.col("se").is_finite()
        & pl.col("p_value").is_finite()
        & pl.col("ci_low").is_finite()
        & pl.col("ci_high").is_finite()
        & pl.col("odds_ratio").is_finite()
        & pl.col("or_ci_low").is_finite()
        & pl.col("or_ci_high").is_finite()
    )
)

if finite_any_check_24c10.height > 0:
    raise ValueError(
        "Non-finite values were found in alternative outcome coefficients."
    )

duplicate_any_terms_24c10 = (
    any_coefficient_results_24c10
    .group_by(["model", "term"])
    .len()
    .filter(pl.col("len") > 1)
)

if duplicate_any_terms_24c10.height > 0:
    raise ValueError(
        "Duplicate model-term combinations found in alternative coefficients."
    )

print(
    "Alternative coefficient rows:",
    any_coefficient_results_24c10.height,
)

print(
    "Alternative models represented:",
    any_coefficient_results_24c10["model"].n_unique(),
)

print()
print("ALTERNATIVE OUTCOME COEFFICIENT SUMMARY")

print(
    any_coefficient_results_24c10.select(
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    )
)


print()
print("=" * 78)
print("24C-10L. VALIDATE PRIMARY OUTCOME COEFFICIENT SOURCE")
print("=" * 78)

major_coefficient_results = coefficient_results.clone()

required_major_columns_24c10 = [
    "model",
    "term",
    "beta",
    "se",
    "p_value",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "or_ci_low",
    "or_ci_high",
]

missing_major_columns_24c10 = [
    column
    for column in required_major_columns_24c10
    if column not in major_coefficient_results.columns
]

if missing_major_columns_24c10:
    raise ValueError(
        "Missing primary outcome coefficient columns: "
        f"{missing_major_columns_24c10}"
    )

expected_major_models_24c10 = expected_any_models_24c10

actual_major_models_24c10 = set(
    major_coefficient_results["model"].unique().to_list()
)

if actual_major_models_24c10 != expected_major_models_24c10:
    raise ValueError(
        "Primary outcome models do not match the expected six models. "
        f"Found: {sorted(actual_major_models_24c10)}"
    )

duplicate_major_terms_24c10 = (
    major_coefficient_results
    .group_by(["model", "term"])
    .len()
    .filter(pl.col("len") > 1)
)

if duplicate_major_terms_24c10.height > 0:
    raise ValueError(
        "Duplicate model-term combinations found in primary coefficients."
    )

print(
    "Primary outcome coefficient rows:",
    major_coefficient_results.height,
)

print(
    "Primary outcome models represented:",
    major_coefficient_results["model"].n_unique(),
)

print("Primary outcome source: coefficient_results")
print("Primary outcome: major_obstacle")
print("Alternative outcome: any_obstacle")


print()
print("=" * 78)
print("24C-10M. MATCH PRIMARY AND ALTERNATIVE COEFFICIENTS")
print("=" * 78)

major_keys_24c10 = major_coefficient_results.select(
    "model",
    "term",
)

any_keys_24c10 = any_coefficient_results_24c10.select(
    "model",
    "term",
)

major_unmatched_24c10 = major_keys_24c10.join(
    any_keys_24c10,
    on=["model", "term"],
    how="anti",
)

any_unmatched_24c10 = any_keys_24c10.join(
    major_keys_24c10,
    on=["model", "term"],
    how="anti",
)

if major_unmatched_24c10.height > 0:
    print("Terms present only in primary outcome:")
    print(major_unmatched_24c10)

if any_unmatched_24c10.height > 0:
    print("Terms present only in alternative outcome:")
    print(any_unmatched_24c10)

if (
    major_unmatched_24c10.height > 0
    or any_unmatched_24c10.height > 0
):
    raise ValueError(
        "Primary and alternative outcome models do not have "
        "identical model-term combinations. Comparison stopped."
    )

outcome_comparison_24c10 = major_coefficient_results.select(
    "model",
    "term",
    pl.col("beta").alias("beta_major"),
    pl.col("se").alias("se_major"),
    pl.col("p_value").alias("p_value_major"),
    pl.col("ci_low").alias("ci_low_major"),
    pl.col("ci_high").alias("ci_high_major"),
    pl.col("odds_ratio").alias("or_major"),
    pl.col("or_ci_low").alias("or_ci_low_major"),
    pl.col("or_ci_high").alias("or_ci_high_major"),
).join(
    any_coefficient_results_24c10.select(
        "model",
        "term",
        pl.col("beta").alias("beta_any"),
        pl.col("se").alias("se_any"),
        pl.col("p_value").alias("p_value_any"),
        pl.col("ci_low").alias("ci_low_any"),
        pl.col("ci_high").alias("ci_high_any"),
        pl.col("odds_ratio").alias("or_any"),
        pl.col("or_ci_low").alias("or_ci_low_any"),
        pl.col("or_ci_high").alias("or_ci_high_any"),
    ),
    on=["model", "term"],
    how="inner",
    validate="1:1",
)

if outcome_comparison_24c10.height != major_coefficient_results.height:
    raise ValueError(
        "The matched comparison row count does not equal "
        "the primary coefficient row count."
    )

print(
    "Matched model-term combinations:",
    outcome_comparison_24c10.height,
)


print()
print("=" * 78)
print("24C-10N. CALCULATE OUTCOME COEFFICIENT DIFFERENCES")
print("=" * 78)

outcome_comparison_24c10 = outcome_comparison_24c10.with_columns(
    (pl.col("beta_any") - pl.col("beta_major")).alias(
        "beta_difference"
    ),
    (pl.col("se_any") - pl.col("se_major")).alias(
        "se_difference"
    ),
    (pl.col("p_value_any") - pl.col("p_value_major")).alias(
        "p_value_difference"
    ),
    (pl.col("or_any") - pl.col("or_major")).alias(
        "or_difference"
    ),
    (
        (pl.col("p_value_major") < 0.05)
        .cast(pl.Int8)
        .alias("significant_major")
    ),
    (
        (pl.col("p_value_any") < 0.05)
        .cast(pl.Int8)
        .alias("significant_any")
    ),
).with_columns(
    (
        pl.col("significant_major")
        != pl.col("significant_any")
    ).alias("significance_status_changed")
)

print()
print("OUTCOME COEFFICIENT COMPARISON")

print(
    outcome_comparison_24c10.select(
        "model",
        "term",
        "beta_major",
        "beta_any",
        "beta_difference",
        "or_major",
        "or_any",
        "or_difference",
        "p_value_major",
        "p_value_any",
        "significance_status_changed",
    )
)

print()
print("SIGNIFICANCE STATUS CHANGES")

significance_changes_24c10 = (
    outcome_comparison_24c10
    .filter(pl.col("significance_status_changed"))
)

print(
    "Number of changed significance statuses:",
    significance_changes_24c10.height,
)

if significance_changes_24c10.height > 0:
    print(significance_changes_24c10.select(
        "model",
        "term",
        "p_value_major",
        "p_value_any",
        "significant_major",
        "significant_any",
    ))


print()
print("=" * 78)
print("24C-10O. SAVE ALTERNATIVE OUTCOME ROBUSTNESS RESULTS")
print("=" * 78)

outcome_robustness_dir_24c10 = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "outcome_robustness"
)

outcome_robustness_dir_24c10.mkdir(
    parents=True,
    exist_ok=True,
)

any_coefficient_results_24c10.write_csv(
    outcome_robustness_dir_24c10
    / "24c10_any_obstacle_regression.csv"
)

outcome_comparison_24c10.write_csv(
    outcome_robustness_dir_24c10
    / "24c10_major_vs_any_obstacle_comparison.csv"
)

significance_changes_24c10.write_csv(
    outcome_robustness_dir_24c10
    / "24c10_significance_status_changes.csv"
)

major_coefficient_results.write_csv(
    outcome_robustness_dir_24c10
    / "24c10_major_obstacle_primary_coefficients.csv"
)

print()
print("FILES SAVED")

print(
    outcome_robustness_dir_24c10
    / "24c10_any_obstacle_regression.csv"
)

print(
    outcome_robustness_dir_24c10
    / "24c10_major_vs_any_obstacle_comparison.csv"
)

print(
    outcome_robustness_dir_24c10
    / "24c10_significance_status_changes.csv"
)

print(
    outcome_robustness_dir_24c10
    / "24c10_major_obstacle_primary_coefficients.csv"
)

print()
print("=" * 78)
print("24C-10 OUTCOME ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 78)


print()
print("=" * 78)
print("24C-11. ROBUSTNESS INTERPRETATION AND STABILITY ASSESSMENT")
print("=" * 78)

expected_model_count_24c11 = 6

robustness_dir_24c11 = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "outcome_robustness"
)

robustness_dir_24c11.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("24C-11A. PREPARE COEFFICIENT DATA")

major_long_24c11 = major_coefficient_results.select(
    "model",
    "term",
    "beta",
    "se",
    "p_value",
    "ci_low",
    "ci_high",
).with_columns(
    pl.lit("major_obstacle").alias("outcome")
)

any_long_24c11 = any_coefficient_results_24c10.select(
    "model",
    "term",
    "beta",
    "se",
    "p_value",
    "ci_low",
    "ci_high",
).with_columns(
    pl.lit("any_obstacle").alias("outcome")
)

robustness_long_24c11 = pl.concat(
    [
        major_long_24c11,
        any_long_24c11,
    ],
    how="vertical",
)

if robustness_long_24c11.is_empty():
    raise ValueError(
        "No coefficient observations are available for stability assessment."
    )

print(
    "Total coefficient observations:",
    robustness_long_24c11.height,
)

print(
    "Outcomes represented:",
    robustness_long_24c11["outcome"].n_unique(),
)

print(
    "Models represented:",
    robustness_long_24c11["model"].n_unique(),
)


print()
print("24C-11B. ASSESS COEFFICIENT DIRECTION AND SIGNIFICANCE")

robustness_long_24c11 = robustness_long_24c11.with_columns(
    pl.when(pl.col("beta") > 0)
    .then(pl.lit("positive"))
    .when(pl.col("beta") < 0)
    .then(pl.lit("negative"))
    .otherwise(pl.lit("zero"))
    .alias("coefficient_direction"),

    (pl.col("p_value") < 0.05)
    .alias("significant_5pct"),

    (pl.col("ci_low") > 0)
    .alias("ci_entirely_positive"),

    (pl.col("ci_high") < 0)
    .alias("ci_entirely_negative"),

    (
        (pl.col("ci_low") <= 0)
        & (pl.col("ci_high") >= 0)
    ).alias("ci_includes_zero"),
)

direction_summary_24c11 = (
    robustness_long_24c11
    .group_by(["outcome", "term"])
    .agg(
        pl.len().alias("n_models"),

        (pl.col("coefficient_direction") == "positive")
        .sum()
        .alias("positive_count"),

        (pl.col("coefficient_direction") == "negative")
        .sum()
        .alias("negative_count"),

        (pl.col("coefficient_direction") == "zero")
        .sum()
        .alias("zero_count"),

        pl.col("beta").mean().alias("mean_beta"),
        pl.col("beta").min().alias("min_beta"),
        pl.col("beta").max().alias("max_beta"),

        pl.col("p_value").min().alias("min_p_value"),
        pl.col("p_value").max().alias("max_p_value"),

        pl.col("significant_5pct")
        .sum()
        .alias("significant_model_count"),

        pl.col("ci_entirely_positive")
        .sum()
        .alias("positive_ci_count"),

        pl.col("ci_entirely_negative")
        .sum()
        .alias("negative_ci_count"),

        pl.col("ci_includes_zero")
        .sum()
        .alias("ci_includes_zero_count"),
    )
    .with_columns(
        pl.when(
            pl.col("positive_count") == pl.col("n_models")
        )
        .then(pl.lit("consistently_positive"))
        .when(
            pl.col("negative_count") == pl.col("n_models")
        )
        .then(pl.lit("consistently_negative"))
        .when(
            pl.col("zero_count") == pl.col("n_models")
        )
        .then(pl.lit("consistently_zero"))
        .otherwise(pl.lit("mixed_direction"))
        .alias("direction_stability"),

        (
            pl.col("significant_model_count")
            == pl.col("n_models")
        ).alias("significant_in_all_available_models"),

        (
            pl.col("n_models")
            == expected_model_count_24c11
        ).alias("present_in_all_six_models"),
    )
    .sort(["outcome", "term"])
)

print()
print("DIRECTION AND SIGNIFICANCE SUMMARY")

print(direction_summary_24c11)


print()
print("24C-11C. IDENTIFY DIRECTIONALLY STABLE COEFFICIENTS")

directionally_stable_24c11 = direction_summary_24c11.filter(
    pl.col("direction_stability").is_in(
        [
            "consistently_positive",
            "consistently_negative",
        ]
    )
)

print(
    "Directionally stable outcome-term combinations:",
    directionally_stable_24c11.height,
)

print(directionally_stable_24c11.select(
    "outcome",
    "term",
    "n_models",
    "direction_stability",
    "mean_beta",
    "min_beta",
    "max_beta",
    "significant_model_count",
    "significant_in_all_available_models",
    "present_in_all_six_models",
))


print()
print("24C-11D. IDENTIFY CONSISTENTLY SIGNIFICANT COEFFICIENTS")

consistently_significant_24c11 = direction_summary_24c11.filter(
    pl.col("significant_in_all_available_models")
    & pl.col("present_in_all_six_models")
)

print(
    "Terms significant in all six models:",
    consistently_significant_24c11.height,
)

if consistently_significant_24c11.height > 0:
    print(consistently_significant_24c11.select(
        "outcome",
        "term",
        "direction_stability",
        "mean_beta",
        "min_p_value",
        "max_p_value",
        "positive_ci_count",
        "negative_ci_count",
    ))
else:
    print(
        "No coefficient is significant in all six specifications "
        "for its outcome."
    )


print()
print("24C-11E. COMPARE COEFFICIENT DIRECTIONS BETWEEN OUTCOMES")

outcome_direction_comparison_24c11 = (
    outcome_comparison_24c10
    .select(
        "model",
        "term",
        "beta_major",
        "beta_any",
        "p_value_major",
        "p_value_any",
        "or_major",
        "or_any",
    )
    .with_columns(
        pl.when(
            (pl.col("beta_major") > 0)
            & (pl.col("beta_any") > 0)
        )
        .then(pl.lit("positive_both"))
        .when(
            (pl.col("beta_major") < 0)
            & (pl.col("beta_any") < 0)
        )
        .then(pl.lit("negative_both"))
        .when(
            (pl.col("beta_major") > 0)
            & (pl.col("beta_any") < 0)
        )
        .then(pl.lit("positive_to_negative"))
        .when(
            (pl.col("beta_major") < 0)
            & (pl.col("beta_any") > 0)
        )
        .then(pl.lit("negative_to_positive"))
        .otherwise(pl.lit("zero_involved"))
        .alias("direction_comparison"),

        (
            (pl.col("p_value_major") < 0.05)
            != (pl.col("p_value_any") < 0.05)
        ).alias("significance_changed"),
    )
)

direction_change_summary_24c11 = (
    outcome_direction_comparison_24c11
    .group_by("direction_comparison")
    .agg(
        pl.len().alias("coefficient_count")
    )
    .sort("direction_comparison")
)

print("DIRECTION COMPARISON COUNTS")
print(direction_change_summary_24c11)

direction_reversals_24c11 = (
    outcome_direction_comparison_24c11
    .filter(
        pl.col("direction_comparison").is_in(
            [
                "positive_to_negative",
                "negative_to_positive",
            ]
        )
    )
)

print()
print(
    "Coefficient direction reversals between outcomes:",
    direction_reversals_24c11.height,
)

if direction_reversals_24c11.height > 0:
    print(direction_reversals_24c11)


print()
print("24C-11F. ASSESS SIGNIFICANCE STABILITY BY MODEL")

model_stability_24c11 = (
    robustness_long_24c11
    .group_by(["outcome", "model"])
    .agg(
        pl.len().alias("coefficient_count"),

        pl.col("significant_5pct")
        .sum()
        .alias("significant_count"),

        pl.col("ci_includes_zero")
        .sum()
        .alias("ci_includes_zero_count"),

        (
            pl.col("coefficient_direction") == "positive"
        )
        .sum()
        .alias("positive_count"),

        (
            pl.col("coefficient_direction") == "negative"
        )
        .sum()
        .alias("negative_count"),
    )
    .with_columns(
        (
            pl.col("significant_count")
            / pl.col("coefficient_count")
        ).alias("significant_share"),

        (
            pl.col("ci_includes_zero_count")
            / pl.col("coefficient_count")
        ).alias("ci_includes_zero_share"),
    )
    .sort(["outcome", "model"])
)

print("MODEL-LEVEL STABILITY SUMMARY")
print(model_stability_24c11)


print()
print("24C-11G. ASSESS WEIGHT SENSITIVITY")

weight_stability_24c11 = (
    robustness_long_24c11
    .with_columns(
        pl.when(pl.col("model").str.ends_with("wmedian"))
        .then(pl.lit("wmedian"))
        .when(pl.col("model").str.ends_with("wweak"))
        .then(pl.lit("wweak"))
        .otherwise(pl.lit("unknown"))
        .alias("weight")
    )
    .group_by(["outcome", "weight"])
    .agg(
        pl.len().alias("coefficient_count"),

        pl.col("significant_5pct")
        .sum()
        .alias("significant_count"),

        (
            pl.col("coefficient_direction") == "positive"
        )
        .sum()
        .alias("positive_count"),

        (
            pl.col("coefficient_direction") == "negative"
        )
        .sum()
        .alias("negative_count"),
    )
    .with_columns(
        (
            pl.col("significant_count")
            / pl.col("coefficient_count")
        ).alias("significant_share")
    )
    .sort(["outcome", "weight"])
)

print("WEIGHT-LEVEL STABILITY SUMMARY")
print(weight_stability_24c11)


print()
print("24C-11H. SAVE STABILITY ASSESSMENT RESULTS")

direction_summary_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_coefficient_direction_and_significance_stability.csv"
)

directionally_stable_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_directionally_stable_coefficients.csv"
)

consistently_significant_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_consistently_significant_coefficients.csv"
)

outcome_direction_comparison_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_outcome_direction_comparison.csv"
)

direction_reversals_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_coefficient_direction_reversals.csv"
)

model_stability_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_model_level_stability.csv"
)

weight_stability_24c11.write_csv(
    robustness_dir_24c11
    / "24c11_weight_level_stability.csv"
)

print()
print("SAVED FILES")

print(
    robustness_dir_24c11
    / "24c11_coefficient_direction_and_significance_stability.csv"
)

print(
    robustness_dir_24c11
    / "24c11_directionally_stable_coefficients.csv"
)

print(
    robustness_dir_24c11
    / "24c11_consistently_significant_coefficients.csv"
)

print(
    robustness_dir_24c11
    / "24c11_outcome_direction_comparison.csv"
)

print(
    robustness_dir_24c11
    / "24c11_coefficient_direction_reversals.csv"
)

print(
    robustness_dir_24c11
    / "24c11_model_level_stability.csv"
)

print(
    robustness_dir_24c11
    / "24c11_weight_level_stability.csv"
)

print()
print("=" * 78)
print("24C-11 ROBUSTNESS STABILITY ASSESSMENT COMPLETED")
print("=" * 78)


print()
print("=" * 78)
print("24C-12. CATEGORICAL REFERENCE, SPARSE CELLS, AND SEPARATION DIAGNOSTICS")
print("=" * 78)

diagnostics_dir_24c12 = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "outcome_robustness"
)

diagnostics_dir_24c12.mkdir(
    parents=True,
    exist_ok=True,
)

categorical_variables_24c12 = ["a2", "a4a"]

sample_definitions_24c12 = [
    ("model1_full", "model1_complete", 2632),
    ("model1_common", "model2_complete", 2528),
    ("model2_common", "model2_complete", 2528),
]

outcome_variables_24c12 = [
    "major_obstacle",
    "any_obstacle",
]

print()
print("24C-12A. VALIDATE DIAGNOSTIC DATA")

required_columns_24c12 = [
    "idstd",
    "a2",
    "a4a",
    "model1_complete",
    "model2_complete",
    "major_obstacle",
    "any_obstacle",
]

missing_columns_24c12 = [
    column
    for column in required_columns_24c12
    if column not in data_24c9.columns
]

if missing_columns_24c12:
    raise ValueError(
        "Missing columns in data_24c9: "
        f"{missing_columns_24c12}"
    )

if data_24c9["idstd"].n_unique() != data_24c9.height:
    raise ValueError(
        "Duplicate idstd values found in data_24c9."
    )

for outcome_name in outcome_variables_24c12:
    invalid_outcomes = data_24c9.filter(
        pl.col(outcome_name).is_not_null()
        & ~pl.col(outcome_name).is_in([0, 1])
    ).height

    if invalid_outcomes > 0:
        raise ValueError(
            f"Invalid values found in {outcome_name}."
        )

print("Diagnostic dataset:", data_24c9.shape)
print("Categorical variables:", categorical_variables_24c12)
print("Outcome variables:", outcome_variables_24c12)


print()
print("24C-12B. CHECK CATEGORY FREQUENCIES AND OUTCOME CELLS")

category_cell_rows_24c12 = []
category_missing_rows_24c12 = []

for sample_name, flag_name, expected_n in sample_definitions_24c12:
    sample_data_24c12 = data_24c9.filter(
        pl.col(flag_name) == True
    )

    if sample_data_24c12.height != expected_n:
        raise ValueError(
            f"Unexpected sample size for {sample_name}: "
            f"{sample_data_24c12.height} instead of {expected_n}."
        )

    for category_name in categorical_variables_24c12:
        missing_category_count = sample_data_24c12.filter(
            pl.col(category_name).is_null()
        ).height

        category_missing_rows_24c12.append({
            "sample": sample_name,
            "category_variable": category_name,
            "missing_category_count": missing_category_count,
            "sample_size": sample_data_24c12.height,
        })

        category_levels = (
            sample_data_24c12
            .select(category_name)
            .drop_nulls()
            .unique()
            .sort(category_name)
            .to_series()
            .to_list()
        )

        for category_value in category_levels:
            category_data_24c12 = sample_data_24c12.filter(
                pl.col(category_name) == category_value
            )

            category_n = category_data_24c12.height

            for outcome_name in outcome_variables_24c12:
                event_count = category_data_24c12.filter(
                    pl.col(outcome_name) == 1
                ).height

                non_event_count = category_data_24c12.filter(
                    pl.col(outcome_name) == 0
                ).height

                category_cell_rows_24c12.append({
                    "sample": sample_name,
                    "category_variable": category_name,
                    "category_value": str(category_value),
                    "outcome": outcome_name,
                    "category_n": category_n,
                    "event_count": event_count,
                    "non_event_count": non_event_count,
                    "event_rate": (
                        event_count / category_n
                        if category_n > 0
                        else None
                    ),
                    "zero_events": event_count == 0,
                    "zero_non_events": non_event_count == 0,
                    "sparse_category_n_lt_10": category_n < 10,
                    "sparse_category_n_lt_20": category_n < 20,
                })

category_cell_diagnostics_24c12 = pl.DataFrame(
    category_cell_rows_24c12
)

category_missing_diagnostics_24c12 = pl.DataFrame(
    category_missing_rows_24c12
)

if category_cell_diagnostics_24c12.is_empty():
    raise ValueError(
        "No categorical cell diagnostics were generated."
    )

print("CATEGORY CELL DIAGNOSTICS")
print(category_cell_diagnostics_24c12)

print()
print("MISSING CATEGORY DIAGNOSTICS")
print(category_missing_diagnostics_24c12)


print()
print("24C-12C. IDENTIFY ZERO-EVENT AND SPARSE CATEGORIES")

zero_event_categories_24c12 = (
    category_cell_diagnostics_24c12
    .filter(
        pl.col("zero_events")
        | pl.col("zero_non_events")
    )
)

sparse_categories_24c12 = (
    category_cell_diagnostics_24c12
    .filter(
        pl.col("sparse_category_n_lt_20")
    )
)

print(
    "Zero-event or zero-non-event cells:",
    zero_event_categories_24c12.height,
)

if zero_event_categories_24c12.height > 0:
    print(zero_event_categories_24c12)

print()
print(
    "Category-outcome cells with fewer than 20 observations:",
    sparse_categories_24c12.height,
)

if sparse_categories_24c12.height > 0:
    print(sparse_categories_24c12)


print()
print()
print("24C-12D. IDENTIFY CANDIDATE REFERENCE CATEGORIES")

reference_candidate_rows_24c12 = []

coefficient_sources_24c12 = [
    (
        "major_obstacle",
        major_coefficient_results,
    ),
    (
        "any_obstacle",
        any_coefficient_results_24c10,
    ),
]

for outcome_name, coefficient_table in coefficient_sources_24c12:
    for model_name in coefficient_table["model"].unique().to_list():

        if model_name.startswith("model1_full_"):
            sample_name = "model1_full"
            flag_name = "model1_complete"
        else:
            sample_name = (
                "model1_common"
                if model_name.startswith("model1_common_")
                else "model2_common"
            )
            flag_name = "model2_complete"

        model_sample_24c12 = data_24c9.filter(
            pl.col(flag_name) == True
        )

        for category_name in categorical_variables_24c12:
            observed_levels = (
                model_sample_24c12
                .select(category_name)
                .drop_nulls()
                .unique()
                .sort(category_name)
                .to_series()
                .to_list()
            )

            model_terms = (
                coefficient_table
                .filter(pl.col("model") == model_name)
                ["term"]
                .to_list()
            )

            represented_levels = []

            for category_value in observed_levels:
                expected_term = (
                    f"{category_name}_{category_value}"
                )

                if expected_term in model_terms:
                    represented_levels.append(
                        str(category_value)
                    )

            omitted_levels = [
                str(category_value)
                for category_value in observed_levels
                if str(category_value) not in represented_levels
            ]

            if len(omitted_levels) == 1:
                candidate_status = "single_omitted_level_candidate"
            elif len(omitted_levels) == 0:
                candidate_status = "no_omitted_level_detected"
            else:
                candidate_status = (
                    "multiple_omitted_levels_review_required"
                )

            reference_candidate_rows_24c12.append({
                "outcome": outcome_name,
                "model": model_name,
                "sample": sample_name,
                "category_variable": category_name,
                "observed_level_count": len(observed_levels),
                "represented_dummy_count": len(represented_levels),
                "omitted_level_count": len(omitted_levels),
                "omitted_levels": ", ".join(omitted_levels),
                "candidate_status": candidate_status,
            })

reference_candidate_diagnostics_24c12 = pl.DataFrame(
    reference_candidate_rows_24c12
)

print("CANDIDATE REFERENCE CATEGORY DIAGNOSTICS")
print(reference_candidate_diagnostics_24c12)

print()
print(
    "Note: omitted levels are candidates, not confirmed reference "
    "categories. Omission can also result from model coding or "
    "non-estimable categories."
)


print()
print("24C-12E. CHECK EXTREME COEFFICIENTS AND STANDARD ERRORS")

coefficient_diagnostics_24c12 = (
    pl.concat(
        [
            major_coefficient_results.with_columns(
                pl.lit("major_obstacle").alias("outcome")
            ),
            any_coefficient_results_24c10.with_columns(
                pl.lit("any_obstacle").alias("outcome")
            ),
        ],
        how="vertical",
    )
    .with_columns(
        (pl.col("ci_high") - pl.col("ci_low"))
        .alias("ci_width"),

        (pl.col("beta").abs() >= 5)
        .alias("extreme_absolute_beta"),

        (pl.col("se") >= 2)
        .alias("large_standard_error"),

        (
            (pl.col("ci_high") - pl.col("ci_low")) >= 10
        ).alias("wide_confidence_interval"),

        (pl.col("odds_ratio") >= 100)
        .alias("very_large_odds_ratio"),

        (
            (pl.col("odds_ratio") > 0)
            & (pl.col("odds_ratio") <= 0.01)
        ).alias("very_small_odds_ratio"),
    )
    .with_columns(
        (
            pl.col("extreme_absolute_beta")
            | pl.col("large_standard_error")
            | pl.col("wide_confidence_interval")
            | pl.col("very_large_odds_ratio")
            | pl.col("very_small_odds_ratio")
        ).alias("requires_manual_review")
    )
    .sort(
        ["requires_manual_review", "outcome", "model"],
        descending=[True, False, False],
    )
)

extreme_coefficients_24c12 = (
    coefficient_diagnostics_24c12
    .filter(pl.col("requires_manual_review"))
)

print(
    "Total coefficient estimates:",
    coefficient_diagnostics_24c12.height,
)

print(
    "Coefficient estimates flagged for manual review:",
    extreme_coefficients_24c12.height,
)

if extreme_coefficients_24c12.height > 0:
    print(extreme_coefficients_24c12.select(
        "outcome",
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
        "ci_width",
        "odds_ratio",
        "extreme_absolute_beta",
        "large_standard_error",
        "wide_confidence_interval",
        "very_large_odds_ratio",
        "very_small_odds_ratio",
    ))


print()
print("24C-12F. SCREEN FOR POSSIBLE SEPARATION")

separation_screen_rows_24c12 = []

for sample_name, flag_name, expected_n in sample_definitions_24c12:
    sample_data_24c12 = data_24c9.filter(
        pl.col(flag_name) == True
    )

    for category_name in categorical_variables_24c12:
        category_levels = (
            sample_data_24c12
            .select(category_name)
            .drop_nulls()
            .unique()
            .sort(category_name)
            .to_series()
            .to_list()
        )

        for outcome_name in outcome_variables_24c12:
            for category_value in category_levels:
                cell_data_24c12 = sample_data_24c12.filter(
                    pl.col(category_name) == category_value
                )

                cell_n = cell_data_24c12.height

                event_count = cell_data_24c12.filter(
                    pl.col(outcome_name) == 1
                ).height

                non_event_count = cell_data_24c12.filter(
                    pl.col(outcome_name) == 0
                ).height

                if event_count == 0 or non_event_count == 0:
                    separation_status = "possible_complete_separation"
                elif event_count < 5 or non_event_count < 5:
                    separation_status = "possible_sparse_cell_instability"
                else:
                    separation_status = "no_obvious_single_category_separation"

                separation_screen_rows_24c12.append({
                    "sample": sample_name,
                    "category_variable": category_name,
                    "category_value": str(category_value),
                    "outcome": outcome_name,
                    "category_n": cell_n,
                    "event_count": event_count,
                    "non_event_count": non_event_count,
                    "separation_screen": separation_status,
                })

separation_screen_24c12 = pl.DataFrame(
    separation_screen_rows_24c12
)

possible_separation_24c12 = (
    separation_screen_24c12
    .filter(
        pl.col("separation_screen")
        != "no_obvious_single_category_separation"
    )
)

print(
    "Potentially problematic category-outcome cells:",
    possible_separation_24c12.height,
)

if possible_separation_24c12.height > 0:
    print(possible_separation_24c12)


print()
print("24C-12G. LINK EXTREME COEFFICIENTS TO CATEGORY DIAGNOSTICS")

category_link_rows_24c12 = []

for row in extreme_coefficients_24c12.iter_rows(named=True):
    term = row["term"]

    matched_category = None
    matched_level = None

    for category_name in categorical_variables_24c12:
        prefix = f"{category_name}_"

        if term.startswith(prefix):
            matched_category = category_name
            matched_level = term[len(prefix):]
            break

    if matched_category is None:
        continue

    model_name = row["model"]

    if model_name.startswith("model1_full_"):
        sample_name = "model1_full"
        flag_name = "model1_complete"
    else:
        sample_name = (
            "model1_common"
            if model_name.startswith("model1_common_")
            else "model2_common"
        )
        flag_name = "model2_complete"

    outcome_name = row["outcome"]

    matching_cells = (
        category_cell_diagnostics_24c12
        .filter(
            (pl.col("sample") == sample_name)
            & (pl.col("category_variable") == matched_category)
            & (pl.col("category_value") == matched_level)
            & (pl.col("outcome") == outcome_name)
        )
    )

    if matching_cells.height == 1:
        cell = matching_cells.row(0, named=True)

        category_link_rows_24c12.append({
            "outcome": outcome_name,
            "model": model_name,
            "term": term,
            "beta": row["beta"],
            "se": row["se"],
            "p_value": row["p_value"],
            "odds_ratio": row["odds_ratio"],
            "category_variable": matched_category,
            "category_value": matched_level,
            "sample": sample_name,
            "category_n": cell["category_n"],
            "event_count": cell["event_count"],
            "non_event_count": cell["non_event_count"],
            "event_rate": cell["event_rate"],
            "zero_events": cell["zero_events"],
            "zero_non_events": cell["zero_non_events"],
        })

extreme_category_links_24c12 = pl.DataFrame(
    category_link_rows_24c12
)

print(
    "Extreme categorical coefficients linked to cell counts:",
    extreme_category_links_24c12.height,
)

if extreme_category_links_24c12.height > 0:
    print(extreme_category_links_24c12)


print()
print("24C-12H. SAVE CATEGORICAL AND SEPARATION DIAGNOSTICS")

category_cell_diagnostics_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_category_outcome_cell_counts.csv"
)

category_missing_diagnostics_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_category_missing_counts.csv"
)

zero_event_categories_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_zero_event_categories.csv"
)

sparse_categories_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_sparse_categories.csv"
)

reference_candidate_diagnostics_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_candidate_reference_categories.csv"
)

coefficient_diagnostics_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_coefficient_diagnostics.csv"
)

extreme_coefficients_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_extreme_coefficients.csv"
)

separation_screen_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_separation_screen.csv"
)

possible_separation_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_possible_separation_cells.csv"
)

extreme_category_links_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_extreme_coefficients_linked_to_cells.csv"
)

print()
print("DIAGNOSTIC FILES SAVED")
print(diagnostics_dir_24c12)

print()
print("24C-12I. VERIFY CATEGORY TERM CODING AND REFERENCE LEVELS")

reference_verification_rows_24c12 = []

coefficient_sources_24c12 = [
    ("major_obstacle", major_coefficient_results),
    ("any_obstacle", any_coefficient_results_24c10),
]

for outcome_name, coefficient_table in coefficient_sources_24c12:
    for model_name in coefficient_table["model"].unique().to_list():

        if model_name.startswith("model1_full_"):
            flag_name = "model1_complete"
        else:
            flag_name = "model2_complete"

        model_sample = data_24c9.filter(
            pl.col(flag_name) == True
        )

        model_coefficients = coefficient_table.filter(
            pl.col("model") == model_name
        )

        for category_name in categorical_variables_24c12:
            observed_levels = (
                model_sample
                .select(category_name)
                .drop_nulls()
                .unique()
                .sort(category_name)
                .to_series()
                .to_list()
            )

            category_terms = [
                term
                for term in model_coefficients["term"].to_list()
                if term.startswith(f"{category_name}_")
            ]

            print()
            print("Outcome:", outcome_name)
            print("Model:", model_name)
            print("Variable:", category_name)
            print("Observed levels:", observed_levels)
            print("Actual coefficient terms:", category_terms)

            for category_value in observed_levels:
                category_string = str(category_value)

                matching_terms = [
                    term
                    for term in category_terms
                    if term == f"{category_name}_{category_string}"
                    or term == f"{category_name}_{category_string}.0"
                ]

                reference_verification_rows_24c12.append({
                    "outcome": outcome_name,
                    "model": model_name,
                    "category_variable": category_name,
                    "category_value": category_string,
                    "matching_term_count": len(matching_terms),
                    "matching_terms": ", ".join(matching_terms),
                    "term_present": len(matching_terms) > 0,
                })

reference_verification_24c12 = pl.DataFrame(
    reference_verification_rows_24c12
)

print()
print("REFERENCE LEVEL VERIFICATION")
print(reference_verification_24c12)

reference_verification_24c12.write_csv(
    diagnostics_dir_24c12
    / "24c12_reference_level_verification.csv"
)

print()
print("Reference verification file saved.")

print()
print("=" * 78)
print("24C-12 CATEGORICAL AND SEPARATION DIAGNOSTICS COMPLETED")
print("=" * 78)


print()
print("=" * 78)
print("24C-13. FINAL ROBUSTNESS EVIDENCE CONSOLIDATION")
print("=" * 78)

print()
print("24C-13A. PREPARE CONSOLIDATED COEFFICIENT PANEL")

robustness_dir = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "outcome_robustness"
)
robustness_dir.mkdir(parents=True, exist_ok=True)

major_panel = (
    major_coefficient_results
    .select(
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    )
    .with_columns(
        pl.lit("major_obstacle").alias("outcome")
    )
)

any_panel = (
    any_coefficient_results_24c10
    .select(
        "model",
        "term",
        "beta",
        "se",
        "p_value",
        "ci_low",
        "ci_high",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    )
    .with_columns(
        pl.lit("any_obstacle").alias("outcome")
    )
)

robustness_panel = (
    pl.concat(
        [major_panel, any_panel],
        how="vertical",
    )
    .with_columns(
        pl.when(pl.col("beta") > 0)
        .then(pl.lit("positive"))
        .when(pl.col("beta") < 0)
        .then(pl.lit("negative"))
        .otherwise(pl.lit("zero"))
        .alias("direction"),

        (pl.col("p_value") < 0.05)
        .alias("significant_5pct"),

        pl.col("model")
        .str.extract(r"_(wmedian|wweak)$", 1)
        .alias("weight"),

        pl.col("model")
        .str.extract(
            r"^(model1_full|model1_common|model2_common)",
            1,
        )
        .alias("specification"),
    )
)

print(
    f"Total coefficient observations: {robustness_panel.height}"
)
print(
    f"Outcomes: {robustness_panel['outcome'].n_unique()}"
)
print(
    f"Models: {robustness_panel['model'].n_unique()}"
)
print(
    f"Unique coefficient terms: {robustness_panel['term'].n_unique()}"
)

print()
print("24C-13B. SUMMARIZE COEFFICIENT STABILITY")

term_stability = (
    robustness_panel
    .group_by(["outcome", "term"])
    .agg(
        pl.col("model").n_unique().alias("model_count"),
        pl.col("direction").n_unique().alias("direction_count"),
        pl.col("significant_5pct")
        .cast(pl.Int64)
        .sum()
        .alias("significant_model_count"),
        pl.col("beta").mean().alias("mean_beta"),
        pl.col("beta").min().alias("min_beta"),
        pl.col("beta").max().alias("max_beta"),
        pl.col("p_value").min().alias("min_p_value"),
        pl.col("p_value").max().alias("max_p_value"),
        pl.col("odds_ratio").mean().alias("mean_odds_ratio"),
        pl.col("or_ci_low").min().alias("min_or_ci_low"),
        pl.col("or_ci_high").max().alias("max_or_ci_high"),
    )
    .with_columns(
        (pl.col("direction_count") == 1)
        .alias("direction_stable"),

        (
            pl.col("significant_model_count")
            == pl.col("model_count")
        ).alias("significant_in_all_available_models"),

        (
            pl.col("significant_model_count") == 0
        ).alias("never_significant_at_5pct"),
    )
    .with_columns(
        pl.when(
            pl.col("significant_in_all_available_models")
        )
        .then(pl.lit("consistently_significant"))
        .when(
            pl.col("significant_model_count") > 0
        )
        .then(pl.lit("significance_sensitive"))
        .otherwise(pl.lit("not_significant"))
        .alias("significance_stability")
    )
    .sort(
        ["outcome", "significant_model_count", "term"],
        descending=[False, True, False],
    )
)

print("Coefficient stability summary:")
print(term_stability)

print()
print("24C-13C. IDENTIFY DIRECTIONALLY STABLE AND SIGNIFICANT TERMS")

directionally_stable_terms = (
    term_stability
    .filter(pl.col("direction_stable"))
)

consistently_significant_terms = (
    term_stability
    .filter(
        pl.col("significant_in_all_available_models")
    )
)

significance_sensitive_terms = (
    term_stability
    .filter(
        (pl.col("significant_model_count") > 0)
        &
        (
            pl.col("significant_model_count")
            < pl.col("model_count")
        )
    )
)

print(
    "Directionally stable outcome-term combinations:",
    directionally_stable_terms.height,
)
print(
    "Consistently significant outcome-term combinations:",
    consistently_significant_terms.height,
)
print(
    "Significance-sensitive outcome-term combinations:",
    significance_sensitive_terms.height,
)

print()
print("24C-13D. COMPARE COEFFICIENT DIRECTIONS ACROSS OUTCOMES")

major_direction = (
    major_panel
    .select(
        "model",
        "term",
        pl.col("beta").alias("major_beta"),
        pl.col("p_value").alias("major_p_value"),
        pl.col("odds_ratio").alias("major_odds_ratio"),
    )
    .with_columns(
        pl.when(pl.col("major_beta") > 0)
        .then(pl.lit("positive"))
        .when(pl.col("major_beta") < 0)
        .then(pl.lit("negative"))
        .otherwise(pl.lit("zero"))
        .alias("major_direction")
    )
)

any_direction = (
    any_panel
    .select(
        "model",
        "term",
        pl.col("beta").alias("any_beta"),
        pl.col("p_value").alias("any_p_value"),
        pl.col("odds_ratio").alias("any_odds_ratio"),
    )
    .with_columns(
        pl.when(pl.col("any_beta") > 0)
        .then(pl.lit("positive"))
        .when(pl.col("any_beta") < 0)
        .then(pl.lit("negative"))
        .otherwise(pl.lit("zero"))
        .alias("any_direction")
    )
)

outcome_direction_comparison = (
    major_direction
    .join(
        any_direction,
        on=["model", "term"],
        how="inner",
        validate="1:1",
    )
    .with_columns(
        (
            pl.col("major_direction")
            == pl.col("any_direction")
        ).alias("same_direction"),

        (
            pl.col("major_beta")
            - pl.col("any_beta")
        ).alias("beta_difference"),

        (
            pl.col("major_odds_ratio")
            / pl.col("any_odds_ratio")
        ).alias("odds_ratio_ratio"),
    )
    .with_columns(
        pl.when(pl.col("same_direction"))
        .then(pl.lit("same_direction"))
        .when(
            (pl.col("major_direction") == "negative")
            &
            (pl.col("any_direction") == "positive")
        )
        .then(pl.lit("negative_to_positive"))
        .when(
            (pl.col("major_direction") == "positive")
            &
            (pl.col("any_direction") == "negative")
        )
        .then(pl.lit("positive_to_negative"))
        .otherwise(pl.lit("other"))
        .alias("direction_comparison")
    )
)

direction_comparison_summary = (
    outcome_direction_comparison
    .group_by("direction_comparison")
    .agg(
        pl.len().alias("coefficient_pair_count")
    )
    .with_columns(
        (
            pl.col("coefficient_pair_count")
            / outcome_direction_comparison.height
            * 100
        ).alias("percentage")
    )
    .sort("coefficient_pair_count", descending=True)
)

print("Outcome direction comparison:")
print(direction_comparison_summary)

print()
print("24C-13E. SUMMARIZE MODEL-LEVEL EVIDENCE")

model_stability = (
    robustness_panel
    .group_by(["outcome", "model", "specification", "weight"])
    .agg(
        pl.len().alias("coefficient_count"),
        pl.col("significant_5pct")
        .cast(pl.Int64)
        .sum()
        .alias("significant_count"),
        pl.col("beta").gt(0)
        .sum()
        .alias("positive_count"),
        pl.col("beta").lt(0)
        .sum()
        .alias("negative_count"),
    )
    .with_columns(
        (
            pl.col("significant_count")
            / pl.col("coefficient_count")
        ).alias("significant_share")
    )
    .sort(["outcome", "model"])
)

print("Model-level descriptive summary:")
print(model_stability)

print()
print("24C-13F. SUMMARIZE WEIGHT-LEVEL EVIDENCE")

weight_stability = (
    robustness_panel
    .group_by(["outcome", "weight"])
    .agg(
        pl.len().alias("coefficient_count"),
        pl.col("significant_5pct")
        .cast(pl.Int64)
        .sum()
        .alias("significant_count"),
        pl.col("beta").gt(0)
        .sum()
        .alias("positive_count"),
        pl.col("beta").lt(0)
        .sum()
        .alias("negative_count"),
    )
    .with_columns(
        (
            pl.col("significant_count")
            / pl.col("coefficient_count")
        ).alias("significant_share")
    )
    .sort(["outcome", "weight"])
)

print("Weight-level descriptive summary:")
print(weight_stability)

print()
print("24C-13G. DOCUMENT REFERENCE CATEGORIES")

reference_categories = pl.DataFrame(
    {
        "category_variable": ["a2", "a4a"],
        "observed_category_count": [22, 19],
        "reference_category": ["1", "1"],
        "estimated_dummy_count": [21, 18],
        "verification_status": [
            "Consistent with observed levels and coefficient terms",
            "Consistent with observed levels and coefficient terms",
        ],
    }
)

print("Reference category summary:")
print(reference_categories)

print()
print("24C-13H. DOCUMENT LOW-EVENT CATEGORY WARNINGS")

low_event_categories = pl.DataFrame(
    {
        "outcome": [
            "major_obstacle",
            "major_obstacle",
            "major_obstacle",
            "major_obstacle",
            "major_obstacle",
            "major_obstacle",
        ],
        "specification": [
            "model1_full",
            "model1_full",
            "model1_common",
            "model1_common",
            "model2_common",
            "model2_common",
        ],
        "category_variable": [
            "a2", "a2", "a2", "a2", "a2", "a2",
        ],
        "category_value": [
            5, 10, 5, 10, 5, 10,
        ],
        "sample_n": [
            113, 104, 110, 87, 110, 87,
        ],
        "event_count": [
            2, 1, 2, 1, 2, 1,
        ],
        "non_event_count": [
            111, 103, 108, 86, 108, 86,
        ],
        "diagnostic_status": [
            "Low event frequency",
            "Low event frequency",
            "Low event frequency",
            "Low event frequency",
            "Low event frequency",
            "Low event frequency",
        ],
    }
)

print("Low-event category warnings:")
print(low_event_categories)

print()
print("24C-13I. SAVE FINAL CONSOLIDATION FILES")

consolidation_files = {
    "24c13_consolidated_coefficient_panel.csv": robustness_panel,
    "24c13_coefficient_stability_summary.csv": term_stability,
    "24c13_directionally_stable_terms.csv": directionally_stable_terms,
    "24c13_consistently_significant_terms.csv": consistently_significant_terms,
    "24c13_significance_sensitive_terms.csv": significance_sensitive_terms,
    "24c13_outcome_direction_comparison.csv": outcome_direction_comparison,
    "24c13_direction_comparison_summary.csv": direction_comparison_summary,
    "24c13_model_level_stability.csv": model_stability,
    "24c13_weight_level_stability.csv": weight_stability,
    "24c13_reference_category_summary.csv": reference_categories,
    "24c13_low_event_category_warnings.csv": low_event_categories,
}

for file_name, output_df in consolidation_files.items():
    output_df.write_csv(robustness_dir / file_name)

print()
print("FINAL CONSOLIDATION FILES SAVED")
print(f"Output directory: {robustness_dir}")
print(f"Files saved: {len(consolidation_files)}")

print()
print("24C-13J. FINAL VALIDATION")

print(
    "Coefficient panel rows:",
    robustness_panel.height,
)
print(
    "Stability summary rows:",
    term_stability.height,
)
print(
    "Outcome comparison pairs:",
    outcome_direction_comparison.height,
)
print(
    "Model-level summary rows:",
    model_stability.height,
)
print(
    "Weight-level summary rows:",
    weight_stability.height,
)

assert robustness_panel.height == (
    major_panel.height + any_panel.height
)

assert outcome_direction_comparison.height == min(
    major_panel.height,
    any_panel.height,
)

assert robustness_panel["outcome"].n_unique() == 2

assert robustness_panel["weight"].null_count() == 0

assert robustness_panel["specification"].null_count() == 0

print()
print("FINAL ROBUSTNESS CONSOLIDATION VALIDATION PASSED")

print()
print("=" * 78)
print("24C-13 FINAL ROBUSTNESS EVIDENCE CONSOLIDATION COMPLETED")
print("=" * 78)


print()
print("=" * 72)
print("24C-14: SAMPLING REGION VS ESTABLISHMENT REGION")
print("=" * 72)

source_path_24c14 = Path(
    "data/processed/indonesia_2023.parquet"
)

if not source_path_24c14.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {source_path_24c14}"
    )

source_data_24c14 = pl.read_parquet(source_path_24c14)

if "idstd" not in source_data_24c14.columns:
    raise ValueError(
        "The source dataset does not contain idstd."
    )

if "a3a" in source_data_24c14.columns:
    establishment_region_column_24c14 = "a3a"
elif "region_code" in source_data_24c14.columns:
    establishment_region_column_24c14 = "region_code"
else:
    raise ValueError(
        "Neither a3a nor region_code exists in the source Parquet."
    )

if "idstd" not in data.columns or "a2" not in data.columns:
    raise ValueError(
        "The regression data must contain idstd and a2."
    )

geography_source_24c14 = (
    source_data_24c14
    .select(
        "idstd",
        pl.col(establishment_region_column_24c14)
        .alias("a3a"),
    )
    .with_columns(
        pl.col("idstd").cast(pl.String),
        pl.col("a3a").cast(pl.Float64, strict=False),
    )
)

duplicate_ids_24c14 = (
    geography_source_24c14
    .group_by("idstd")
    .len()
    .filter(pl.col("len") > 1)
)

if duplicate_ids_24c14.height > 0:
    raise ValueError(
        "Duplicate idstd values found in the geography source."
    )

geography_data_24c14 = (
    data
    .select(
        "idstd",
        "a2",
    )
    .with_columns(
        pl.col("idstd").cast(pl.String),
        pl.col("a2").cast(pl.Float64, strict=False),
    )
    .join(
        geography_source_24c14,
        on="idstd",
        how="left",
        validate="1:1",
    )
)

geography_data_24c14 = geography_data_24c14.with_columns(
    (
        pl.col("a2") == pl.col("a3a")
    ).alias("same_numeric_code")
)

geography_coverage_24c14 = (
    geography_data_24c14
    .select(
        pl.len().alias("total_observations"),
        pl.col("a2").is_null().sum().alias("missing_a2"),
        pl.col("a3a").is_null().sum().alias("missing_a3a"),
    )
)

geography_match_summary_24c14 = (
    geography_data_24c14
    .filter(
        pl.col("a2").is_not_null()
        & pl.col("a3a").is_not_null()
    )
    .group_by("same_numeric_code")
    .agg(
        pl.len().alias("observations")
    )
    .with_columns(
        (
            pl.col("observations")
            / pl.col("observations").sum()
            * 100
        ).alias("percentage")
    )
    .sort("same_numeric_code", descending=True)
)

geography_cross_tab_24c14 = (
    geography_data_24c14
    .filter(
        pl.col("a2").is_not_null()
        & pl.col("a3a").is_not_null()
    )
    .group_by(["a2", "a3a"])
    .agg(
        pl.len().alias("observations")
    )
    .with_columns(
        (
            pl.col("observations")
            / pl.col("observations").sum()
            * 100
        ).alias("percentage")
    )
    .sort(["a2", "a3a"])
)

geography_mismatch_24c14 = (
    geography_data_24c14
    .filter(
        pl.col("same_numeric_code") == False
    )
    .select(
        "idstd",
        "a2",
        "a3a",
    )
)

print()
print("GEOGRAPHY SOURCE")
print(f"Source file: {source_path_24c14}")
print(
    f"Establishment region column: "
    f"{establishment_region_column_24c14}"
)

print()
print("GEOGRAPHY DATA COVERAGE")
print(geography_coverage_24c14)

print()
print("NUMERIC CODE MATCH SUMMARY")
print(geography_match_summary_24c14)

print()
print("SAMPLING REGION BY ESTABLISHMENT REGION")
print(geography_cross_tab_24c14)

print()
print("NON-MATCHING OBSERVATIONS")
print(f"Number of mismatches: {geography_mismatch_24c14.height}")

output_dir_24c14 = Path(
    "data/processed/outcome_robustness"
)
output_dir_24c14.mkdir(
    parents=True,
    exist_ok=True,
)

geography_cross_tab_24c14.write_csv(
    output_dir_24c14
    / "24c14_sampling_vs_establishment_region.csv"
)

geography_match_summary_24c14.write_csv(
    output_dir_24c14
    / "24c14_geography_code_match_summary.csv"
)

geography_mismatch_24c14.write_csv(
    output_dir_24c14
    / "24c14_geography_mismatches.csv"
)

geography_coverage_24c14.write_csv(
    output_dir_24c14
    / "24c14_geography_data_coverage.csv"
)

print()
print("24C-14 GEOGRAPHY AUDIT FILES SAVED")
print(f"Output directory: {output_dir_24c14}")
print("24C-14 GEOGRAPHY AUDIT COMPLETED")


print()
print("=" * 72)
print("24C-15: GEOGRAPHIC MISMATCHES IN REGRESSION SAMPLES")
print("=" * 72)

mismatch_ids_24c15 = [
    "1197771",
    "1197857",
    "1198368",
    "1198578",
    "1198788",
    "1199068",
    "1199225",
    "1199766",
    "1199907",
]

sample_check_24c15 = (
    data_24c9
    .with_columns(
        pl.col("idstd").cast(pl.String)
    )
    .filter(
        pl.col("idstd").is_in(mismatch_ids_24c15)
    )
    .select(
        "idstd",
        "a2",
        "major_obstacle",
        "any_obstacle",
        "model1_complete",
        "model2_complete",
    )
    .sort("idstd")
)

print()
print("MISMATCH OBSERVATIONS IN REGRESSION DATA")
print(sample_check_24c15)

print()
print("SAMPLE MEMBERSHIP SUMMARY")

sample_membership_24c15 = (
    sample_check_24c15
    .select(
        pl.len().alias("matched_observations"),
        pl.col("model1_complete")
        .cast(pl.Int64)
        .sum()
        .alias("model1_full_members"),
        pl.col("model2_complete")
        .cast(pl.Int64)
        .sum()
        .alias("common_sample_members"),
    )
)

print(sample_membership_24c15)

output_path_24c15 = (
    Path("data/processed/outcome_robustness")
    / "24c15_geographic_mismatch_sample_membership.csv"
)

sample_check_24c15.write_csv(output_path_24c15)

print()
print(f"Output saved: {output_path_24c15}")
print("24C-15 COMPLETED")


print()
print("=" * 72)
print("24C-16: ESTABLISHMENT REGION CATEGORY VALIDATION")
print("=" * 72)

source_data_24c16 = pl.read_parquet(
    "data/processed/indonesia_2023.parquet"
)

establishment_region_24c16 = (
    source_data_24c16
    .select(
        "idstd",
        "a3a",
    )
    .with_columns(
        pl.col("idstd").cast(pl.String),
        pl.col("a3a").cast(pl.Float64, strict=False),
    )
)

print()
print("ESTABLISHMENT REGION CATEGORY DISTRIBUTION")

a3a_distribution_24c16 = (
    establishment_region_24c16
    .group_by("a3a")
    .agg(
        pl.len().alias("observations")
    )
    .with_columns(
        (
            pl.col("observations")
            / pl.col("observations").sum()
            * 100
        ).alias("percentage")
    )
    .sort("a3a")
)

print(a3a_distribution_24c16)

print()
print("SAMPLING REGION CATEGORY DISTRIBUTION")

a2_distribution_24c16 = (
    data_24c9
    .group_by("a2")
    .agg(
        pl.len().alias("observations")
    )
    .sort("a2")
)

print(a2_distribution_24c16)

print()
print("ESTABLISHMENT REGION CATEGORY SUMMARY")

print(
    establishment_region_24c16.select(
        pl.col("a3a").n_unique().alias("unique_categories"),
        pl.col("a3a").min().alias("minimum_code"),
        pl.col("a3a").max().alias("maximum_code"),
        pl.col("a3a").is_null().sum().alias("missing_values"),
    )
)

output_dir_24c16 = Path(
    "data/processed/outcome_robustness"
)
output_dir_24c16.mkdir(
    parents=True,
    exist_ok=True,
)

a3a_distribution_24c16.write_csv(
    output_dir_24c16
    / "24c16_establishment_region_distribution.csv"
)

print()
print("24C-16 CATEGORY VALIDATION COMPLETED")


print()
print("=" * 72)
print("24C-17: GEOGRAPHIC SPECIFICATION SENSITIVITY")
print("=" * 72)

import polars as pl
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
data_path = project_dir / "data" / "processed" / "indonesia_2023.parquet"
output_dir = project_dir / "data" / "processed" / "outcome_robustness"
output_dir.mkdir(parents=True, exist_ok=True)

source_data = pl.read_parquet(data_path).select(
    "idstd",
    "a3a",
)

if source_data.select(pl.col("idstd").n_unique()).item() != source_data.height:
    raise ValueError("Duplicate idstd values found in establishment-region data.")

data_geo = data_24c9.join(
    source_data,
    on="idstd",
    how="left",
    validate="1:1",
)

if data_geo.get_column("a3a").null_count() != 0:
    raise ValueError("Missing a3a values detected after joining.")

if data_geo.height != data_24c9.height:
    raise ValueError("Geographic sensitivity data row count changed.")

if data_geo.select(
    (pl.col("a2").cast(pl.Float64) != pl.col("a3a")).sum()
).item() != 9:
    raise ValueError("The expected nine geographic mismatches were not reproduced.")

geo_model1_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
    svy.Cat("a4a"),
    svy.Cat("a3a"),
]

geo_model2_predictors = geo_model1_predictors + [
    "bank_account",
]

print()
print("EXACT ESTIMATION-DOMAIN DESIGN DF AUDIT")

audit_specs = [
    (
        "model1_full",
        "model1_complete",
        geo_model1_predictors,
    ),
    (
        "model1_common",
        "model2_complete",
        geo_model1_predictors,
    ),
    (
        "model2_common",
        "model2_complete",
        geo_model2_predictors,
    ),
]

audit_outcomes = [
    "major_obstacle",
    "any_obstacle",
]

audit_weights = [
    "wmedian",
    "wweak",
]

audit_rows = []

for outcome_name in audit_outcomes:
    for weight_name in audit_weights:
        for model_name, condition_name, predictors in audit_specs:

            predictor_cols = [
    predictor if isinstance(predictor, str) else predictor.name
    for predictor in predictors
]

            required_cols = [
    outcome_name,
    *predictor_cols,
    "strata",
    "idstd",
]

            domain_conditions = [
                (pl.col(condition_name) == True).fill_null(False),
                pl.col(weight_name).is_not_null(),
                pl.col(weight_name).is_finite(),
                pl.col(weight_name) > 0,
            ]

            domain_conditions.extend(
                pl.col(col_name).is_not_null()
                for col_name in required_cols
            )

            exact_domain = data_geo.filter(
                pl.all_horizontal(domain_conditions)
            )

            psu_by_stratum = (
                exact_domain
                .group_by("strata")
                .agg(
                    pl.col("idstd")
                    .n_unique()
                    .alias("psu_count")
                )
            )

            total_psu_within_strata = int(
                psu_by_stratum["psu_count"].sum()
            )

            number_of_strata = psu_by_stratum.height

            raw_design_df = (
                total_psu_within_strata
                - number_of_strata
            )

            duplicate_psu = (
                exact_domain
                .group_by("idstd")
                .agg(
                    pl.col("strata")
                    .n_unique()
                    .alias("strata_count")
                )
                .filter(
                    pl.col("strata_count") > 1
                )
                .height
            )

            singleton_strata = (
                psu_by_stratum
                .filter(
                    pl.col("psu_count") == 1
                )
                .height
            )

            audit_rows.append({
                "outcome": outcome_name,
                "model": model_name,
                "weight": weight_name,
                "observations": exact_domain.height,
                "unique_psu_within_strata": total_psu_within_strata,
                "unique_strata": number_of_strata,
                "singleton_strata": singleton_strata,
                "raw_design_df": raw_design_df,
                "duplicate_psu_across_strata": duplicate_psu,
            })

design_df_audit = pl.DataFrame(audit_rows)

print(
    design_df_audit.sort(
        ["outcome", "weight", "model"]
    )
)

audit_path = (
    project_dir
    / "data"
    / "processed"
    / "script55_survey_weighted_logistic_regression"
    / "exact_estimation_domain_design_df_audit.csv"
)

audit_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

design_df_audit.write_csv(audit_path)

print()
print(f"Audit saved to: {audit_path}")

geo_designs = {
    "wmedian": design_median,
    "wweak": design_weak,
}

geo_samples = {}

for weight_name, design_obj in geo_designs.items():
    geo_samples[weight_name] = svy.Sample(
        data_geo,
        design_obj,
    )

def fit_geographic_models(sample_obj, weight_name, outcome_name):
    fitted_models = {}

    fitted_models[f"model1_full_{weight_name}"] = sample_obj.glm.fit(
        y=outcome_name,
        x=geo_model1_predictors,
        family="binomial",
        link="logit",
        where=(svy.col("model1_complete") == True),
    )

    fitted_models[f"model1_common_{weight_name}"] = sample_obj.glm.fit(
        y=outcome_name,
        x=geo_model1_predictors,
        family="binomial",
        link="logit",
        where=(svy.col("model2_complete") == True),
    )

    fitted_models[f"model2_common_{weight_name}"] = sample_obj.glm.fit(
        y=outcome_name,
        x=geo_model2_predictors,
        family="binomial",
        link="logit",
        where=(svy.col("model2_complete") == True),
    )

    return fitted_models

geo_results = {}

for outcome_name in ["major_obstacle", "any_obstacle"]:
    for weight_name, sample_obj in geo_samples.items():
        model_key = f"{outcome_name}_{weight_name}"

        geo_results[model_key] = fit_geographic_models(
            sample_obj,
            weight_name,
            outcome_name,
        )

print()
print("=" * 72)
print("24C-17A: GEOGRAPHIC MODEL INFERENCE DIAGNOSTICS")
print("=" * 72)

for model_key, fitted_models in geo_results.items():
    for model_name, fitted_model in fitted_models.items():
        fitted = fitted_model.fitted

        print()
        print(f"MODEL KEY: {model_key}")
        print(f"MODEL NAME: {model_name}")
        print("FEATURE NAMES:")
        print(fitted.feature_names)

        print("TERM INFORMATION:")
        print(fitted.term_info)

        print("DESIGN DEGREES OF FREEDOM:")
        try:
            print(fitted._design_df())
        except Exception as exc:
            print(f"Could not retrieve design df: {type(exc)._name_}: {exc}")

        print("COVARIANCE MATRIX:")
        print(f"Shape: {fitted.cov_matrix.shape}")
        print(f"Finite: {np.isfinite(fitted.cov_matrix).all()}")

        print("REGION TERM TEST:")
        try:
            print(fitted.term_test("a3a"))
        except Exception as exc:
            print(f"Term test failed: {type(exc)._name_}: {exc}")

geo_coefficient_rows = []

for outcome_name in ["major_obstacle", "any_obstacle"]:
    for weight_name in ["wmedian", "wweak"]:
        model_key = f"{outcome_name}_{weight_name}"

        for model_name, fitted_model in geo_results[model_key].items():
            coefficients = fitted_model.fitted.coefs

            for coef in coefficients:
                geo_coefficient_rows.append(
                    {
                        "outcome": outcome_name,
                        "weight": weight_name,
                        "model": model_name,
                        "term": coef.term,
                        "estimate": coef.est,
                        "std_error": coef.se,
                        "conf_low": coef.lci,
                        "conf_high": coef.uci,
                        "p_value": coef.wald.p_value,
                    }
                )

geo_coefficients = pl.DataFrame(geo_coefficient_rows)

geo_coefficients.write_csv(
    output_dir / "24c17_geographic_sensitivity_coefficients.csv"
)

print()
print("GEOGRAPHIC SENSITIVITY MODEL SUMMARY")
print(
    geo_coefficients.group_by(
        ["outcome", "weight", "model"]
    ).agg(
        pl.len().alias("coefficient_count"),
        pl.col("p_value").is_finite().sum().alias("finite_p_values"),
    ).sort(
        ["outcome", "weight", "model"]
    )
)

print()
print("GEOGRAPHIC SENSITIVITY COEFFICIENTS")
print(geo_coefficients)

print()
print("24C-17 GEOGRAPHIC SENSITIVITY COMPLETED")

geo_coefficients.write_csv(
    output_dir / "24c17_geographic_sensitivity_coefficients.csv"
)

print()
print("GEOGRAPHIC SENSITIVITY MODEL SUMMARY")
print(
    geo_coefficients.group_by(
        ["outcome", "weight", "model"]
    ).agg(
        pl.len().alias("coefficient_count"),
        pl.col("p_value").is_finite().sum().alias("finite_p_values"),
    ).sort(
        ["outcome", "weight", "model"]
    )
)

print()
print("GEOGRAPHIC SENSITIVITY COEFFICIENTS")
print(geo_coefficients)

print()
print("24C-17 GEOGRAPHIC SENSITIVITY COMPLETED")

print()
print()

# 24C-23: VALIDATED JOINT WALD TEST FOR ESTABLISHMENT REGION (a3a)

import numpy as np
import polars as pl
from scipy.stats import f as f_distribution

print()
print("=" * 78)
print("24C-23: VALIDATED JOINT WALD TEST FOR ESTABLISHMENT REGION (a3a)")
print("=" * 78)

if "geo_results" not in globals():
    raise RuntimeError(
        "geo_results is not available. "
        "Run this section in the same Python session after model estimation."
    )

joint_rows = []

for outcome_name in ["major_obstacle", "any_obstacle"]:
    for weight_name in ["wmedian", "wweak"]:

        group_key = f"{outcome_name}_{weight_name}"

        if group_key not in geo_results:
            raise KeyError(f"Missing fitted model group: {group_key}")

        fitted_group = geo_results[group_key]

        for model_name, model in fitted_group.items():

            fitted = model.fitted

            row = {
                "outcome": outcome_name,
                "weight": weight_name,
                "model": model_name,
                "status": "not_validated",
                "n_region_coefficients": None,
                "numerator_df": None,
                "denominator_df": None,
                "covariance_rank": None,
                "covariance_condition_number": None,
                "manual_wald_f": None,
                "manual_p_value": None,
                "package_wald_f": None,
                "package_p_value": None,
                "absolute_f_difference": None,
                "absolute_p_difference": None,
                "f_match": False,
                "p_match": False,
                "validation_note": None,
            }

            try:
                if fitted.cov_matrix is None:
                    raise ValueError("Covariance matrix is unavailable.")

                term_info = fitted.term_info or {}
                region_info = term_info.get("a3a")

                if (
                    region_info is None
                    or region_info.get("type") != "categorical"
                ):
                    raise ValueError(
                        "a3a categorical term information is unavailable."
                    )

                levels = region_info.get("levels", [])

                if len(levels) < 2:
                    raise ValueError(
                        "Insufficient a3a levels for a joint test."
                    )

                # Replicate the package's coefficient-name construction.
                coefficient_names = [
                    f"a3a_{level}"
                    for level in levels[1:]
                ]

                fitted_names = list(fitted.keys())

                missing_names = [
                    name
                    for name in coefficient_names
                    if name not in fitted_names
                ]

                if missing_names:
                    raise ValueError(
                        f"Missing region coefficients: {missing_names}"
                    )

                region_indices = [
                    fitted_names.index(name)
                    for name in coefficient_names
                ]

                coefficients = fitted.coefs

                beta = np.asarray(
                    [
                        coefficients[index].est
                        for index in region_indices
                    ],
                    dtype=float,
                )

                covariance = np.asarray(
                    fitted.cov_matrix,
                    dtype=float,
                )

                if covariance.shape != (
                    len(fitted_names),
                    len(fitted_names),
                ):
                    raise ValueError(
                        "Covariance matrix dimensions do not match "
                        "the fitted coefficient names."
                    )

                region_covariance = covariance[
                    np.ix_(region_indices, region_indices)
                ]

                numerator_df = len(region_indices)

                if numerator_df != 21:
                    raise ValueError(
                        f"Expected 21 regional coefficients, "
                        f"found {numerator_df}."
                    )

                if not np.isfinite(beta).all():
                    raise ValueError(
                        "Regional coefficients contain non-finite values."
                    )

                if not np.isfinite(region_covariance).all():
                    raise ValueError(
                        "Regional covariance matrix contains "
                        "non-finite values."
                    )

                if not np.allclose(
                    region_covariance,
                    region_covariance.T,
                    rtol=1e-8,
                    atol=1e-10,
                ):
                    raise ValueError(
                        "Regional covariance matrix is not symmetric."
                    )

                covariance_rank = int(
                    np.linalg.matrix_rank(region_covariance)
                )

                condition_number = float(
                    np.linalg.cond(region_covariance)
                )

                if covariance_rank != numerator_df:
                    raise ValueError(
                        "Regional covariance matrix is rank deficient: "
                        f"rank={covariance_rank}, q={numerator_df}."
                    )

                if not np.isfinite(condition_number):
                    raise ValueError(
                        "Regional covariance condition number is not finite."
                    )

                # Independent calculation using the same fitted coefficients
                # and covariance matrix used by the package.
                solved_beta = np.linalg.solve(
                    region_covariance,
                    beta,
                )

                wald_quadratic = float(
                    beta @ solved_beta
                )

                manual_wald_f = (
                    wald_quadratic / numerator_df
                )

                denominator_df = float(
                    fitted._design_df()
                )

                if denominator_df <= 0:
                    raise ValueError(
                        f"Invalid design degrees of freedom: "
                        f"{denominator_df}."
                    )

                manual_p_value = float(
                    f_distribution.sf(
                        manual_wald_f,
                        numerator_df,
                        denominator_df,
                    )
                )

                # Package result from the exact same fitted object.
                package_result = fitted.term_test("a3a")

                package_wald_f = float(package_result.value)
                package_p_value = float(package_result.p_value)
                package_numerator_df = float(package_result.df_num)
                package_denominator_df = float(package_result.df_den)

                f_match = bool(
                    np.isclose(
                        manual_wald_f,
                        package_wald_f,
                        rtol=1e-8,
                        atol=1e-10,
                    )
                )

                p_match = bool(
                    np.isclose(
                        manual_p_value,
                        package_p_value,
                        rtol=1e-8,
                        atol=1e-12,
                    )
                )

                df_match = bool(
                    np.isclose(
                        denominator_df,
                        package_denominator_df,
                        rtol=0,
                        atol=1e-10,
                    )
                    and np.isclose(
                        numerator_df,
                        package_numerator_df,
                        rtol=0,
                        atol=1e-10,
                    )
                )

                row.update(
                    {
                        "status": (
                            "validated"
                            if f_match and p_match and df_match
                            else "mismatch"
                        ),
                        "n_region_coefficients": numerator_df,
                        "numerator_df": numerator_df,
                        "denominator_df": denominator_df,
                        "covariance_rank": covariance_rank,
                        "covariance_condition_number": condition_number,
                        "manual_wald_f": manual_wald_f,
                        "manual_p_value": manual_p_value,
                        "package_wald_f": package_wald_f,
                        "package_p_value": package_p_value,
                        "absolute_f_difference": abs(
                            manual_wald_f - package_wald_f
                        ),
                        "absolute_p_difference": abs(
                            manual_p_value - package_p_value
                        ),
                        "f_match": f_match,
                        "p_match": p_match,
                        "validation_note": (
                            "Manual calculation agrees with package."
                            if f_match and p_match and df_match
                            else (
                                "F, p-value, or degrees of freedom "
                                "do not agree."
                            )
                        ),
                    }
                )

            except Exception as exc:
                row["status"] = "error"
                row["validation_note"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            joint_rows.append(row)

            print()
            print(f"MODEL: {model_name}")
            print(f"OUTCOME: {outcome_name}")
            print(f"WEIGHT: {weight_name}")
            print(f"STATUS: {row['status']}")

            if row["status"] in ["validated", "mismatch"]:
                print(
                    f"Manual Wald F: "
                    f"{row['manual_wald_f']:.12f}"
                )
                print(
                    f"Package Wald F: "
                    f"{row['package_wald_f']:.12f}"
                )
                print(
                    f"Manual p-value: "
                    f"{row['manual_p_value']:.12g}"
                )
                print(
                    f"Package p-value: "
                    f"{row['package_p_value']:.12g}"
                )
                print(
                    f"Denominator df: "
                    f"{row['denominator_df']}"
                )
                print(
                    f"Covariance rank: "
                    f"{row['covariance_rank']}"
                )
                print(
                    f"Covariance condition number: "
                    f"{row['covariance_condition_number']:.6g}"
                )
            else:
                print(row["validation_note"])


# Build and save the validation table.
wald_validation = pl.DataFrame(joint_rows)

validation_path = (
    output_dir / "24c23_a3a_wald_validation.csv"
)

wald_validation.write_csv(validation_path)

print()
print("=" * 78)
print("WALD VALIDATION SUMMARY")
print("=" * 78)

print(
    wald_validation.group_by("status")
    .len()
    .sort("status")
)

print()
print("VALIDATION DETAILS")

print(
    wald_validation.select(
        [
            "outcome",
            "weight",
            "model",
            "status",
            "numerator_df",
            "denominator_df",
            "manual_wald_f",
            "package_wald_f",
            "manual_p_value",
            "package_p_value",
            "covariance_rank",
            "f_match",
            "p_match",
            "validation_note",
        ]
    )
)

validated_count = wald_validation.filter(
    pl.col("status") == "validated"
).height

mismatch_count = wald_validation.filter(
    pl.col("status") == "mismatch"
).height

error_count = wald_validation.filter(
    pl.col("status") == "error"
).height

print()
print(f"Validated models: {validated_count}")
print(f"Mismatched models: {mismatch_count}")
print(f"Models with errors: {error_count}")
print(f"Total models checked: {len(joint_rows)}")
print(f"Validation file: {validation_path}")

if validated_count == len(joint_rows):
    print()
    print(
        "CONCLUSION: All fitted models passed the independent "
        "Wald calculation validation."
    )
else:
    print()
    print(
        "CONCLUSION: Validation is incomplete. "
        "Do not use the affected Wald tests as final manuscript results."
    )

import polars as pl
import svy
from pathlib import Path

sparse_output_dir = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "outcome_robustness"
)
sparse_output_dir.mkdir(parents=True, exist_ok=True)

sparse_region_codes = [2, 5, 10, 13, 16, 18, 19]

# Pre-specified macro-regions based on the establishment-region labels.
# Java is the reference category (code 1).
data_sparse = data_geo.with_columns(
    pl.when(pl.col("a3a").cast(pl.Int64).is_in([1, 2, 3, 4, 5, 6]))
    .then(pl.lit(1))
    .when(pl.col("a3a").cast(pl.Int64).is_in([10, 11, 12, 13, 14, 15, 16]))
    .then(pl.lit(2))
    .when(pl.col("a3a").cast(pl.Int64).is_in([7, 8, 9]))
    .then(pl.lit(3))
    .when(pl.col("a3a").cast(pl.Int64).is_in([17, 18, 19]))
    .then(pl.lit(4))
    .when(pl.col("a3a").cast(pl.Int64) == 20)
    .then(pl.lit(5))
    .when(pl.col("a3a").cast(pl.Int64).is_in([21, 22]))
    .then(pl.lit(6))
    .otherwise(None)
    .cast(pl.Int8)
    .alias("a3a_macro")
)

if data_sparse.get_column("a3a_macro").null_count() != 0:
    raise ValueError("Unmapped a3a codes detected in macro-region mapping.")

macro_labels = {
    1: "Java",
    2: "Sumatra",
    3: "Bali_Nusa_Tenggara",
    4: "Kalimantan",
    5: "Sulawesi",
    6: "Maluku_Papua",
}

geo_required_columns = [
    "major_obstacle",
    "any_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a3a",
    "bank_account",
    "idstd",
    "strata",
]

missing_geo_columns = [
    column
    for column in geo_required_columns
    if column not in data_sparse.columns
]

if missing_geo_columns:
    raise ValueError(
        f"Missing geographic model columns: {missing_geo_columns}"
    )

geo_base_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a3a",
    "idstd",
    "strata",
]

geo_complete_expressions = []

for outcome_name in ["major_obstacle", "any_obstacle"]:
    model1_columns = [outcome_name] + geo_base_predictors
    model2_columns = model1_columns + ["bank_account"]

    geo_complete_expressions.extend([
        pl.all_horizontal([
            pl.col(column).is_not_null()
            for column in model1_columns
        ]).alias(f"{outcome_name}_geo_model1_complete"),

        pl.all_horizontal([
            pl.col(column).is_not_null()
            for column in model2_columns
        ]).alias(f"{outcome_name}_geo_model2_complete"),
    ])

data_sparse = data_sparse.with_columns(
    geo_complete_expressions
)

print("24C-24 outcome-specific complete-case flags created.")

# Export valid-outcome regional cell counts for each complete-case model sample.
count_rows = []
for outcome_name in ["major_obstacle", "any_obstacle"]:
    for model_name, sample_flag in [
        ("model1_full", "model1_complete"),
        ("model1_common", "model2_complete"),
        ("model2_common", "model2_complete"),
    ]:
        selected = data_sparse.filter(
            (pl.col(sample_flag) == True)
            & pl.col(outcome_name).is_not_null()
        )
        counts = (
            selected.group_by("a3a")
            .agg(
                pl.len().alias("n"),
                pl.col(outcome_name).sum().alias("events"),
            )
            .with_columns(
                pl.lit(outcome_name).alias("outcome"),
                pl.lit(model_name).alias("model"),
            )
            .select("outcome", "model", "a3a", "n", "events")
        )
        count_rows.append(counts)

regional_cell_counts = pl.concat(count_rows).sort(
    ["outcome", "model", "a3a"]
)
regional_cell_counts.write_csv(
    sparse_output_dir / "24c24_a3a_complete_case_event_counts.csv"
)

macro_count_rows = []
for outcome_name in ["major_obstacle", "any_obstacle"]:
    for model_name, sample_flag in [
        ("model1_full", "model1_complete"),
        ("model1_common", "model2_complete"),
        ("model2_common", "model2_complete"),
    ]:
        selected = data_sparse.filter(
            (pl.col(sample_flag) == True)
            & pl.col(outcome_name).is_not_null()
        )
        counts = (
            selected.group_by("a3a_macro")
            .agg(
                pl.len().alias("n"),
                pl.col(outcome_name).sum().alias("events"),
            )
            .with_columns(
                pl.lit(outcome_name).alias("outcome"),
                pl.lit(model_name).alias("model"),
                pl.col("a3a_macro").replace_strict(macro_labels).alias("macro_region"),
            )
            .select("outcome", "model", "a3a_macro", "macro_region", "n", "events")
        )
        macro_count_rows.append(counts)

macro_cell_counts = pl.concat(macro_count_rows).sort(
    ["outcome", "model", "a3a_macro"]
)
macro_cell_counts.write_csv(
    sparse_output_dir / "24c24_macro_region_event_counts.csv"
)

# Fit three complementary geographic specifications:
# 1. Original 22-region a3a model, retaining all observations.
# 2. Six macro-region model, retaining all observations.
# 3. Original a3a model after excluding seven regions
#    with fewer than 10 major-obstacle events in model1_full.
# The exclusion model represents a changed target population
# and is used only as a supplementary sensitivity analysis.
model_specs = {
    "original_22_regions": {
        "data": data_sparse,
        "region_term": svy.Cat("a3a"),
    },
    "pooled_6_macro_regions": {
        "data": data_sparse,
        "region_term": svy.Cat("a3a_macro"),
    },
    "excluding_7_sparse_regions": {
        "data": data_sparse.filter(
            ~pl.col("a3a").cast(pl.Int64).is_in(sparse_region_codes)
        ),
        "region_term": svy.Cat("a3a"),
    },
}

predictors_base = [
    "firm_age",
    "log_employees",
    "female_owner",
    svy.Cat("a4a"),
]

sparse_model_rows = []
sparse_sample_audit_rows = []

for spec_name, spec in model_specs.items():
    for weight_name, design_obj in [
        ("wmedian", design_median),
        ("wweak", design_weak),
    ]:
        sample_obj = svy.Sample(
            spec["data"],
            design_obj,
        )

        for outcome_name in ["major_obstacle", "any_obstacle"]:

            model_definitions = [
                (
                    "model1_full",
                    f"{outcome_name}_geo_model1_complete",
                    predictors_base + [spec["region_term"]],
                ),
                (
                    "model1_common",
                    f"{outcome_name}_geo_model2_complete",
                    predictors_base + [spec["region_term"]],
                ),
                (
                    "model2_common",
                    f"{outcome_name}_geo_model2_complete",
                    predictors_base
                    + [spec["region_term"], "bank_account"],
                ),
            ]

            for model_name, sample_flag, predictors in model_definitions:

                selected_data = spec["data"].filter(
                    pl.col(sample_flag) == True
                )

                sample_summary = selected_data.select(
                    pl.len().alias("observations"),
                    pl.col(outcome_name).sum().alias("events"),
                )

                sparse_sample_audit_rows.append({
                    "specification": spec_name,
                    "outcome": outcome_name,
                    "weight": weight_name,
                    "model": model_name,
                    "observations": sample_summary["observations"][0],
                    "events": sample_summary["events"][0],
                })

                fitted = sample_obj.glm.fit(
                    y=outcome_name,
                    x=predictors,
                    family="binomial",
                    link="logit",
                    where=(svy.col(sample_flag) == True),
                )

                for coef in fitted.fitted.coefs:
                    sparse_model_rows.append({
                        "specification": spec_name,
                        "outcome": outcome_name,
                        "weight": weight_name,
                        "model": model_name,
                        "term": coef.term,
                        "estimate": coef.est,
                        "std_error": coef.se,
                        "conf_low": coef.lci,
                        "conf_high": coef.uci,
                        "p_value": coef.wald.p_value,
                    })

sparse_model_results = pl.DataFrame(sparse_model_rows)

sparse_sample_audit = pl.DataFrame(
    sparse_sample_audit_rows
)

sparse_model_results.write_csv(
    sparse_output_dir
    / "24c24_sparse_region_robustness_coefficients.csv"
)

sparse_sample_audit.write_csv(
    sparse_output_dir
    / "24c24_sparse_region_sample_audit.csv"
)

print("24C-24 sparse-region robustness completed.")
print("Sparse region codes:", sparse_region_codes)

print()
print("SAMPLE AUDIT")
print(
    sparse_sample_audit.sort(
        ["outcome", "specification", "weight", "model"]
    )
)

print()
print("COEFFICIENT VALIDATION")
print(
    sparse_model_results.group_by(
        ["specification", "outcome", "weight", "model"]
    ).agg(
        pl.len().alias("coefficient_count"),
        pl.col("estimate").is_finite().sum().alias("finite_estimates"),
        pl.col("std_error").is_finite().sum().alias("finite_standard_errors"),
        pl.col("p_value").is_finite().sum().alias("finite_p_values"),
    ).sort(
        ["specification", "outcome", "weight", "model"]
    )
)

print()
print("Outputs:", sparse_output_dir)

print("24C-24 sparse-region robustness completed.")
print("Sparse region codes (<10 major events):", sparse_region_codes)
print("Outputs:", sparse_output_dir)
print(sparse_model_results.group_by(
    ["specification", "outcome", "weight", "model"]
).agg(
    pl.len().alias("coefficient_count"),
    pl.col("p_value").is_finite().sum().alias("finite_p_values"),
).sort(["specification", "outcome", "weight", "model"]))


import polars as pl
from pathlib import Path

print()
print("=" * 78)
print("24C-25. MAJOR OBSTACLE FREQUENCY BY ESTABLISHMENT REGION")
print("=" * 78)

from pathlib import Path
import polars as pl

output_dir_24c25 = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "outcome_robustness"
)

output_dir_24c25.mkdir(
    parents=True,
    exist_ok=True,
)

required_columns_24c25 = [
    "idstd",
    "a3a",
    "major_obstacle",
    "model1_complete",
    "model2_complete",
]

missing_columns_24c25 = [
    column
    for column in required_columns_24c25
    if column not in data_geo.columns
]

if missing_columns_24c25:
    raise ValueError(
        "Required columns are missing from data_geo: "
        f"{missing_columns_24c25}"
    )

data_region_24c25 = (
    data_geo
    .with_columns(
        pl.col("idstd").cast(pl.String),
        pl.col("a3a").cast(pl.Int64, strict=False),
        pl.col("major_obstacle").cast(
            pl.Int64,
            strict=False,
        ),
    )
)

if data_region_24c25.get_column("a3a").null_count() > 0:
    raise ValueError(
        "Missing establishment-region codes detected in a3a."
    )

if data_region_24c25.get_column("major_obstacle").drop_nulls().is_in(
    [0, 1]
).not_().any():
    raise ValueError(
        "major_obstacle contains values other than 0 or 1."
    )

if data_region_24c25.get_column("idstd").n_unique() != (
    data_region_24c25.height
):
    raise ValueError(
        "Duplicate idstd values detected in geographic data."
    )

observed_regions_24c25 = (
    data_region_24c25
    .get_column("a3a")
    .unique()
    .sort()
    .to_list()
)

if observed_regions_24c25 != list(range(1, 23)):
    raise ValueError(
        "Unexpected establishment-region categories. "
        f"Observed: {observed_regions_24c25}"
    )

region_sample_definitions_24c25 = [
    ("model1_full", "model1_complete"),
    ("model1_common", "model2_complete"),
    ("model2_common", "model2_complete"),
]

region_frequency_frames_24c25 = []

for model_name_24c25, sample_flag_24c25 in (
    region_sample_definitions_24c25
):

    model_region_data_24c25 = (
        data_region_24c25
        .filter(
            pl.col(sample_flag_24c25)
            & pl.col("major_obstacle").is_not_null()
        )
    )

    region_frequency_24c25 = (
        model_region_data_24c25
        .group_by("a3a")
        .agg(
            pl.len().alias("observations"),

            pl.col("major_obstacle")
            .sum()
            .alias("major_obstacle_cases"),
        )
        .with_columns(
            pl.lit(model_name_24c25).alias("model"),

            (
                pl.col("major_obstacle_cases")
                / pl.col("observations")
                * 100
            ).alias("major_obstacle_percentage"),
        )
        .select(
            "model",
            "a3a",
            "observations",
            "major_obstacle_cases",
            "major_obstacle_percentage",
        )
        .sort("a3a")
    )

    region_frequency_frames_24c25.append(
        region_frequency_24c25
    )

major_region_frequency_24c25 = pl.concat(
    region_frequency_frames_24c25,
    how="vertical",
)

print()
print("MAJOR OBSTACLE FREQUENCY BY ESTABLISHMENT REGION (a3a)")
print(major_region_frequency_24c25)

major_region_frequency_24c25.write_csv(
    output_dir_24c25
    / "24c25_major_obstacle_frequency_by_a3a.csv"
)


print()
print("=" * 78)
print("24C-26. SPARSE ESTABLISHMENT REGION AUDIT")
print("=" * 78)

sparse_event_threshold_24c26 = 10

full_sample_frequency_24c26 = (
    major_region_frequency_24c25
    .filter(
        pl.col("model") == "model1_full"
    )
    .select(
        "a3a",
        "observations",
        "major_obstacle_cases",
        "major_obstacle_percentage",
    )
    .sort("a3a")
)

sparse_region_summary_24c26 = (
    full_sample_frequency_24c26
    .with_columns(
        (
            pl.col("major_obstacle_cases")
            < sparse_event_threshold_24c26
        ).alias("sparse_region"),

        pl.lit(
            sparse_event_threshold_24c26
        ).alias("event_threshold"),
    )
)

sparse_region_codes_24c26 = (
    sparse_region_summary_24c26
    .filter(pl.col("sparse_region"))
    .get_column("a3a")
    .to_list()
)

print()
print("SPARSE ESTABLISHMENT REGIONS")
print(sparse_region_summary_24c26)

print()
print("DYNAMICALLY IDENTIFIED SPARSE REGION CODES")
print(sparse_region_codes_24c26)

expected_sparse_region_codes_24c26 = [
    2, 5, 10, 13, 16, 18, 19
]

if sparse_region_codes_24c26 != (
    expected_sparse_region_codes_24c26
):
    print(
        "WARNING: Sparse-region codes differ from "
        "the previous audit."
    )
    print(
        "Use the newly calculated codes after investigating "
        "the sample and source data."
    )

sparse_region_summary_24c26.write_csv(
    output_dir_24c25
    / "24c26_sparse_establishment_region_audit.csv"
)


print()
print("24C-26A. CREATE SPARSE-REGION EXCLUSION DATA")

data_without_sparse_regions_24c26 = (
    data_region_24c25
    .filter(
        ~pl.col("a3a").is_in(
            sparse_region_codes_24c26
        )
    )
)

retained_region_count_24c26 = (
    data_without_sparse_regions_24c26
    .get_column("a3a")
    .n_unique()
)

if retained_region_count_24c26 != (
    22 - len(sparse_region_codes_24c26)
):
    raise ValueError(
        "Unexpected number of retained establishment regions."
    )

print(
    "Rows before exclusion:",
    data_region_24c25.height,
)

print(
    "Rows after exclusion:",
    data_without_sparse_regions_24c26.height,
)

print(
    "Retained establishment regions:",
    retained_region_count_24c26,
)


print()
print("24C-26B. MODEL-SPECIFIC SAMPLE AUDIT")

sparse_sample_audit_rows_24c26 = []

for model_name_24c26, sample_flag_24c26 in (
    region_sample_definitions_24c25
):

    original_sample_24c26 = (
        data_region_24c25
        .filter(
            pl.col(sample_flag_24c26)
            & pl.col("major_obstacle").is_not_null()
        )
    )

    retained_sample_24c26 = (
        data_without_sparse_regions_24c26
        .filter(
            pl.col(sample_flag_24c26)
            & pl.col("major_obstacle").is_not_null()
        )
    )

    original_n_24c26 = original_sample_24c26.height
    retained_n_24c26 = retained_sample_24c26.height

    original_events_24c26 = int(
        original_sample_24c26[
            "major_obstacle"
        ].sum()
    )

    retained_events_24c26 = int(
        retained_sample_24c26[
            "major_obstacle"
        ].sum()
    )

    sparse_sample_audit_rows_24c26.append(
        {
            "model": model_name_24c26,
            "original_observations": original_n_24c26,
            "retained_observations": retained_n_24c26,
            "excluded_observations": (
                original_n_24c26 - retained_n_24c26
            ),
            "original_major_obstacle_cases": (
                original_events_24c26
            ),
            "retained_major_obstacle_cases": (
                retained_events_24c26
            ),
            "excluded_major_obstacle_cases": (
                original_events_24c26
                - retained_events_24c26
            ),
        }
    )

sparse_sample_audit_24c26 = pl.DataFrame(
    sparse_sample_audit_rows_24c26
)

print(sparse_sample_audit_24c26)

sparse_sample_audit_24c26.write_csv(
    output_dir_24c25
    / "24c26_sparse_region_sample_audit.csv"
)


print()
print("24C-26C. FINAL GEOGRAPHY VALIDATION")

remaining_sparse_regions_24c26 = (
    data_without_sparse_regions_24c26
    .filter(
        pl.col("a3a").is_in(
            sparse_region_codes_24c26
        )
    )
    .height
)

if remaining_sparse_regions_24c26 != 0:
    raise ValueError(
        "Sparse establishment regions remain after exclusion."
    )

if (
    sparse_sample_audit_24c26
    .get_column("excluded_observations")
    .min()
    < 0
):
    raise ValueError(
        "Invalid negative exclusion count detected."
    )

print("Geographic variable: a3a")
print("Sampling variable a2 was not used.")
print("Sparse-region audit completed successfully.")

print()
print("24C-25 AND 24C-26 COMPLETED")
print("=" * 78)

print("=" * 78)
print("=" * 78)
print("24C-27. VERIFY AND RESTORE ESTABLISHMENT REGION")
print("=" * 78)

print("\nINITIAL DATAFRAME")
print("Shape:", data.shape)

if "idstd" not in data.columns:
    raise ValueError("Missing idstd in analytic dataset.")

if "a3a" not in data.columns:

    geography_file = (
        processed_dir / "indonesia_2023.parquet"
    )

    if not geography_file.exists():
        raise FileNotFoundError(
            f"Geography source not found: {geography_file}"
        )

    print("\nLoading establishment region from:")
    print(geography_file)

    geography_source = (
        pl.read_parquet(geography_file)
        .select("idstd", "a3a")
        .with_columns(
            pl.col("idstd")
            .cast(pl.String)
            .alias("__idstd_key"),

            pl.col("a3a")
            .cast(pl.Int64, strict=False)
            .alias("a3a"),
        )
        .select("__idstd_key", "a3a")
    )

    if geography_source["__idstd_key"].null_count() > 0:
        raise ValueError(
            "Null establishment IDs in geography source."
        )

    duplicate_source_ids = (
        geography_source
        .group_by("__idstd_key")
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicate_source_ids.height > 0:
        raise ValueError(
            "Duplicate establishment IDs in geography source."
        )

    if geography_source["a3a"].null_count() > 0:
        raise ValueError(
            "Missing or invalid a3a values in geography source."
        )

    data_keyed = data.with_columns(
        pl.col("idstd")
        .cast(pl.String)
        .alias("__idstd_key")
    )

    if data_keyed["__idstd_key"].null_count() > 0:
        raise ValueError(
            "Null establishment IDs in analytic dataset."
        )

    duplicate_data_ids = (
        data_keyed
        .group_by("__idstd_key")
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicate_data_ids.height > 0:
        raise ValueError(
            "Duplicate establishment IDs in analytic dataset."
        )

    original_rows = data.height

    data = (
        data_keyed
        .join(
            geography_source,
            on="__idstd_key",
            how="left",
            validate="1:1",
        )
        .drop("__idstd_key")
    )

    if data.height != original_rows:
        raise ValueError(
            "Row count changed after geography join."
        )

    if data["a3a"].null_count() > 0:
        raise ValueError(
            "Some analytic observations could not be matched "
            "to establishment region a3a."
        )

    print("Establishment region successfully restored.")

else:
    print("a3a already exists in analytic dataset.")

print("\nDATAFRAME AFTER GEOGRAPHY JOIN")
print("Shape:", data.shape)

required_columns = [
    "idstd",
    "a3a",
    "major_obstacle",
    "model1_complete",
    "model2_complete",
    "wmedian",
    "wweak",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nREQUIRED COLUMN CHECK PASSED")

print("\nREGION CATEGORY CHECK")
print(
    data.select(
        pl.col("a3a").n_unique().alias("region_categories"),
        pl.col("a3a").null_count().alias("missing_region"),
    )
)

print("\nESTABLISHMENT REGION DISTRIBUTION")
print(
    data.group_by("a3a")
    .len()
    .sort("a3a")
)

print("\n24C-27 VALIDATION PASSED")


import polars as pl
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]

file_path = (
    project_dir
    / "data"
    / "processed"
    / "outcome_robustness"
    / "24c24_sparse_region_robustness_coefficients.csv"
)

df = pl.read_csv(file_path)

main_terms = [
    "female_owner",
    "firm_age",
    "log_employees",
    "bank_account",
]

comparison = (
    df.filter(
        pl.col("term").is_in(main_terms)
    )
    .select(
        "outcome",
        "model",
        "weight",
        "specification",
        "term",
        "estimate",
        "std_error",
        "p_value",
    )
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "term",
            "specification",
        ]
    )
)

summary = (
    comparison
    .group_by(
        ["outcome", "model", "weight", "term"]
    )
    .agg(
        pl.col("estimate").min().alias("min_estimate"),
        pl.col("estimate").max().alias("max_estimate"),
        pl.col("estimate").std().alias("estimate_sd"),
        pl.col("p_value").min().alias("min_p_value"),
        pl.col("p_value").max().alias("max_p_value"),
        (
            pl.col("estimate").min() < 0
        ).alias("has_negative"),
        (
            pl.col("estimate").max() > 0
        ).alias("has_positive"),
        pl.len().alias("specification_count"),
    )
    .with_columns(
        (
            pl.col("has_negative")
            & pl.col("has_positive")
        ).alias("sign_reversal")
    )
    .sort(["outcome", "model", "weight", "term"])
)

print("FULL MAIN COVARIATE ROBUSTNESS SUMMARY")
print(summary)

output_path = (
    project_dir
    / "data"
    / "processed"
    / "outcome_robustness"
    / "24c24_full_main_covariate_robustness.csv"
)

summary.write_csv(output_path)

print(f"Saved: {output_path}")


import polars as pl
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]

file_path = (
    project_dir
    / "data"
    / "processed"
    / "outcome_robustness"
    / "24c24_full_main_covariate_robustness.csv"
)

df = pl.read_csv(file_path)

print()
print("ROBUSTNESS AUDIT")

print()
print("1. SPECIFICATION COMPLETENESS")

print(
    df.group_by(
        ["outcome", "model", "weight"]
    )
    .agg(
        pl.col("specification_count").min().alias("min_specs"),
        pl.col("specification_count").max().alias("max_specs"),
        pl.len().alias("number_of_terms"),
    )
    .sort(["outcome", "model", "weight"])
)

print()
print("2. SIGN REVERSAL SUMMARY")

print(
    df.group_by("outcome")
    .agg(
        pl.len().alias("total_combinations"),
        pl.col("sign_reversal").sum().alias("sign_reversals"),
        (
            pl.col("sign_reversal").mean() * 100
        ).round(2).alias("sign_reversal_pct"),
    )
)

print()
print("3. SIGN REVERSALS BY TERM")

print(
    df.group_by(
        ["outcome", "term"]
    )
    .agg(
        pl.col("sign_reversal").sum().alias("reversal_cases"),
        pl.len().alias("total_cases"),
    )
    .with_columns(
        (
            pl.col("reversal_cases")
            / pl.col("total_cases")
            * 100
        ).round(2).alias("reversal_pct")
    )
    .sort(["outcome", "reversal_pct"], descending=[False, True])
)

print()
print("4. P-VALUE SENSITIVITY")

print(
    df.with_columns(
        (pl.col("min_p_value") < 0.05).alias("some_significant"),
        (pl.col("max_p_value") < 0.05).alias("all_significant"),
    )
    .group_by("outcome")
    .agg(
        pl.len().alias("total_combinations"),
        pl.col("some_significant").sum().alias("at_least_one_below_005"),
        pl.col("all_significant").sum().alias("all_below_005"),
    )
)

print()
print("5. DETAILED MAIN COVARIATE RESULTS")

print(
    df.sort(
        ["outcome", "model", "weight", "term"]
    )
)


print()
print("=" * 72)
print("SCRIPT 55 COMPLETED")
print("=" * 72)