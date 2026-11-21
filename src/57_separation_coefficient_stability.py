from pathlib import Path
import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm

from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
)


project_dir = Path(__file__).resolve().parent.parent

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
    / "script57_separation_stability"
)

output_dir.mkdir(parents=True, exist_ok=True)

required_columns = [
    "idstd",
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
]

base_data_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

if not data_path.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {data_path}"
    )

if not base_data_path.exists():
    raise FileNotFoundError(
        f"Base dataset not found: {base_data_path}"
    )

df = pl.read_parquet(data_path)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

if "bank_account" not in df.columns:
    base_df = pl.read_parquet(base_data_path)

    if "k6" not in base_df.columns:
        raise ValueError(
            "Variable k6 is not available in the base dataset."
        )

    bank_account_data = base_df.select(
        [
            "idstd",
            pl.col("k6")
            .cast(pl.Int64, strict=False)
            .alias("k6"),
        ]
    )

    if bank_account_data.get_column("idstd").n_unique() != bank_account_data.height:
        raise ValueError(
            "Duplicate idstd values found in the base dataset."
        )

    df = df.join(
        bank_account_data,
        on="idstd",
        how="left",
        validate="1:1",
    )

    df = df.with_columns(
        pl.when(pl.col("k6") == 1)
        .then(1)
        .when(pl.col("k6") == 2)
        .then(0)
        .otherwise(None)
        .cast(pl.Int64)
        .alias("bank_account")
    )

    print("Bank account variable reconstructed from k6.")

else:
    print("Bank account variable already exists.")

if "bank_account" not in df.columns:
    raise ValueError(
        "Bank account variable could not be created."
    )

print(
    "Bank account distribution:"
)

print(
    df.group_by("bank_account")
    .agg(pl.len().alias("observations"))
    .sort("bank_account")
)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("=" * 70)
print("SEPARATION AND COEFFICIENT STABILITY DIAGNOSTICS")
print("=" * 70)

print(f"Input observations: {df.height:,}")
print(f"Input variables: {df.width:,}")

analysis_columns = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
    "bank_account",
]

df = df.drop_nulls(subset=analysis_columns)

df = df.filter(
    pl.col("major_obstacle").is_in([0, 1])
    & pl.col("female_owner").is_in([0, 1])
    & pl.col("bank_account").is_in([0, 1])
    & pl.col("firm_age").is_finite()
    & pl.col("log_employees").is_finite()
)

if df.height == 0:
    raise ValueError(
        "No observations remain after complete-case filtering."
    )

outcome_values = set(
    df.get_column("major_obstacle").unique().to_list()
)

if outcome_values != {0, 1}:
    raise ValueError(
        f"The outcome must contain both 0 and 1. Found: {outcome_values}"
    )

print(f"Common complete-case sample: {df.height:,}")

outcome_summary = (
    df.group_by("major_obstacle")
    .agg(pl.len().alias("observations"))
    .sort("major_obstacle")
)

outcome_summary.write_csv(
    output_dir / "common_sample_outcome_counts.csv"
)

industry_counts = (
    df.group_by("a4a", "major_obstacle")
    .agg(pl.len().alias("observations"))
    .sort(["a4a", "major_obstacle"])
)

region_counts = (
    df.group_by("a2", "major_obstacle")
    .agg(pl.len().alias("observations"))
    .sort(["a2", "major_obstacle"])
)

industry_counts.write_csv(
    output_dir / "industry_outcome_counts.csv"
)

region_counts.write_csv(
    output_dir / "region_outcome_counts.csv"
)

def build_design_matrix(data, include_bank_account):
    numeric_columns = [
        "firm_age",
        "log_employees",
        "female_owner",
    ]

    if include_bank_account:
        numeric_columns.append("bank_account")

    matrix_parts = [
        np.ones(data.height, dtype=float)
    ]

    variable_names = ["intercept"]

    for column in numeric_columns:
        values = (
            data.get_column(column)
            .cast(pl.Float64)
            .to_numpy()
        )

        matrix_parts.append(values)
        variable_names.append(column)

    for column in ["a4a", "a2"]:
        values = (
            data.get_column(column)
            .cast(pl.String)
            .to_numpy()
        )

        categories = np.unique(values)

        if len(categories) < 2:
            raise ValueError(
                f"{column} has fewer than two categories."
            )

        reference_category = categories[0]

        for category in categories[1:]:
            dummy = (
                values == category
            ).astype(float)

            matrix_parts.append(dummy)

            variable_names.append(
                f"{column}_{category}"
            )

        print(
            f"{column}: {len(categories)} categories; "
            f"reference category = {reference_category}"
        )

    design_matrix = np.column_stack(matrix_parts)

    return design_matrix, variable_names


def fit_diagnostic_model(
    data,
    model_name,
    include_bank_account,
):
    print()
    print("-" * 70)
    print(f"Fitting {model_name}")
    print("-" * 70)

    x, variable_names = build_design_matrix(
        data,
        include_bank_account,
    )

    y = (
        data.get_column("major_obstacle")
        .cast(pl.Float64)
        .to_numpy()
    )

    matrix_rank = int(np.linalg.matrix_rank(x))
    matrix_columns = int(x.shape[1])

    condition_number = float(np.linalg.cond(x))

    print(f"Observations: {len(y):,}")
    print(f"Design matrix columns: {matrix_columns:,}")
    print(f"Design matrix rank: {matrix_rank:,}")
    print(f"Condition number: {condition_number:.4f}")

    if matrix_rank < matrix_columns:
        print(
            "WARNING: The design matrix is rank deficient."
        )

    model = sm.Logit(y, x)

    warning_messages = []
    fit_status = "not_fitted"
    error_message = ""
    converged = False
    iterations = None

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            result = model.fit(
                disp=False,
                maxiter=300,
                method="newton",
            )

            warning_messages = [
                str(item.message)
                for item in captured
            ]

        converged = bool(
            result.mle_retvals.get("converged", False)
        )

        iterations = result.mle_retvals.get(
            "iterations",
            None,
        )

        fit_status = (
            "converged"
            if converged
            else "not_converged"
        )

        params = np.asarray(result.params)
        standard_errors = np.asarray(result.bse)
        p_values = np.asarray(result.pvalues)
        confidence_intervals = np.asarray(
            result.conf_int()
        )

        predicted_probabilities = np.asarray(
            result.predict(x)
        )

        coefficient_table = pl.DataFrame(
            {
                "model": [model_name] * len(variable_names),
                "variable": variable_names,
                "coefficient": params,
                "standard_error": standard_errors,
                "p_value": p_values,
                "odds_ratio": np.exp(
                    np.clip(params, -700, 700)
                ),
                "ci_lower": confidence_intervals[:, 0],
                "ci_upper": confidence_intervals[:, 1],
            }
        )

        coefficient_table.write_csv(
            output_dir / f"{model_name.lower()}_coefficients.csv"
        )

        extreme_coefficients = int(
            np.sum(np.abs(params) > 10)
        )

        large_standard_errors = int(
            np.sum(standard_errors > 5)
        )

        extreme_probabilities = int(
            np.sum(
                (predicted_probabilities < 1e-6)
                | (predicted_probabilities > 1 - 1e-6)
            )
        )

        minimum_probability = float(
            predicted_probabilities.min()
        )

        maximum_probability = float(
            predicted_probabilities.max()
        )

        if not converged:
            fit_status = "not_converged"

        if extreme_coefficients > 0:
            fit_status += "_extreme_coefficients"

        if large_standard_errors > 0:
            fit_status += "_large_standard_errors"

        if any(
            "separation" in message.lower()
            for message in warning_messages
        ):
            fit_status += "_separation_warning"

        print(f"Fit status: {fit_status}")
        print(f"Converged: {converged}")
        print(f"Iterations: {iterations}")
        print(
            f"Coefficients with absolute value > 10: "
            f"{extreme_coefficients}"
        )
        print(
            f"Standard errors > 5: "
            f"{large_standard_errors}"
        )
        print(
            f"Extreme predicted probabilities: "
            f"{extreme_probabilities}"
        )
        print(
            f"Minimum predicted probability: "
            f"{minimum_probability:.8f}"
        )
        print(
            f"Maximum predicted probability: "
            f"{maximum_probability:.8f}"
        )

    except PerfectSeparationError as error:
        fit_status = "perfect_separation_detected"
        error_message = str(error)

        extreme_coefficients = None
        large_standard_errors = None
        extreme_probabilities = None
        minimum_probability = None
        maximum_probability = None

        print("Perfect separation detected.")
        print(error_message)

    except Exception as error:
        fit_status = "fit_failed"
        error_message = (
            f"{type(error).__name__}: {error}"
        )

        extreme_coefficients = None
        large_standard_errors = None
        extreme_probabilities = None
        minimum_probability = None
        maximum_probability = None

        print(f"Model fitting failed: {error_message}")

    return {
        "model": model_name,
        "observations": data.height,
        "design_columns": matrix_columns,
        "matrix_rank": matrix_rank,
        "condition_number": condition_number,
        "fit_status": fit_status,
        "converged": converged,
        "iterations": iterations,
        "extreme_coefficients": extreme_coefficients,
        "large_standard_errors": large_standard_errors,
        "extreme_probabilities": extreme_probabilities,
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "warning_messages": " | ".join(warning_messages),
        "error_message": error_message,
    }


model1_diagnostics = fit_diagnostic_model(
    data=df,
    model_name="Model1",
    include_bank_account=False,
)

model2_diagnostics = fit_diagnostic_model(
    data=df,
    model_name="Model2",
    include_bank_account=True,
)

diagnostics = pl.DataFrame(
    [
        model1_diagnostics,
        model2_diagnostics,
    ]
)

diagnostics.write_csv(
    output_dir / "model_separation_diagnostics.csv"
)

print()
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

print(diagnostics)

print()
print("Output files:")
print(output_dir)

print()
print("Important interpretation:")
print(
    "These are unweighted auxiliary logistic diagnostics. "
    "They do not replace the survey-weighted models."
)
print(
    "A successful fit does not prove that complete or "
    "quasi-complete separation is absent."
)
print(
    "Do not interpret these diagnostic p-values as "
    "survey-adjusted inferential results."
)

print()
print("Script57 completed.")