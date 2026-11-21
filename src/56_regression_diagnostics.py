from pathlib import Path

import numpy as np
import polars as pl


project_dir = Path(__file__).resolve().parent.parent

candidate_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023_candidate_analytic.parquet"
)

base_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "script56_regression_diagnostics"
)

output_dir.mkdir(parents=True, exist_ok=True)

if candidate_path.exists():
    data = pl.read_parquet(candidate_path)
    source_path = candidate_path
elif base_path.exists():
    data = pl.read_parquet(base_path)
    source_path = base_path
else:
    raise FileNotFoundError(
        "Neither candidate analytic data nor base parquet data was found."
    )

required_columns = [
    "idstd",
    "strata",
    "wmedian",
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
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

if "bank_account" not in data.columns:
    if "k6" in data.columns:
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
        if not base_path.exists():
            raise FileNotFoundError(
                "Base parquet file is required to retrieve k6."
            )

        base_data = pl.read_parquet(base_path)

        if "idstd" not in base_data.columns:
            raise ValueError(
                "The base parquet file does not contain idstd."
            )

        if "k6" not in base_data.columns:
            raise ValueError(
                "The base parquet file does not contain k6."
            )

        if base_data["idstd"].n_unique() != base_data.height:
            raise ValueError(
                "Duplicate idstd values found in base parquet."
            )

        bank_data = (
            base_data
            .select(["idstd", "k6"])
            .with_columns(
                pl.when(pl.col("k6") == 1)
                .then(1)
                .when(pl.col("k6") == 2)
                .then(0)
                .otherwise(None)
                .cast(pl.Int8)
                .alias("bank_account")
            )
            .select(["idstd", "bank_account"])
        )

        data = data.join(
            bank_data,
            on="idstd",
            how="left",
            validate="1:1",
        )

if data["bank_account"].null_count() == data.height:
    raise ValueError(
        "bank_account contains no valid observations."
    )

print("=" * 70)
print("WBES INDONESIA 2023")
print("REGRESSION DIAGNOSTICS AND MODEL VALIDATION")
print("=" * 70)

print(f"Source file: {source_path}")
print(f"Dataset dimensions: {data.height:,} rows x {data.width:,} columns")

print("\n1. BANK ACCOUNT CODING AUDIT")
print("-" * 70)

if "k6" in data.columns:
    k6_audit = (
        data.group_by("k6")
        .agg(pl.len().alias("observations"))
        .sort("k6")
    )

    k6_audit.write_csv(
        output_dir / "bank_account_k6_coding_audit.csv"
    )

    print(k6_audit)

bank_account_audit = (
    data.group_by("bank_account")
    .agg(pl.len().alias("observations"))
    .sort("bank_account")
)

bank_account_audit.write_csv(
    output_dir / "bank_account_distribution.csv"
)

print(bank_account_audit)

print("\n2. MODEL SAMPLE DEFINITIONS")
print("-" * 70)

model1_columns = [
    "major_obstacle",
    "firm_age",
    "log_employees",
    "female_owner",
    "a4a",
    "a2",
    "strata",
    "idstd",
    "wmedian",
]

model2_columns = model1_columns + ["bank_account"]

model1_sample = data.drop_nulls(model1_columns)
model2_sample = data.drop_nulls(model2_columns)

common_columns = model2_columns

common_sample = data.drop_nulls(common_columns)

sample_summary = pl.DataFrame(
    {
        "sample": [
            "Model 1 full sample",
            "Model 1 common sample",
            "Model 2 common sample",
        ],
        "observations": [
            model1_sample.height,
            common_sample.height,
            model2_sample.height,
        ],
        "major_obstacle_cases": [
            model1_sample["major_obstacle"].sum(),
            common_sample["major_obstacle"].sum(),
            model2_sample["major_obstacle"].sum(),
        ],
    }
).with_columns(
    (
        pl.col("major_obstacle_cases")
        / pl.col("observations")
    ).alias("unweighted_major_obstacle_rate")
)

sample_summary.write_csv(
    output_dir / "sample_size_validation.csv"
)

print(sample_summary)

print("\n3. OUTCOME DISTRIBUTION BY INDUSTRY")
print("-" * 70)

industry_counts = (
    common_sample
    .group_by(["a4a", "major_obstacle"])
    .agg(pl.len().alias("observations"))
    .sort(["a4a", "major_obstacle"])
)

industry_pivot = (
    industry_counts
    .pivot(
        on="major_obstacle",
        index="a4a",
        values="observations",
        aggregate_function="sum",
    )
    .fill_null(0)
)

industry_pivot.write_csv(
    output_dir / "industry_outcome_counts.csv"
)

print(industry_pivot)

print("\n4. OUTCOME DISTRIBUTION BY REGION")
print("-" * 70)

region_counts = (
    common_sample
    .group_by(["a2", "major_obstacle"])
    .agg(pl.len().alias("observations"))
    .sort(["a2", "major_obstacle"])
)

region_pivot = (
    region_counts
    .pivot(
        on="major_obstacle",
        index="a2",
        values="observations",
        aggregate_function="sum",
    )
    .fill_null(0)
)

region_pivot.write_csv(
    output_dir / "region_outcome_counts.csv"
)

print(region_pivot)

print("\n5. SPARSE CATEGORY AND ZERO-CELL AUDIT")
print("-" * 70)

sparse_results = []

for variable in ["a4a", "a2"]:
    category_summary = (
        common_sample
        .group_by(variable)
        .agg(
            pl.len().alias("total"),
            (pl.col("major_obstacle") == 1)
            .sum()
            .alias("major_cases"),
            (pl.col("major_obstacle") == 0)
            .sum()
            .alias("non_major_cases"),
        )
        .sort(variable)
    )

    for row in category_summary.iter_rows(named=True):
        total = row["total"]
        major_cases = row["major_cases"]
        non_major_cases = row["non_major_cases"]

        sparse_results.append(
            {
                "variable": variable,
                "category": str(row[variable]),
                "total": total,
                "major_cases": major_cases,
                "non_major_cases": non_major_cases,
                "zero_major_cases": major_cases == 0,
                "zero_non_major_cases": non_major_cases == 0,
                "fewer_than_10_major_cases": major_cases < 10,
                "fewer_than_10_non_major_cases": non_major_cases < 10,
            }
        )

sparse_audit = pl.DataFrame(sparse_results)

sparse_audit.write_csv(
    output_dir / "sparse_category_audit.csv"
)

print(sparse_audit)

print("\n6. CONTINUOUS PREDICTOR CORRELATIONS")
print("-" * 70)

numeric_predictors = [
    "firm_age",
    "log_employees",
    "female_owner",
]

correlation_data = (
    common_sample
    .select(numeric_predictors)
    .cast(
        {
            column: pl.Float64
            for column in numeric_predictors
        }
    )
)

correlation_matrix = correlation_data.corr()

correlation_matrix.write_csv(
    output_dir / "continuous_predictor_correlations.csv"
)

print(correlation_matrix)

print("\n7. MULTICOLLINEARITY: VIF")
print("-" * 70)


def build_design_matrix(frame, include_bank_account):
    columns = [
        "firm_age",
        "log_employees",
        "female_owner",
    ]

    if include_bank_account:
        columns.append("bank_account")

    matrix_columns = []
    matrix_names = []

    for column in columns:
        values = (
            frame
            .get_column(column)
            .cast(pl.Float64)
            .to_numpy()
        )

        matrix_columns.append(values)
        matrix_names.append(column)

    for categorical in ["a4a", "a2"]:
        categories = (
            frame
            .get_column(categorical)
            .cast(pl.String)
            .unique()
            .sort()
            .to_list()
        )

        for category in categories[1:]:
            values = (
                frame
                .get_column(categorical)
                .cast(pl.String)
                .eq(category)
                .cast(pl.Float64)
                .to_numpy()
            )

            matrix_columns.append(values)
            matrix_names.append(
                f"{categorical}_{category}"
            )

    matrix = np.column_stack(matrix_columns)

    return matrix, matrix_names


def calculate_vif(matrix, names):
    results = []

    for index, name in enumerate(names):
        y = matrix[:, index]

        other_columns = [
            column
            for column in range(matrix.shape[1])
            if column != index
        ]

        if not other_columns:
            vif_value = 1.0
        else:
            x = matrix[:, other_columns]

            x_with_intercept = np.column_stack(
                [
                    np.ones(x.shape[0]),
                    x,
                ]
            )

            coefficients = np.linalg.lstsq(
                x_with_intercept,
                y,
                rcond=None,
            )[0]

            predicted = x_with_intercept @ coefficients

            residual_sum_squares = np.sum(
                (y - predicted) ** 2
            )

            total_sum_squares = np.sum(
                (y - np.mean(y)) ** 2
            )

            if total_sum_squares == 0:
                vif_value = float("inf")
            else:
                r_squared = (
                    1
                    - residual_sum_squares
                    / total_sum_squares
                )

                if r_squared >= 1:
                    vif_value = float("inf")
                else:
                    vif_value = 1 / (1 - r_squared)

        results.append(
            {
                "predictor": name,
                "vif": vif_value,
            }
        )

    return pl.DataFrame(results)


model1_matrix, model1_names = build_design_matrix(
    common_sample,
    include_bank_account=False,
)

model2_matrix, model2_names = build_design_matrix(
    common_sample,
    include_bank_account=True,
)

model1_vif = calculate_vif(
    model1_matrix,
    model1_names,
).with_columns(
    pl.lit("Model 1").alias("model")
)

model2_vif = calculate_vif(
    model2_matrix,
    model2_names,
).with_columns(
    pl.lit("Model 2").alias("model")
)

vif_results = pl.concat(
    [
        model1_vif,
        model2_vif,
    ]
)

vif_results.write_csv(
    output_dir / "vif_results.csv"
)

print(vif_results)

print("\n8. MODEL 1 AND MODEL 2 COMMON SAMPLE COMPARISON")
print("-" * 70)

comparison_variables = [
    "firm_age",
    "log_employees",
    "female_owner",
    "bank_account",
]

comparison_rows = []

for variable in comparison_variables:
    if variable not in common_sample.columns:
        continue

    values = common_sample.get_column(variable)

    comparison_rows.append(
        {
            "variable": variable,
            "observations": values.len(),
            "mean": values.mean(),
            "standard_deviation": values.std(),
            "minimum": values.min(),
            "maximum": values.max(),
        }
    )

predictor_summary = pl.DataFrame(comparison_rows)

predictor_summary.write_csv(
    output_dir / "common_sample_predictor_summary.csv"
)

print(predictor_summary)

print("\n9. WEIGHTED AND UNWEIGHTED OUTCOME RATES")
print("-" * 70)

outcome_rates = (
    common_sample
    .group_by("major_obstacle")
    .agg(
        pl.len().alias("observations"),
        pl.col("wmedian").sum().alias("weight_total"),
    )
    .sort("major_obstacle")
)

outcome_rates.write_csv(
    output_dir / "common_sample_outcome_distribution.csv"
)

weighted_rate = (
    common_sample
    .select(
        (
            (
                pl.col("major_obstacle")
                * pl.col("wmedian")
            ).sum()
            / pl.col("wmedian").sum()
        ).alias("weighted_rate")
    )
)

print(outcome_rates)
print(weighted_rate)

print("\n10. DIAGNOSTIC SUMMARY")
print("-" * 70)

zero_cells = sparse_audit.filter(
    pl.col("zero_major_cases")
    | pl.col("zero_non_major_cases")
)

high_vif = vif_results.filter(
    pl.col("vif") >= 5
)

print(f"Common sample observations: {common_sample.height:,}")
print(f"Industry or region zero cells: {zero_cells.height:,}")
print(f"Predictors with VIF >= 5: {high_vif.height:,}")

print("\nDiagnostic files saved to:")
print(output_dir)

print("\nDIAGNOSTICS COMPLETED")
print("=" * 70)