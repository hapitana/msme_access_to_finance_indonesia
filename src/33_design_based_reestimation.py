from pathlib import Path

import polars as pl
import pyreadstat

project_dir = Path(__file__).resolve().parent.parent

data_path = (
    project_dir
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

output_dir = project_dir / "data" / "processed"

print(f"Project directory: {project_dir}")
print(f"Dataset path: {data_path}")
print(f"Dataset exists: {data_path.exists()}")

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

output_dir.mkdir(parents=True, exist_ok=True)

required_variables = [
    "idstd",
    "strata",
    "wmedian",
    "wweak",
    "wstrong",
    "a4a",
    "a2",
    "a6a",
    "panel",
]

weight_variables = {
    "wmedian",
    "wweak",
    "wstrong",
}

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

output_dir.mkdir(parents=True, exist_ok=True)

print("Reading Stata dataset")

data_dict, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

df = pl.DataFrame(
    {
        column: values
        for column, values in data_dict.items()
    }
)

del data_dict

variable_labels = metadata.column_names_to_labels

print("\nDataset audit")
print(f"File: {data_path.name}")
print(f"Rows: {df.height}")
print(f"Columns: {df.width}")
print(f"Polars version: {pl.__version__}")

available_variables = set(df.columns)
variable_report = []

for variable in required_variables:
    exists = variable in available_variables

    row = {
        "variable": variable,
        "exists": exists,
        "label": (
            variable_labels.get(variable, "")
            if exists
            else ""
        ),
    }

    if exists:
        series = df.get_column(variable)

        row["missing_count"] = series.null_count()
        row["unique_nonmissing"] = series.n_unique()

        if variable in weight_variables:
            numeric = series.cast(
                pl.Float64,
                strict=False,
            )
            valid = numeric.drop_nulls()

            row["zero_count"] = int(
                (valid == 0).sum()
            )
            row["negative_count"] = int(
                (valid < 0).sum()
            )
            row["minimum"] = valid.min()
            row["median"] = valid.median()
            row["maximum"] = valid.max()

        if variable == "idstd":
            nonmissing = df.filter(
                pl.col("idstd").is_not_null()
            ).get_column("idstd")

            row["duplicate_nonmissing"] = (
                nonmissing.len() - nonmissing.n_unique()
            )

    variable_report.append(row)

report = pl.DataFrame(variable_report)

report_path = (
    output_dir
    / "survey_design_variable_audit.csv"
)
report.write_csv(report_path)

print("\nVariable report")
print(report)

if "strata" in available_variables:
    strata_sizes = (
        df.group_by("strata")
        .agg(pl.len().alias("observations"))
        .sort("observations", "strata")
    )

    nonmissing_strata = df.filter(
        pl.col("strata").is_not_null()
    )

    singleton_count = strata_sizes.filter(
        pl.col("strata").is_not_null()
        & (pl.col("observations") == 1)
    ).height

    missing_strata_count = df.filter(
        pl.col("strata").is_null()
    ).height

    print("\nStrata checks")
    print(
        "Distinct non-missing strata: "
        f"{nonmissing_strata['strata'].n_unique()}"
    )
    print(f"Singleton strata: {singleton_count}")
    print(
        "Observations with missing strata: "
        f"{missing_strata_count}"
    )

    strata_sizes_path = (
        output_dir
        / "survey_design_strata_sizes.csv"
    )
    strata_sizes.write_csv(strata_sizes_path)

    print(f"Saved strata sizes: {strata_sizes_path}")

if "idstd" in available_variables:
    id_series = df.get_column("idstd")
    nonmissing_ids = df.filter(
        pl.col("idstd").is_not_null()
    ).get_column("idstd")

    print("\nID checks")
    print(f"Missing IDs: {id_series.null_count()}")
    print(
        "Duplicate nonmissing IDs: "
        f"{nonmissing_ids.len() - nonmissing_ids.n_unique()}"
    )

print("\nRequired variable availability")

for variable in required_variables:
    status = (
        "PRESENT"
        if variable in available_variables
        else "MISSING"
    )
    print(f"{variable}: {status}")

print("\nAudit completed")
print(f"Saved variable report: {report_path}")