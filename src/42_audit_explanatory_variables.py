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
output_dir.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise FileNotFoundError(data_path)

data, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

df = pl.DataFrame(data)

keywords = [
    "age",
    "year",
    "start",
    "establish",
    "employee",
    "worker",
    "employment",
    "account",
    "credit",
    "loan",
    "overdraft",
    "finance",
    "ownership",
    "owner",
    "industry",
    "sector",
    "region",
    "location",
    "size",
    "sales",
    "export",
    "manager",
    "female",
]

variable_labels = metadata.column_names_to_labels
value_labels = metadata.variable_value_labels

inventory_rows = []
frequency_rows = []

for variable in df.columns:
    label = variable_labels.get(variable, "")
    label_lower = label.lower()
    variable_lower = variable.lower()

    matched_keywords = [
        keyword
        for keyword in keywords
        if keyword in label_lower
        or keyword in variable_lower
    ]

    if not matched_keywords:
        continue

    series = df[variable]

    unique_count = series.n_unique()
    missing_count = series.null_count()
    valid_count = series.count()

    labels = value_labels.get(variable, {})

    inventory_rows.append(
        {
            "variable": variable,
            "label": label,
            "matched_keywords": ", ".join(matched_keywords),
            "data_type": str(series.dtype),
            "observations": df.height,
            "valid_observations": valid_count,
            "missing_observations": missing_count,
            "unique_values": unique_count,
            "value_labels": str(labels),
        }
    )

    if unique_count <= 30:
        frequency = (
            df.group_by(variable)
            .agg(
                pl.len().alias("observations")
            )
            .sort(variable)
        )

        for row in frequency.iter_rows(named=True):
            value = row[variable]

            if value is None:
                value_label = "System missing"
            else:
                value_label = str(
                    labels.get(value, "Unlabelled value")
                )

            frequency_rows.append(
                {
                    "variable": variable,
                    "value": str(value),
                    "value_label": value_label,
                    "observations": row["observations"],
                }
            )

inventory = pl.DataFrame(
    inventory_rows,
    infer_schema_length=None,
)

if frequency_rows:
    frequencies = pl.DataFrame(
        frequency_rows,
        infer_schema_length=None,
    )
else:
    frequencies = pl.DataFrame(
        schema={
            "variable": pl.String,
            "value": pl.String,
            "value_label": pl.String,
            "observations": pl.Int64,
        }
    )

inventory_path = (
    output_dir / "explanatory_variable_inventory.csv"
)

frequency_path = (
    output_dir / "explanatory_variable_frequencies.csv"
)

inventory.write_csv(inventory_path)
frequencies.write_csv(frequency_path)

print("=" * 70)
print("WBES INDONESIA 2023: EXPLANATORY VARIABLE AUDIT")
print("=" * 70)

print("\n1. Dataset dimensions")
print("Rows:", df.height)
print("Columns:", df.width)

print("\n2. Survey design and weight variables")

design_candidates = [
    "idstd",
    "strata",
    "stratificationpanelcode",
    "a2",
    "a4a",
    "a6a",
    "a6b",
    "wmedian",
    "wweak",
    "wstrong",
]

for variable in design_candidates:
    if variable in df.columns:
        print(
            variable,
            "|",
            variable_labels.get(variable, ""),
            "| missing:",
            df[variable].null_count(),
            "| unique:",
            df[variable].n_unique(),
        )
    else:
        print(variable, "| NOT FOUND")

print("\n3. Candidate explanatory variables")
print("Number of candidates:", inventory.height)
print(inventory)

print("\n4. Candidate categorical frequencies")
print("Number of frequency records:", frequencies.height)
print(frequencies)

print("\n5. Output files")
print(inventory_path)
print(frequency_path)

print("\nAudit completed.")