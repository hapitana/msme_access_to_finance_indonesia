from pathlib import Path

import polars as pl

project_dir = Path(__file__).resolve().parent.parent

processed_dir = project_dir / "data" / "processed"

inventory_path = (
    processed_dir / "explanatory_variable_inventory.csv"
)

frequency_path = (
    processed_dir / "explanatory_variable_frequencies.csv"
)

if not inventory_path.exists():
    raise FileNotFoundError(inventory_path)

if not frequency_path.exists():
    raise FileNotFoundError(frequency_path)

inventory = pl.read_csv(inventory_path)
frequencies = pl.read_csv(frequency_path)

groups = {
    "firm_age": [
        "year established",
        "year establishment",
        "started operations",
        "start of operations",
        "established",
        "years in operation",
        "firm age",
    ],
    "employment": [
        "permanent full-time workers",
        "full-time employees",
        "number of employees",
        "number of workers",
        "employment",
        "temporary workers",
    ],
    "bank_account": [
        "checking account",
        "savings account",
        "bank account",
        "current account",
    ],
    "credit_access": [
        "credit line",
        "line of credit",
        "overdraft",
        "loan",
        "borrowed",
        "credit",
        "financing",
    ],
    "ownership": [
        "foreign ownership",
        "foreign owned",
        "female ownership",
        "female participation",
        "female owner",
        "ownership",
    ],
    "export": [
        "export",
        "exporter",
    ],
    "management": [
        "female top manager",
        "female manager",
        "top manager",
        "manager gender",
    ],
}

selected_rows = []

for row in inventory.iter_rows(named=True):
    variable = row["variable"]
    label = str(row["label"] or "").lower()

    matched_groups = []

    for group_name, terms in groups.items():
        if any(term in label for term in terms):
            matched_groups.append(group_name)

    if matched_groups:
        selected_rows.append(
            {
                "variable": variable,
                "label": row["label"],
                "groups": ", ".join(matched_groups),
                "valid_observations": row["valid_observations"],
                "missing_observations": row["missing_observations"],
                "unique_values": row["unique_values"],
                "value_labels": row["value_labels"],
            }
        )

selected = pl.DataFrame(
    selected_rows,
    infer_schema_length=None,
)

print("=" * 70)
print("WBES INDONESIA 2023: MODEL VARIABLE SELECTION")
print("=" * 70)

print("\n1. Selected explanatory variables")
print("Number of selected variables:", selected.height)
print(selected)

selected_names = selected["variable"].to_list()

print("\n2. Categorical frequency distributions")

selected_frequencies = frequencies.filter(
    pl.col("variable").is_in(selected_names)
)

print(selected_frequencies)

print("\n3. Core sampling variables")

core_names = ["a2", "a4a", "a6a", "a6b"]

core_variables = inventory.filter(
    pl.col("variable").is_in(core_names)
)

print(core_variables)

print("\n4. Outcome variable")

outcome = inventory.filter(
    pl.col("variable") == "k30"
)

print(outcome)

selected_path = (
    processed_dir / "selected_model_variables.csv"
)

selected_frequency_path = (
    processed_dir / "selected_model_variable_frequencies.csv"
)

selected.write_csv(selected_path)
selected_frequencies.write_csv(selected_frequency_path)

print("\n5. Output files")
print(selected_path)
print(selected_frequency_path)

print("\nVariable selection audit completed.")