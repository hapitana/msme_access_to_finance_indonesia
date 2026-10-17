
from pathlib import Path

import pyreadstat

project_root = Path(__file__).resolve().parents[1]

stata_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

_, metadata = pyreadstat.read_dta(
    str(stata_path),
    metadataonly=True
)

variables = [
    "a6a",
    "a6b",
    "a6c",
    "l1",
    "a4a",
    "a4b_v4",
    "a2",
    "a3a",
]

print("VARIABLE LABELS")
print("=" * 70)

for variable in variables:
    label = metadata.column_names_to_labels.get(
        variable,
        "Label not available"
    )
    print(f"\n{variable}: {label}")

print("\nVALUE LABELS")
print("=" * 70)

value_labels = metadata.variable_value_labels

for variable in variables:
    print(f"\n{variable}:")

    labels = value_labels.get(variable)

    if labels:
        for code, description in labels.items():
            print(f"  {code}: {description}")
    else:
        print("  No value labels available.")

print("\nInspection completed.")
