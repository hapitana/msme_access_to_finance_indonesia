
from pathlib import Path
import pyreadstat

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

print("Checking file:", data_path)
print("File exists:", data_path.exists())

if not data_path.exists():
    raise FileNotFoundError(
        f"Stata file not found: {data_path}"
    )

_, meta = pyreadstat.read_dta(
    str(data_path),
    metadataonly=True,
)

variables = [
    "a6a",
    "a6b",
    "stratificationsizecode",
    "stratificationpanelcode",
]

print("=" * 70)
print("ORIGINAL STATA LABELS")
print("=" * 70)

print("\n1. VARIABLE LABELS")

for variable in variables:
    label = meta.column_names_to_labels.get(variable)
    print(f"{variable}: {label}")

print("\n2. VALUE LABELS")

for variable in variables:
    print(f"\n{variable}")
    labels = meta.variable_value_labels.get(variable)

    if labels:
        for code, description in sorted(labels.items()):
            print(f"  {code}: {description}")
    else:
        print("  No value labels found.")

print("\nInspection completed.")
