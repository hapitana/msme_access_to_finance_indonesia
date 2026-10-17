
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

keywords = [
    "size",
    "employee",
    "employment",
    "sales",
    "industry",
    "sector",
    "establishment",
    "region",
    "firm",
    "enterprise",
    "micro",
    "small",
    "medium",
    "classification",
]

print("FIRM CHARACTERISTICS: VARIABLE DICTIONARY")
print("=" * 80)

for name in metadata.column_names:
    label = metadata.column_names_to_labels.get(name, "")
    searchable_text = f"{name} {label}".lower()

    if any(keyword in searchable_text for keyword in keywords):
        print(f"{name}: {label}")

print("\nInspection completed.")
