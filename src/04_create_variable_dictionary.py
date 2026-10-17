
from pathlib import Path
import csv
import pyreadstat

# Project paths
project_dir = Path(r"D:\portofolio\02_msme_access_to_finance")

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

output_dir = project_dir / "data" / "processed"
output_path = output_dir / "variable_dictionary.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

# Read metadata only — no pandas DataFrame
_, metadata = pyreadstat.read_dta(
    str(data_path),
    metadataonly=True
)

variable_names = metadata.column_names
variable_labels = metadata.column_names_to_labels
value_labels = metadata.variable_value_labels

output_dir.mkdir(parents=True, exist_ok=True)

with output_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "variable_name",
        "variable_description",
        "value_labels"
    ])

    for variable in variable_names:
        description = variable_labels.get(
            variable, ""
        )

        labels = value_labels.get(variable, {})

        label_text = "; ".join(
            f"{code} = {meaning}"
            for code, meaning in labels.items()
        )

        writer.writerow([
            variable,
            description,
            label_text
        ])

print("Variable dictionary created successfully.")
print(f"Total variables: {len(variable_names)}")
print(f"Saved to: {output_path}")
