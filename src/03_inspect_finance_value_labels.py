
from pathlib import Path
import pyreadstat

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

_, metadata = pyreadstat.read_dta(
    str(data_path),
    metadataonly=True
)

target_variables = [
    "k82",
    "k162",
    "k20a1",
    "k17",
    "k30",
    "k6",
    "k3a",
    "k3bc",
    "k3e",
    "k3f",
    "b1",
]

variable_labels = metadata.column_names_to_labels
value_labels = metadata.variable_value_labels

print("FINANCE VARIABLES: VALUE LABEL INSPECTION")
print()

for variable in target_variables:
    label = variable_labels.get(variable, "No variable label available")
    print(f"{variable}: {label}")

    labels = value_labels.get(variable)

    if labels:
        for code, description in labels.items():
            print(f"    {code} = {description}")
    else:
        print("    No value labels found in metadata.")

    print()

print("Value label inspection completed.")
