from pathlib import Path
import pyreadstat

# 1. Define dataset location

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

# 2. Read Stata metadata

_, metadata = pyreadstat.read_dta(
    str(data_path),
    metadataonly=True
)

# 3. Search for finance-related variables

keywords = [
    "finance",
    "credit",
    "loan",
    "bank",
    "borrow",
    "capital",
    "fund",
    "collateral",
    "interest",
    "payment",
    "digital",
    "informal",
    "technology"
]

print("FINANCE-RELATED VARIABLES")
print()

for column in metadata.column_names:
    label = metadata.column_names_to_labels.get(column, "")

    text = f"{column} {label}".lower()

    if any(keyword in text for keyword in keywords):
        print(f"{column}: {label}")

print("\nVariable search completed.")