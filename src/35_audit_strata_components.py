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

print("=" * 65)
print("WBES INDONESIA 2023: STRATA COMPONENT AUDIT")
print("=" * 65)

print(f"\nDataset rows: {df.height}")
print(f"Dataset columns: {df.width}")

component_columns = [
    "a4a",
    "a2",
    "a6a",
    "stratificationpanelcode",
]

required_columns = ["strata"] + component_columns

print("\n1. Variable availability")

for column in required_columns:
    if column in df.columns:
        label = metadata.column_names_to_labels.get(
            column,
            "No variable label",
        )
        missing = df[column].null_count()
        unique = df[column].n_unique()

        print(
            f"{column}: FOUND | "
            f"Missing={missing} | "
            f"Unique={unique} | "
            f"Label={label}"
        )
    else:
        print(f"{column}: NOT FOUND")

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required variables are missing: {missing_columns}"
    )

print("\n2. Panel variable distribution")

panel_distribution = (
    df.group_by("stratificationpanelcode")
    .agg(pl.len().alias("observations"))
    .sort("stratificationpanelcode")
)

print(panel_distribution)

print("\n3. Missing values in strata components")

missing_report = (
    df.select(
        [
            pl.col(column)
            .null_count()
            .alias(column)
            for column in required_columns
        ]
    )
)

print(missing_report)

missing_component_rows = df.filter(
    pl.any_horizontal(
        [
            pl.col(column).is_null()
            for column in required_columns
        ]
    )
)

print(
    "Rows with at least one missing component:",
    missing_component_rows.height,
)

print("\n4. Test: one component combination maps to one strata code")

combo_to_strata = (
    df.group_by(component_columns)
    .agg(
        pl.col("strata")
        .n_unique()
        .alias("n_strata_codes"),
        pl.len().alias("n_observations"),
    )
)

combo_conflicts = combo_to_strata.filter(
    pl.col("n_strata_codes") != 1
)

print("Distinct component combinations:", combo_to_strata.height)
print("Conflicting combinations:", combo_conflicts.height)

print("\n5. Test: one strata code maps to one component combination")

strata_to_combo = (
    df.group_by("strata")
    .agg(
        pl.struct(component_columns)
        .n_unique()
        .alias("n_combinations"),
        pl.len().alias("n_observations"),
    )
)

strata_conflicts = strata_to_combo.filter(
    pl.col("n_combinations") != 1
)

print("Distinct strata codes:", strata_to_combo.height)
print("Strata codes with conflicting combinations:", strata_conflicts.height)

print("\n6. Save audit outputs")

combo_to_strata.write_csv(
    output_dir / "strata_component_mapping.csv"
)

strata_to_combo.write_csv(
    output_dir / "strata_reverse_mapping.csv"
)

combo_conflicts.write_csv(
    output_dir / "strata_component_conflicts.csv"
)

strata_conflicts.write_csv(
    output_dir / "strata_reverse_conflicts.csv"
)

missing_component_rows.write_csv(
    output_dir / "strata_missing_component_rows.csv"
)

print("Audit files saved to:", output_dir)

print("\n7. Audit conclusion")

if missing_component_rows.height > 0:
    print("WARNING: Some strata components are missing.")

if combo_conflicts.height == 0 and strata_conflicts.height == 0:
    print(
        "PASS: The observed mapping between component "
        "combinations and strata codes is one-to-one."
    )
else:
    print(
        "WARNING: The observed mapping is not one-to-one. "
        "Review the conflict files."
    )

print(
    "\nThis audit checks internal consistency only. "
    "It does not independently validate the sampling design "
    "or survey variance estimates."
)

print("\nAudit completed.")