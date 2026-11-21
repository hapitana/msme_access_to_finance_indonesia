from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

inventory_path = processed_dir / "explanatory_variable_inventory.csv"
frequency_path = processed_dir / "explanatory_variable_frequencies.csv"

inventory = pl.read_csv(inventory_path, infer_schema_length=10000)
frequencies = pl.read_csv(frequency_path, infer_schema_length=10000)

selected_variables = [
    "k30",
    "k6",
    "k82",
    "k7",
    "b4",
    "b4a",
    "b5",
    "l1",
    "b7a",
    "d3a",
    "d3b",
    "d3c",
]

print("=" * 78)
print("WBES INDONESIA 2023: SELECTED PREDICTOR DISTRIBUTION AUDIT")
print("=" * 78)

print("\n1. VARIABLE INVENTORY")

inventory_selected = inventory.filter(
    pl.col("variable").is_in(selected_variables)
)

print(inventory_selected)

print("\n2. FREQUENCY DISTRIBUTIONS")

frequency_selected = frequencies.filter(
    pl.col("variable").is_in(selected_variables)
).sort(
    ["variable", "value"]
)

for variable in selected_variables:
    print("\n" + "-" * 78)
    print(f"VARIABLE: {variable}")
    print("-" * 78)

    variable_frequency = frequency_selected.filter(
        pl.col("variable") == variable
    )

    if variable_frequency.height == 0:
        print("No frequency records found.")
    else:
        print(variable_frequency)

print("\n3. PERCENTAGE VARIABLE RANGE AUDIT")

percentage_variables = [
    "b4a",
    "d3a",
    "d3b",
    "d3c",
]

for variable in percentage_variables:
    variable_frequency = frequency_selected.filter(
        pl.col("variable") == variable
    )

    print("\n" + "-" * 78)
    print(f"VARIABLE: {variable}")
    print("-" * 78)

    if variable_frequency.height == 0:
        print("No frequency records found.")
        continue

    print(variable_frequency)

    numeric_values = variable_frequency.select(
        pl.col("value").cast(pl.Float64, strict=False)
    ).to_series()

    valid_values = numeric_values.filter(
        numeric_values.is_not_null()
        & (numeric_values >= 0)
    )

    if len(valid_values) > 0:
        print(f"Minimum nonnegative value: {valid_values.min()}")
        print(f"Maximum nonnegative value: {valid_values.max()}")

        outside_range = valid_values.filter(
            valid_values > 100
        )

        print(f"Values above 100: {len(outside_range)}")

print("\n4. CROSS-VARIABLE AUDIT")

cross_checks = [
    ("b4", "b4a"),
    ("d3a", "d3b"),
    ("d3a", "d3c"),
]

for first, second in cross_checks:
    first_data = frequency_selected.filter(
        pl.col("variable") == first
    )

    second_data = frequency_selected.filter(
        pl.col("variable") == second
    )

    print(f"\n{first} versus {second}")
    print(f"{first} frequency rows: {first_data.height}")
    print(f"{second} frequency rows: {second_data.height}")

print("\n5. OUTPUT")

output_path = processed_dir / "selected_predictor_distribution_audit.csv"

frequency_selected.write_csv(output_path)

print(f"Saved: {output_path}")

print("\nSelected predictor distribution audit completed.")