from pathlib import Path

import polars as pl
import pyreadstat

project_root = Path(__file__).resolve().parents[1]

stata_path = (
    project_root
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)
parquet_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

if not stata_path.exists():
    raise FileNotFoundError(
        f"Stata file not found: {stata_path}"
    )

if not parquet_path.exists():
    raise FileNotFoundError(
        f"Parquet file not found: {parquet_path}"
    )

df = pl.read_parquet(parquet_path)

required_columns = [
    "a6a",
    "a6b",
    "a6c",
    "stratificationsizecode",
    "stratificationpanelcode",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )
print("=" * 70)
print("SIZE CODE CROSSWALK")
print("=" * 70)

# 1. Inspect original Stata value labels
_, metadata = pyreadstat.read_dta(
    str(stata_path),
    metadataonly=True,
)

for variable in ["a6a", "a6b", "stratificationsizecode",
                 "stratificationpanelcode"]:
    print(f"\n{'-' * 60}")
    print(f"Variable: {variable}")

    label = metadata.column_names_to_labels.get(variable)
    print("Variable label:", label)

    value_label_name = metadata.variable_to_label.get(variable)
    print("Value-label set:", value_label_name)

    if value_label_name:
        labels = metadata.value_labels.get(value_label_name, {})
        print("Value labels:", labels)

# 2. Cross-tabulate the two size variables
print("\n" + "=" * 70)
print("CROSS-TABULATION: a6a vs stratificationsizecode")
print("=" * 70)

if "a6a" in df.columns and "stratificationsizecode" in df.columns:
    cross_tab = (
        df.group_by(["a6a", "stratificationsizecode"])
        .len(name="n")
        .sort(["a6a", "stratificationsizecode"])
    )
    print(cross_tab)

# 3. Inspect size codes by panel
print("\n" + "=" * 70)
print("SIZE CODES BY PANEL")
print("=" * 70)

if all(
    c in df.columns
    for c in ["stratificationpanelcode", "stratificationsizecode"]
):
    panel_size = (
        df.group_by(
            ["stratificationpanelcode", "stratificationsizecode"]
        )
        .len(name="n")
        .sort(["stratificationpanelcode", "stratificationsizecode"])
    )
    print(panel_size)


print("\n" + "=" * 70)
print("INSPECT SIZE CODE 4-6")
print("=" * 70)

size_code = pl.col("stratificationsizecode")
print("\nColumn data types:")
print(
    df.select([
        "a6a",
        "a6b",
        "a6c",
        "stratificationsizecode",
        "stratificationpanelcode",
    ]).schema
)
for code in [4, 5, 6]:
    subset = df.filter(size_code == code)

    print(f"\nCode {code}: {subset.height} observations")

    print("a6a distribution:")
    print(
        subset.group_by("a6a")
        .len(name="n")
        .sort("a6a")
    )

    print("a6b distribution:")
    print(
        subset.group_by("a6b")
        .len(name="n")
        .sort("a6b")
    )

    print("panel distribution:")
    print(
        subset.group_by("stratificationpanelcode")
        .len(name="n")
        .sort("stratificationpanelcode")
    )

    print("Original size-related variables:")
    for variable in ["a6", "a6a", "a6b", "a6c"]:
        if variable in subset.columns:
            print(
                variable,
                subset.select(variable).unique().sort(variable)
            )


print("\n" + "=" * 70)
print("INSPECT a6c METADATA AND DISTRIBUTION")
print("=" * 70)

variable = "a6c"

print("Variable label:")
print(metadata.column_names_to_labels.get(variable))

value_label_name = metadata.variable_to_label.get(variable)
print("Value-label set:", value_label_name)

if value_label_name:
    print(
        "Value labels:",
        metadata.value_labels.get(value_label_name, {})
    )

print("\nOverall a6c summary:")
print(
    df.select([
        pl.col("a6c").count().alias("non_null"),
        pl.col("a6c").null_count().alias("null_count"),
        pl.col("a6c").min().alias("min"),
        pl.col("a6c").median().alias("median"),
        pl.col("a6c").max().alias("max"),
    ])
)

print("\na6c summary by stratification size code:")
print(
    df.group_by("stratificationsizecode")
    .agg([
        pl.len().alias("n"),
        pl.col("a6c").min().alias("min_a6c"),
        pl.col("a6c").median().alias("median_a6c"),
        pl.col("a6c").max().alias("max_a6c"),
    ])
    .sort("stratificationsizecode")
)

print("\na6c summary by sampling size:")
print(
    df.group_by("a6a")
    .agg([
        pl.len().alias("n"),
        pl.col("a6c").min().alias("min_a6c"),
        pl.col("a6c").median().alias("median_a6c"),
        pl.col("a6c").max().alias("max_a6c"),
    ])
    .sort("a6a")
)


# 7. Investigate the relationship between size variables
print("\n" + "=" * 70)
print("CROSS-TABULATION: STRATIFICATION SIZE CODE VS a6b")
print("=" * 70)

print(
    df.group_by([
        "stratificationsizecode",
        "a6b",
    ])
    .len(name="n")
    .sort([
        "stratificationsizecode",
        "a6b",
    ])
)

# 8. Inspect a6c values within exploratory size ranges
print("\n" + "=" * 70)
print("a6b VS a6c: EXPLORATORY RANGE CHECK")
print("=" * 70)

a6c_range = (
    pl.when(pl.col("a6c") < 20)
    .then(pl.lit("Below 20"))
    .when(pl.col("a6c") < 100)
    .then(pl.lit("20-99"))
    .otherwise(pl.lit("100 or more"))
    .alias("a6c_range")
)

print(
    df.with_columns(a6c_range)
    .group_by(["a6b", "a6c_range"])
    .len(name="n")
    .sort(["a6b", "a6c_range"])
)

# 9. Inspect individual observations for codes 4-6
print("\n" + "=" * 70)
print("SAMPLE RECORDS FOR SIZE CODES 4-6")
print("=" * 70)

inspection_columns = [
    "a6a",
    "a6b",
    "a6c",
    "stratificationsizecode",
    "stratificationpanelcode",
]

print(
    df.filter(
        pl.col("stratificationsizecode").is_in([4, 5, 6])
    )
    .select(inspection_columns)
    .sort([
        "stratificationsizecode",
        "a6c",
    ])
    .head(30)
)


# 10. Validate screener size classification
print("\n" + "=" * 70)
print("VALIDATION: a6b VS a6c")
print("=" * 70)

df_validated = df.with_columns(
    pl.when(pl.col("a6c") < 20)
    .then(pl.lit(1))
    .when(pl.col("a6c") < 100)
    .then(pl.lit(2))
    .otherwise(pl.lit(3))
    .alias("a6b_inferred_from_a6c")
)

print("Observed a6b vs inferred category from a6c:")
print(
    df_validated.group_by([
        "a6b",
        "a6b_inferred_from_a6c",
    ])
    .len(name="n")
    .sort([
        "a6b",
        "a6b_inferred_from_a6c",
    ])
)

mismatches = df_validated.filter(
    pl.col("a6b") != pl.col("a6b_inferred_from_a6c")
)

print("\nNumber of mismatches:", mismatches.height)

# 11. Full crosswalk diagnostic
print("\n" + "=" * 70)
print("FULL STRATIFICATION SIZE CROSSWALK")
print("=" * 70)

print(
    df.group_by([
        "stratificationsizecode",
        "a6a",
        "a6b",
        "stratificationpanelcode",
    ])
    .len(name="n")
    .sort([
        "stratificationsizecode",
        "a6a",
        "a6b",
        "stratificationpanelcode",
    ])
)

# 12. Inspect all observed stratification codes
print("\nObserved stratification size codes:")
print(
    df.group_by("stratificationsizecode")
    .len(name="n")
    .sort("stratificationsizecode")
)

print("\nInspection completed.")