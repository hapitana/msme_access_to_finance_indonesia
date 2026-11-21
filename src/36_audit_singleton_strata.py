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

required_columns = [
    "idstd",
    "strata",
    "stratificationpanelcode",
    "wmedian",
    "wweak",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required variables are missing: {missing_columns}"
    )

print("=" * 65)
print("WBES INDONESIA 2023: SINGLETON STRATA AUDIT")
print("=" * 65)

print("\n1. Strata size distribution")

strata_sizes = (
    df.group_by("strata")
    .agg(
        pl.len().alias("strata_size"),
        pl.col("stratificationpanelcode")
        .n_unique()
        .alias("panel_categories"),
    )
)

size_distribution = (
    strata_sizes
    .with_columns(
        pl.when(pl.col("strata_size") == 1)
        .then(pl.lit("singleton"))
        .when(pl.col("strata_size") == 2)
        .then(pl.lit("two_units"))
        .otherwise(pl.lit("three_or_more"))
        .alias("size_category")
    )
    .group_by("size_category")
    .agg(
        pl.len().alias("number_of_strata"),
        pl.col("strata_size").sum().alias("observations"),
    )
    .sort("size_category")
)

print(size_distribution)

singleton_strata = strata_sizes.filter(
    pl.col("strata_size") == 1
)

print("\n2. Singleton strata count")
print("Singleton strata:", singleton_strata.height)

print("\n3. Singleton strata by panel category")

singleton_details = (
    df.join(
        singleton_strata.select("strata"),
        on="strata",
        how="inner",
    )
    .group_by("stratificationpanelcode")
    .agg(
        pl.len().alias("singleton_observations"),
        pl.col("strata").n_unique().alias("singleton_strata"),
    )
    .sort("stratificationpanelcode")
)

print(singleton_details)

print("\n4. All strata by panel category")

panel_strata_summary = (
    df.group_by(
        "stratificationpanelcode",
        "strata",
    )
    .agg(
        pl.len().alias("strata_size")
    )
    .with_columns(
        (pl.col("strata_size") == 1)
        .alias("is_singleton")
    )
    .group_by("stratificationpanelcode")
    .agg(
        pl.len().alias("total_strata"),
        pl.col("is_singleton")
        .sum()
        .alias("singleton_strata"),
        pl.col("strata_size")
        .sum()
        .alias("observations"),
    )
    .sort("stratificationpanelcode")
)

print(panel_strata_summary)

print("\n5. Sampling unit and weight checks")

print(
    "Unique idstd:",
    df["idstd"].n_unique(),
)

print(
    "Missing idstd:",
    df["idstd"].null_count(),
)

for weight in ["wmedian", "wweak"]:
    print(f"\n{weight}")

    print(
        "Missing:",
        df[weight].null_count(),
    )

    print(
        "Minimum:",
        df[weight].min(),
    )

    print(
        "Non-positive:",
        df.filter(
            pl.col(weight).is_not_null()
            & (pl.col(weight) <= 0)
        ).height,
    )

print("\n6. Save detailed audit outputs")

strata_sizes.write_csv(
    output_dir / "strata_size_details.csv"
)

singleton_strata.write_csv(
    output_dir / "singleton_strata.csv"
)

singleton_details.write_csv(
    output_dir / "singleton_strata_by_panel.csv"
)

panel_strata_summary.write_csv(
    output_dir / "strata_summary_by_panel.csv"
)

print("Audit outputs saved to:", output_dir)

print("\nAudit completed.")