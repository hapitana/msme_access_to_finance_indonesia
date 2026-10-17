from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

df = pl.read_parquet(data_path)

print("FIRM SIZE CODE DISTRIBUTION")
print("=" * 60)
print(f"Total observations: {df.height:,}")

# Display all firm size codes, including nulls
distribution = (
    df.group_by("a6b")
    .agg(
        pl.len().alias("sample_count"),
        pl.col("wmedian").sum().alias("weight_sum"),
    )
    .with_columns(
        (
            pl.col("sample_count")
            / df.height
            * 100
        )
        .round(2)
        .alias("sample_percent")
    )
    .sort("a6b", nulls_last=True)
)

print(distribution)

print("\nFIRM SIZE LABELS")
print("=" * 60)

labels = {
    0: "Micro",
    1: "Small",
    2: "Medium",
    3: "Large",
}

for code, label in labels.items():
    count = df.filter(pl.col("a6b") == code).height
    print(f"{code} - {label}: {count:,}")

print("\nNull size values:", df["a6b"].null_count())
print("Inspection completed.")
