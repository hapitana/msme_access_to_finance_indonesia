from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path)

weights = ["wstrict", "wmedian", "wweak"]

required = ["k30", *weights]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Required columns not found: {missing}")

# Define the outcome:
# 1 = major or very severe obstacle
# 0 = no, minor, or moderate obstacle
analysis = df.with_columns(
    pl.when(pl.col("k30").is_in([3, 4]))
    .then(1)
    .when(pl.col("k30").is_in([0, 1, 2]))
    .then(0)
    .otherwise(None)
    .alias("major_obstacle")
)

# Keep valid outcome responses and a common set of valid weights.
valid = analysis.filter(
    pl.col("major_obstacle").is_not_null()
    & pl.all_horizontal([
        pl.col(weight).is_not_null()
        & pl.col(weight).is_finite()
        & (pl.col(weight) > 0)
        for weight in weights
    ])
)

if valid.height == 0:
    raise ValueError("No valid observations remain in the analysis sample.")

unweighted_rate = valid.get_column("major_obstacle").mean()
summary_rows = []

for weight in weights:
    result = valid.select(
        pl.len().alias("observations"),
        pl.col(weight).sum().alias("weight_sum"),
        (
            (pl.col("major_obstacle") * pl.col(weight)).sum()
            / pl.col(weight).sum()
        ).alias("weighted_rate"),
    ).row(0, named=True)

    summary_rows.append({
        "weight": weight,
        "observations": result["observations"],
        "unweighted_pct": unweighted_rate * 100,
        "weighted_pct": result["weighted_rate"] * 100,
        "weight_sum": result["weight_sum"],
    })


summary = pl.DataFrame(summary_rows)

# Round only for console display.
display_summary = summary.with_columns(
    pl.col("unweighted_pct").round(2),
    pl.col("weighted_pct").round(2),
)

print("=" * 65)
print("WEIGHT SENSITIVITY CHECK: FINANCE OBSTACLE")
print("=" * 65)
print(f"Original observations: {df.height:,}")
print(f"Analysis sample: {valid.height:,}")
print(f"Excluded observations: {df.height - valid.height:,}")
print("Valid outcome codes: 0, 1, 2, 3, 4")
print("Outcome: k30 = 3 or 4")

print("\nOVERALL ESTIMATES")
print(display_summary)

output_path = output_dir / "weight_sensitivity_summary.csv"

# Preserve full precision for downstream calculations.
summary.write_csv(output_path)

print(f"\nSaved to: {output_path}")
