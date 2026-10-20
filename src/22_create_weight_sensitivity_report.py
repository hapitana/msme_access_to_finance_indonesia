
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_dir = project_root / "data" / "processed"
input_path = data_dir / "weight_sensitivity_summary.csv"
output_path = data_dir / "weight_sensitivity_report.csv"

if not input_path.exists():
    raise FileNotFoundError(
        f"Run src/21_weight_sensitivity_check.py first: {input_path}"
    )

df = pl.read_csv(input_path)

required = {
    "weight",
    "observations",
    "unweighted_pct",
    "weighted_pct",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Required columns missing: {sorted(missing)}")

median_rows = df.filter(pl.col("weight") == "wmedian")

if median_rows.height != 1:
    raise ValueError(
        "Expected exactly one wmedian row in the sensitivity summary."
    )

median_pct = median_rows.get_column("weighted_pct").item()


report = (
    df.with_columns(
        (
            pl.col("weighted_pct") - pl.col("unweighted_pct")
        ).alias("difference_vs_unweighted_pp"),
        pl.lit(median_pct).alias("wmedian_pct"),
    )
    .with_columns(
        (
            pl.col("weighted_pct") - pl.col("wmedian_pct")
        ).alias("difference_vs_wmedian_pp")
    )
    .select(
        "weight",
        "observations",
        "unweighted_pct",
        "weighted_pct",
        "difference_vs_unweighted_pp",
        "wmedian_pct",
        "difference_vs_wmedian_pp",
    )
    .with_columns(
        pl.col("unweighted_pct").round(2),
        pl.col("weighted_pct").round(2),
        pl.col("difference_vs_unweighted_pp").round(2),
        pl.col("wmedian_pct").round(2),
        pl.col("difference_vs_wmedian_pp").round(2),
    )
)

print("=" * 70)
print("PORTFOLIO REPORT: WEIGHT SENSITIVITY")
print("=" * 70)
print("Positive differences indicate a higher estimated percentage.")
print("\nDifferences are expressed in percentage points (pp).")
print(report)

report.write_csv(output_path)

print(f"\nReport saved to: {output_path}")
