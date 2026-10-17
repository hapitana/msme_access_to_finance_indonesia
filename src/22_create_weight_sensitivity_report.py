
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_dir = (
    project_root / "data" / "processed"
)
input_path = data_dir / "weight_sensitivity_summary.csv"
output_path = data_dir / "weight_sensitivity_report.csv"

df = pl.read_csv(input_path)

report = (
    df.with_columns(
        (
            pl.col("weighted_pct")
            - pl.col("unweighted_pct")
        ).round(2).alias("difference_vs_unweighted_pp")
    )
    .select(
        "weight",
        "observations",
        "unweighted_pct",
        "weighted_pct",
        "difference_vs_unweighted_pp",
    )
)

print("=" * 65)
print("PORTFOLIO REPORT: WEIGHT SENSITIVITY")
print("=" * 65)
print(report)

report.write_csv(output_path)

print(f"\nReport saved to: {output_path}")
