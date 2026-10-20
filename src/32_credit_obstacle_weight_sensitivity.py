
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
required = ["k82", "k30", *weights]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Required columns not found: {missing}")

# Define valid credit-status groups and the binary obstacle outcome.
analysis = df.with_columns(
    pl.when(pl.col("k82").is_in([1, 2, 3]))
    .then(pl.lit("Has credit"))
    .when(pl.col("k82") == 4)
    .then(pl.lit("No credit"))
    .otherwise(None)
    .alias("credit_status"),

    pl.when(pl.col("k30").is_in([3, 4]))
    .then(1)
    .when(pl.col("k30").is_in([0, 1, 2]))
    .then(0)
    .otherwise(None)
    .alias("major_obstacle"),
)

# Use one common complete-case sample for all weight specifications.

valid = analysis.filter(
    pl.col("credit_status").is_not_null()
    & pl.col("major_obstacle").is_not_null()
    & pl.all_horizontal([
        pl.col(weight).is_not_null()
        & pl.col(weight).is_finite()
        & (pl.col(weight) > 0)
        for weight in weights
    ])
)

if valid.height == 0:
    raise ValueError("No valid observations remain in the common sample.")

expected_groups = {"Has credit", "No credit"}
observed_groups = set(valid.get_column("credit_status").unique().to_list())

if observed_groups != expected_groups:
    raise ValueError(
        f"Expected both credit-status groups; found: {observed_groups}"
    )

detail_rows = []
comparison_rows = []

for weight in weights:
    group_estimates = {}

    for status in ["Has credit", "No credit"]:
        group = valid.filter(pl.col("credit_status") == status)

        result = group.select(
            pl.len().alias("sample_count"),
            pl.col("major_obstacle").mean().alias("unweighted_rate"),
            (
                (pl.col("major_obstacle") * pl.col(weight)).sum()
                / pl.col(weight).sum()
            ).alias("weighted_rate"),
            pl.col(weight).sum().alias("weight_sum"),
        ).row(0, named=True)

        if result["weight_sum"] <= 0:
            raise ValueError(
                f"Non-positive total weight for {status} using {weight}"
            )

        unweighted_pct = result["unweighted_rate"] * 100
        weighted_pct = result["weighted_rate"] * 100

        group_estimates[status] = {
            "unweighted_pct": unweighted_pct,
            "weighted_pct": weighted_pct,
        }

        detail_rows.append({
            "weight": weight,
            "credit_status": status,
            "sample_count": result["sample_count"],
            "unweighted_major_severe_pct": unweighted_pct,
            "weighted_major_severe_pct": weighted_pct,
            "weight_sum": result["weight_sum"],
        })

    has_credit = group_estimates["Has credit"]
    no_credit = group_estimates["No credit"]

    comparison_rows.append({
        "weight": weight,
        "has_credit_pct": has_credit["weighted_pct"],
        "no_credit_pct": no_credit["weighted_pct"],
        "weighted_difference_pp": (
            has_credit["weighted_pct"] - no_credit["weighted_pct"]
        ),
        "unweighted_difference_pp": (
            has_credit["unweighted_pct"] - no_credit["unweighted_pct"]
        ),
    })

detail = pl.DataFrame(detail_rows).with_columns(
    pl.col("unweighted_major_severe_pct").round(2),
    pl.col("weighted_major_severe_pct").round(2),
    pl.col("weight_sum").round(2),
)

comparison = pl.DataFrame(comparison_rows).with_columns(
    pl.col("has_credit_pct").round(2),
    pl.col("no_credit_pct").round(2),
    pl.col("weighted_difference_pp").round(2),
    pl.col("unweighted_difference_pp").round(2),
)

print("=" * 70)
print("CREDIT-STATUS WEIGHT SENSITIVITY ANALYSIS")
print("=" * 70)
print(f"Original observations: {df.height:,}")
print(f"Common analysis sample: {valid.height:,}")
print(f"Excluded observations: {df.height - valid.height:,}")
print("Outcome: k30 = 3 or 4")
print("Credit status: k82 = 1, 2, 3 versus k82 = 4")

print("\nGROUP ESTIMATES")
print(detail)

print("\nBETWEEN-GROUP COMPARISON")
print("Difference = Has credit minus No credit, in percentage points")
print(comparison)

detail_path = (
    output_dir / "credit_obstacle_weight_sensitivity_detail.csv"
)
comparison_path = (
    output_dir / "credit_obstacle_weight_sensitivity_comparison.csv"
)

detail.write_csv(detail_path)
comparison.write_csv(comparison_path)

print(f"\nSaved: {detail_path}")
print(f"Saved: {comparison_path}")
print("Sensitivity analysis completed.")