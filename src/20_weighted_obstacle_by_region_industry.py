
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "model_dataset.parquet"
)

output_dir = project_root / "data" / "processed"

df = pl.read_parquet(data_path)

region_labels = {
    1: "Jakarta",
    2: "West Java",
    3: "East Java",
    4: "Central Java",
    5: "Banten",
    6: "Yogyakarta",
    7: "Bali",
    8: "East Nusa Tenggara",
    9: "West Nusa Tenggara",
    10: "Riau",
    11: "West Sumatra",
    12: "Aceh",
    13: "Lampung",
    14: "South Sumatra / Bengkulu / Jambi",
    15: "Bangka Belitung / Riau Islands",
    16: "North Sumatra",
    17: "West Kalimantan",
    18: "South / Central Kalimantan",
    19: "East / North Kalimantan",
    20: "Sulawesi",
    21: "Papua",
    22: "Maluku / North Maluku",
}

industry_labels = {
    1: "Food",
    2: "Textiles",
    3: "Garments",
    4: "Leather products",
    5: "Chemicals",
    6: "Rubber and plastics",
    7: "Non-metallic minerals",
    8: "Fabricated metals",
    9: "Machinery and equipment",
    10: "Motor vehicles",
    11: "Other manufacturing",
    12: "Retail",
    13: "Construction",
    14: "Transport",
    15: "Hotels",
    16: "Restaurants",
    17: "ICT",
    18: "Professional services",
    19: "Other services",
}

def summarize_by(df, group_column, labels):
    result = (
        df.group_by(group_column)
        .agg(
            pl.len().alias("observations"),
            pl.col("major_obstacle").sum().alias("major_obstacle_count"),
            pl.col("major_obstacle").mean().alias("unweighted_rate"),
            (
                (
                    pl.col("major_obstacle")
                    * pl.col("survey_weight")
                ).sum()
                / pl.col("survey_weight").sum()
            ).alias("weighted_rate"),
            pl.col("survey_weight").sum().alias("weight_sum"),
        )
        .with_columns(
            pl.col(group_column)
            .replace_strict(labels, default="Unknown")
            .alias("category")
        )
        .with_columns(
            (pl.col("unweighted_rate") * 100).round(2)
            .alias("unweighted_pct"),
            (pl.col("weighted_rate") * 100).round(2)
            .alias("weighted_pct"),
        )
        .sort("weighted_pct", descending=True)
    )
    return result

region_result = summarize_by(df, "region_code", region_labels)
industry_result = summarize_by(df, "industry_code", industry_labels)

print("=" * 65)
print("WEIGHTED OBSTACLE RATES BY REGION")
print("=" * 65)
print(
    region_result.select(
        "category",
        "observations",
        "major_obstacle_count",
        "unweighted_pct",
        "weighted_pct",
    )
)

print("\n" + "=" * 65)
print("WEIGHTED OBSTACLE RATES BY INDUSTRY")
print("=" * 65)
print(
    industry_result.select(
        "category",
        "observations",
        "major_obstacle_count",
        "unweighted_pct",
        "weighted_pct",
    )
)

region_result.write_csv(
    output_dir / "weighted_obstacle_by_region.csv"
)
industry_result.write_csv(
    output_dir / "weighted_obstacle_by_industry.csv"
)

print("\nSaved weighted regional and industry tables.")
