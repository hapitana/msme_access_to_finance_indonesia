from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parents[1]
processed_dir = project_dir / "data" / "processed"
output_dir = (
    processed_dir
    / "outcome_robustness"
    / "24c19_geographic_label_audit"
)

output_dir.mkdir(parents=True, exist_ok=True)

data_path = processed_dir / "indonesia_2023.parquet"
coefficient_path = (
    processed_dir
    / "outcome_robustness"
    / "24c17_geographic_sensitivity_coefficients.csv"
)

region_labels = {
    1: "Special Capital Region of Jakarta",
    2: "West Java",
    3: "East Java",
    4: "Central Java",
    5: "Banten",
    6: "Special Region of Yogyakarta",
    7: "Bali",
    8: "East Nusa Tenggara",
    9: "West Nusa Tenggara",
    10: "Riau",
    11: "West Sumatra",
    12: "Aceh",
    13: "Lampung",
    14: "South Sumatra, Bengkulu, and Jambi",
    15: "Bangka Belitung Islands and Riau Islands",
    16: "North Sumatra",
    17: "West Kalimantan",
    18: "South Kalimantan and Central Kalimantan",
    19: "East Kalimantan and North Kalimantan",
    20: "Sulawesi",
    21: "Papua",
    22: "Maluku Islands",
}

official_counts = {
    1: 172,
    2: 231,
    3: 140,
    4: 141,
    5: 145,
    6: 122,
    7: 131,
    8: 122,
    9: 124,
    10: 125,
    11: 121,
    12: 117,
    13: 121,
    14: 142,
    15: 147,
    16: 135,
    17: 132,
    18: 125,
    19: 73,
    20: 134,
    21: 119,
    22: 136,
}

print("=" * 72)
print("24C-19 GEOGRAPHIC REGION LABEL AUDIT")
print("=" * 72)

if not data_path.exists():
    raise FileNotFoundError(data_path)

data = pl.read_parquet(data_path)

required_columns = [
    "idstd",
    "a3a",
    "k30",
    "wweak",
    "wmedian",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if data.select(pl.col("idstd").n_unique()).item() != data.height:
    raise ValueError("Duplicate idstd values detected.")

data = data.with_columns(
    pl.col("a3a").cast(pl.Int64, strict=False),
    pl.col("k30").cast(pl.Float64, strict=False),
)

invalid_regions = data.filter(
    pl.col("a3a").is_null()
    | ~pl.col("a3a").is_in(list(region_labels.keys()))
)

if invalid_regions.height > 0:
    raise ValueError(
        f"Invalid geographic codes detected: {invalid_regions.height}"
    )

allowed_k30_values = [
    -9.0,
    -7.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
]

invalid_k30 = data.filter(
    pl.col("k30").is_not_null()
    & ~pl.col("k30").is_in(allowed_k30_values)
)

if invalid_k30.height > 0:
    raise ValueError(
        f"Unexpected k30 values detected: {invalid_k30.height}"
    )

print()
print("K30 VALUE VALIDATION")

print(
    data.group_by("k30")
    .agg(pl.len().alias("count"))
    .sort("k30")
)

data = data.with_columns(
    pl.when(
        pl.col("k30").is_between(0, 4, closed="both")
    )
    .then(
        (pl.col("k30") >= 3).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("major_obstacle_rebuilt"),

    pl.when(
        pl.col("k30").is_between(0, 4, closed="both")
    )
    .then(
        (pl.col("k30") >= 1).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("any_obstacle"),

    pl.when(
        pl.col("k30").is_between(0, 4, closed="both")
    )
    .then(
        (pl.col("k30") == 4).cast(pl.Int8)
    )
    .otherwise(None)
    .alias("severe_obstacle"),
)

data = data.with_columns(
    pl.col("a3a")
    .replace_strict(region_labels, return_dtype=pl.String)
    .alias("region")
)

print()
print("DATA VALIDATION")
print("Rows:", data.height)
print("Unique establishments:", data.select(pl.col("idstd").n_unique()).item())
print("Valid k30 observations:", data.filter(pl.col("k30").is_between(0, 4, closed="both")).height)

region_distribution = (
    data.group_by(["a3a", "region"])
    .agg(
        pl.len().alias("sample_n")
    )
    .sort("a3a")
)

official_distribution = pl.DataFrame(
    {
        "a3a": list(official_counts.keys()),
        "official_n": list(official_counts.values()),
    }
)

region_distribution = (
    pl.DataFrame(
        {
            "a3a": list(region_labels.keys()),
            "region": list(region_labels.values()),
        }
    )
    .join(region_distribution, on=["a3a", "region"], how="left")
    .join(official_distribution, on="a3a", how="left")
    .with_columns(
        pl.col("sample_n").fill_null(0),
        (
            pl.col("sample_n").fill_null(0)
            - pl.col("official_n")
        ).alias("count_difference"),
    )
    .with_columns(
        (pl.col("count_difference") == 0)
        .alias("matches_official_count")
    )
    .sort("a3a")
)

region_distribution.write_csv(
    output_dir / "24c19_region_distribution.csv"
)

print()
print("REGION DISTRIBUTION VALIDATION")
print(
    region_distribution.select(
        pl.col("sample_n").sum().alias("total_sample"),
        pl.col("official_n").sum().alias("official_total"),
        pl.col("matches_official_count").sum().alias("matching_regions"),
    )
)

outcome_variables = [
    "major_obstacle_rebuilt",
    "any_obstacle",
    "severe_obstacle",
]

weight_variables = [
    "wweak",
    "wmedian",
]

prevalence_rows = []

for outcome in outcome_variables:
    for weight in weight_variables:
        valid = data.filter(
            pl.col(outcome).is_not_null()
            & pl.col(weight).is_not_null()
            & (pl.col(weight) > 0)
        )

        result = (
            valid.group_by(["a3a", "region"])
            .agg(
                pl.len().alias("valid_n"),
                pl.col(outcome).sum().alias("event_n"),
                pl.col(weight).sum().alias("weight_total"),
                (
                    pl.col(weight) * pl.col(outcome)
                ).sum().alias("weighted_events"),
                (
                    pl.col(weight) * pl.col(weight)
                ).sum().alias("weight_squared_total"),
            )
            .with_columns(
                (
                    pl.col("weighted_events")
                    / pl.col("weight_total")
                    * 100
                ).alias("weighted_prevalence_pct"),

                (
                    pl.col("weight_total")
                    * pl.col("weight_total")
                    / pl.col("weight_squared_total")
                ).alias("kish_effective_n"),
            )
            .with_columns(
                pl.lit(outcome).alias("outcome"),
                pl.lit(weight).alias("weight"),
                (
                    (pl.col("event_n") < 10)
                    |
                    ((pl.col("valid_n") - pl.col("event_n")) < 10)
                ).alias("sparse_cell_flag"),
            )
        )

        prevalence_rows.append(result)

weighted_prevalence = pl.concat(
    prevalence_rows,
    how="diagonal_relaxed",
).sort(
    ["outcome", "weight", "a3a"]
)

weighted_prevalence.write_csv(
    output_dir / "24c19_weighted_geographic_prevalence.csv"
)

print()
print("WEIGHTED PREVALENCE AUDIT")
print(weighted_prevalence.head(10))

if coefficient_path.exists():
    coefficients = pl.read_csv(coefficient_path)

    required_coefficient_columns = [
        "term",
        "model",
        "estimate",
        "p_value",
    ]

    missing_coefficient_columns = [
        column
        for column in required_coefficient_columns
        if column not in coefficients.columns
    ]

    if missing_coefficient_columns:
        raise ValueError(
            "Missing coefficient columns: "
            f"{missing_coefficient_columns}"
        )

    geographic_coefficients = (
        coefficients
        .with_columns(
            pl.col("term")
            .str.extract(
                r"^a3a_([0-9]+)(?:\.0)?$",
                1,
            )
            .cast(pl.Int64, strict=False)
            .alias("a3a")
        )
        .filter(pl.col("a3a").is_not_null())
        .with_columns(
            pl.col("a3a")
            .replace_strict(
                region_labels,
                return_dtype=pl.String,
            )
            .alias("region"),

            pl.col("estimate").exp().alias("odds_ratio"),
        )
    )

    if "std_error" in geographic_coefficients.columns:
        geographic_coefficients = geographic_coefficients.with_columns(
            (
                pl.col("estimate")
                - 1.96 * pl.col("std_error")
            ).exp().alias("or_ci_lower"),

            (
                pl.col("estimate")
                + 1.96 * pl.col("std_error")
            ).exp().alias("or_ci_upper"),
        )

    geographic_coefficients.write_csv(
        output_dir / "24c19_geographic_odds_ratios.csv"
    )

    print()
    print("GEOGRAPHIC COEFFICIENT AUDIT")
    print("Coefficient rows:", geographic_coefficients.height)

else:
    print()
    print("Coefficient file not found. Coefficient audit skipped.")

print()
print("24C-19 GEOGRAPHIC AUDIT COMPLETED")
print("Output directory:", output_dir)