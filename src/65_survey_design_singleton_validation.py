import polars as pl
import svy

from pathlib import Path


project_dir = Path(__file__).resolve().parents[1]

data_path = (
    project_dir
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "survey_design_validation"
)

output_dir.mkdir(parents=True, exist_ok=True)


def write_csv_safe(dataframe, output_path):
    expressions = []

    for column, dtype in dataframe.schema.items():

        if dtype.base_type() == pl.List:

            expressions.append(
                pl.col(column)
                .list.eval(
                    pl.element().cast(pl.String)
                )
                .list.join(";")
                .alias(column)
            )

        elif dtype.base_type() == pl.Struct:

            raise TypeError(
                f"Unsupported Struct column '{column}' "
                f"in {output_path.name}. "
                "Explicit serialization is required."
            )

        else:

            expressions.append(
                pl.col(column)
            )

    export_dataframe = dataframe.select(expressions)

    export_dataframe.write_csv(output_path)


print("=" * 78)
print("24C-28 SURVEY DESIGN AND SINGLETON VALIDATION")
print("=" * 78)

if not data_path.exists():

    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

df = pl.read_parquet(data_path)

required_columns = [
    "idstd",
    "strata",
    "a4a",
    "a2",
    "a6a",
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
        f"Missing required columns: {missing_columns}"
    )


print()
print("1. INITIAL DATA VALIDATION")

print(f"Observations: {df.height:,}")
print(f"Columns: {df.width:,}")

design_columns = [
    "idstd",
    "strata",
    "a4a",
    "a2",
    "a6a",
]

missing_report = df.select(
    [
        pl.col(column)
        .null_count()
        .alias(column)
        for column in design_columns + ["wmedian", "wweak"]
    ]
)

print(missing_report)

missing_design_rows = df.filter(
    pl.any_horizontal(
        [
            pl.col(column).is_null()
            for column in design_columns
        ]
    )
).height

if missing_design_rows > 0:

    raise ValueError(
        f"Missing sampling design identifiers in "
        f"{missing_design_rows} observations."
    )


print()
print("2. ESTABLISHMENT AND PSU IDENTIFIER VALIDATION")

id_audit = df.select(
    pl.len().alias("observations"),
    pl.col("idstd").n_unique().alias("unique_idstd"),
    pl.col("idstd").is_duplicated().sum().alias("duplicate_idstd"),
)

print(id_audit)

if (
    id_audit["observations"][0]
    != id_audit["unique_idstd"][0]
):

    raise ValueError(
        "idstd is not unique in the establishment dataset."
    )

print("Establishment identifier uniqueness: PASSED")


print()
print("3. STRATUM INTERNAL CONSISTENCY")

stratum_mapping = (
    df.group_by("strata")
    .agg(
        pl.col("a4a").n_unique().alias("n_a4a"),
        pl.col("a2").n_unique().alias("n_a2"),
        pl.col("a6a").n_unique().alias("n_a6a"),
        pl.col("idstd").n_unique().alias("n_psus"),
        pl.len().alias("n_establishments"),
    )
    .sort("strata")
)

invalid_strata = stratum_mapping.filter(
    (pl.col("n_a4a") != 1)
    | (pl.col("n_a2") != 1)
    | (pl.col("n_a6a") != 1)
)

print(f"Number of strata: {stratum_mapping.height:,}")
print(f"Invalid stratum definitions: {invalid_strata.height:,}")

if invalid_strata.height > 0:

    print(invalid_strata)

    raise ValueError(
        "At least one stratum contains multiple observed "
        "sampling-variable combinations."
    )

print("Stratum internal consistency: PASSED")


print()
print("4. REVERSE STRATUM MAPPING DIAGNOSTICS")

reverse_mapping = (
    df.group_by(["a4a", "a2", "a6a"])
    .agg(
        pl.col("strata").n_unique().alias("n_strata"),
        pl.col("strata").unique().sort().alias("strata_codes"),
        pl.col("idstd").n_unique().alias("n_psus"),
        pl.len().alias("n_establishments"),
    )
    .sort(
        ["n_strata", "n_establishments"],
        descending=True,
    )
)

multiple_mapping = reverse_mapping.filter(
    pl.col("n_strata") > 1
)

print(
    "Combinations mapping to multiple strata:",
    multiple_mapping.height,
)

if multiple_mapping.height > 0:

    print(multiple_mapping)

else:

    print("No multiple stratum mappings detected.")


print()
print("5. PANEL VARIABLE CHECK")

panel_column = "stratificationpanelcode"

if panel_column in df.columns:

    print(f"Panel variable detected: {panel_column}")

    print("Panel code distribution:")

    print(
        df.group_by(panel_column)
        .agg(
            pl.len().alias("observations"),
            pl.col("idstd").n_unique().alias("unique_psus"),
        )
        .sort(panel_column)
    )

    panel_mapping = (
        df.group_by(
            ["a4a", "a2", "a6a", panel_column]
        )
        .agg(
            pl.col("strata").n_unique().alias("n_strata"),
            pl.col("strata").unique().sort().alias("strata_codes"),
            pl.col("idstd").n_unique().alias("n_psus"),
            pl.len().alias("n_establishments"),
        )
        .sort(
            ["n_strata", "n_establishments"],
            descending=True,
        )
    )

    panel_multiple_mapping = panel_mapping.filter(
        pl.col("n_strata") > 1
    )

    print(
        "Four-variable combinations mapping to multiple strata:",
        panel_multiple_mapping.height,
    )

    if panel_multiple_mapping.height > 0:
        print(panel_multiple_mapping)

else:

    print(
        f"{panel_column} is not present in the loaded dataset."
    )

    panel_mapping = pl.DataFrame()
    panel_multiple_mapping = pl.DataFrame()


print()
print("6. SINGLETON STRATUM AUDIT")

stratum_audit = (
    df.group_by("strata")
    .agg(
        pl.col("idstd").n_unique().alias("n_psus"),
        pl.len().alias("n_establishments"),
    )
    .with_columns(
        (pl.col("n_psus") == 1).alias("is_singleton")
    )
    .sort("strata")
)

n_strata = stratum_audit.height

n_singletons = (
    stratum_audit
    .filter(pl.col("is_singleton"))
    .height
)

n_non_singletons = n_strata - n_singletons

singleton_fraction = (
    n_singletons / n_strata
    if n_strata > 0
    else None
)

print(f"Total strata: {n_strata:,}")
print(f"Singleton strata: {n_singletons:,}")
print(f"Non-singleton strata: {n_non_singletons:,}")
print(f"Singleton fraction: {singleton_fraction:.4%}")

if singleton_fraction < 1:

    scale_variance_multiplier = (
        1 / (1 - singleton_fraction)
    )

    scale_se_multiplier = (
        scale_variance_multiplier ** 0.5
    )

else:

    scale_variance_multiplier = None
    scale_se_multiplier = None

print(
    "Illustrative scale variance multiplier:",
    scale_variance_multiplier,
)

print(
    "Illustrative scale standard-error multiplier:",
    scale_se_multiplier,
)


print()
print("7. SURVEY WEIGHT VALIDATION")

weight_results = []

for weight in ["wmedian", "wweak"]:

    weight_audit = df.select(
        pl.col(weight).null_count().alias("missing"),

        (
            (pl.col(weight) <= 0)
            .fill_null(False)
            .sum()
        ).alias("nonpositive"),

        (
            (~pl.col(weight).is_finite())
            .fill_null(True)
            .sum()
        ).alias("nonfinite"),

        pl.col(weight).min().alias("minimum"),
        pl.col(weight).max().alias("maximum"),
        pl.col(weight).mean().alias("mean"),
        pl.col(weight).sum().alias("weight_total"),
    )

    print()
    print(f"Weight: {weight}")
    print(weight_audit)

    weight_results.append(
        weight_audit.with_columns(
            pl.lit(weight).alias("weight")
        )
    )

    if (
        weight_audit["missing"][0] > 0
        or weight_audit["nonpositive"][0] > 0
        or weight_audit["nonfinite"][0] > 0
    ):

        raise ValueError(
            f"Invalid weight values detected in {weight}."
        )


print()
print("8. SVY DESIGN AND SINGLETON HANDLING")

singleton_results = []
singleton_handling_results = []
design_results = []

for weight in ["wmedian", "wweak"]:

    print()
    print(f"Testing survey design with {weight}")

    design = svy.Design(
        stratum="strata",
        psu="idstd",
        wgt=weight,
        singleton="scale",
    )

    sample = svy.Sample(
        data=df,
        design=design,
    )

    print(f"Number of PSUs: {sample.n_psus:,}")
    print(f"Number of strata: {sample.n_strata:,}")
    print(f"Number of singletons: {sample.n_singletons:,}")

    singleton_method = getattr(
        sample.design.singleton,
        "method",
        None,
    )

    print(
        "Singleton rule:",
        sample.design.singleton,
    )

    print(
        "Detected singleton method:",
        singleton_method,
    )

    if sample.n_strata != n_strata:

        raise ValueError(
            f"Stratum count mismatch for {weight}."
        )

    if sample.n_singletons != n_singletons:

        raise ValueError(
            f"Singleton count mismatch for {weight}."
        )

    if singleton_method != "scale":

        raise ValueError(
            f"Singleton scale rule not retained for {weight}. "
            f"Detected method: {singleton_method}"
        )

    print("Singleton method validation: PASSED")

    if "handled" not in sample.singletons.columns:

        raise ValueError(
            "The singleton report does not contain "
            "the expected 'handled' column."
        )

    if sample.singletons["handled"].null_count() > 0:

        raise ValueError(
            f"Some singleton strata have missing handling "
            f"information for {weight}."
        )

    print()
    print("Singleton handling details:")

    print(sample.singletons)

    handling_summary = (
        sample.singletons
        .group_by("handled")
        .agg(
            pl.len().alias("number_of_strata")
        )
        .with_columns(
            pl.lit(weight).alias("weight")
        )
    )

    print()
    print("Singleton handling summary:")
    print(handling_summary)

    singleton_results.append(
        sample.singletons.with_columns(
            pl.lit(weight).alias("weight")
        )
    )

    singleton_handling_results.append(
        handling_summary
    )

    design_results.append(
        pl.DataFrame(
            {
                "weight": [weight],
                "n_psus": [sample.n_psus],
                "n_strata": [sample.n_strata],
                "n_singletons": [sample.n_singletons],
                "singleton_rule": [str(singleton_method)],
            }
        )
    )


print()
print("9. MODEL-SAMPLE SINGLETON REFERENCE")

print(
    "Previously reported model sample singleton counts:"
)

model_singleton_reference = pl.DataFrame(
    {
        "model": [
            "model1_full",
            "model1_common",
            "model2_common",
        ],
        "n_observations": [
            2632,
            2528,
            2528,
        ],
        "n_strata_singletons": [
            578,
            560,
            560,
        ],
        "total_strata": [
            1047,
            1016,
            1016,
        ],
    }
).with_columns(
    (
        pl.col("n_strata_singletons")
        / pl.col("total_strata")
        * 100
    ).round(2).alias("singleton_pct")
)

print(model_singleton_reference)


print()
print("10. SAVE ALL AUDIT OUTPUTS")

write_csv_safe(
    stratum_audit,
    output_dir / "24c28_stratum_audit.csv",
)

write_csv_safe(
    reverse_mapping,
    output_dir / "24c28_reverse_stratum_mapping.csv",
)

write_csv_safe(
    multiple_mapping,
    output_dir / "24c28_multiple_stratum_mapping.csv",
)

write_csv_safe(
    pl.concat(
        weight_results,
        how="diagonal_relaxed",
    ),
    output_dir / "24c28_weight_audit.csv",
)

write_csv_safe(
    pl.concat(
        singleton_results,
        how="diagonal_relaxed",
    ),
    output_dir / "24c28_singleton_report.csv",
)

write_csv_safe(
    pl.concat(
        singleton_handling_results,
        how="diagonal_relaxed",
    ),
    output_dir / "24c28_singleton_handling_summary.csv",
)

write_csv_safe(
    pl.concat(
        design_results,
        how="diagonal_relaxed",
    ),
    output_dir / "24c28_design_report.csv",
)

write_csv_safe(
    model_singleton_reference,
    output_dir / "24c28_model_singleton_reference.csv",
)

if panel_mapping.height > 0:

    write_csv_safe(
        panel_mapping,
        output_dir / "24c28_panel_mapping.csv",
    )

    write_csv_safe(
        panel_multiple_mapping,
        output_dir / "24c28_panel_multiple_mapping.csv",
    )


print()
print("=" * 78)
print("24C-28 AUDIT COMPLETED")
print("=" * 78)

print("Establishment identifier uniqueness: PASSED")
print("Stratum internal consistency: PASSED")
print(
    "Reverse mapping diagnostics: COMPLETED "
    "(not automatically treated as an error)"
)
print("Singleton detection: COMPLETED")
print("Weight validation: PASSED")
print("SVY singleton scale declaration: PASSED")
print("Singleton handling report: SAVED")

print()
print("OUTPUT DIRECTORY:")
print(output_dir)

print()
print("IMPORTANT LIMITATIONS")

print(
    "This audit does not independently establish the official "
    "sampling frame or the complete original stratum construction."
)

print(
    "Full-sample singleton counts do not replace model-specific "
    "complete-case singleton counts."
)

print(
    "The scale multiplier above is illustrative and must not be "
    "interpreted as a direct estimate of the actual variance "
    "adjustment applied by the package."
)

print(
    "The singleton handling summary reports the package's "
    "recorded handling labels. It does not independently prove "
    "the numerical variance contribution of every singleton stratum."
)

print(
    "The model-sample singleton counts in section 9 are reference "
    "values from previous analyses and are not recalculated here."
)

print(
    "No regression models were re-estimated in this audit."
)

print()
print("Audit script finished.")