from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import inspect
import traceback

import polars as pl
import svy


project_dir = Path(__file__).resolve().parent.parent

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
    / "script58_survey_design_verification"
)

output_dir.mkdir(parents=True, exist_ok=True)

required_columns = [
    "idstd",
    "strata",
    "stratificationpanelcode",
    "a4a",
    "a2",
    "a6a",
    "wmedian",
    "wweak",
]

if not data_path.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {data_path}"
    )

df = pl.read_parquet(data_path)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required survey design variables are missing: {missing_columns}"
    )

print("=" * 75)
print("SURVEY DESIGN VERIFICATION")
print("=" * 75)

print(f"Input dataset: {data_path}")
print(f"Observations: {df.height:,}")
print(f"Variables: {df.width:,}")

report_lines = []

def add_report(text):
    report_lines.append(str(text))


add_report("SURVEY DESIGN VERIFICATION")
add_report("=" * 75)
add_report(f"Dataset: {data_path}")
add_report(f"Observations: {df.height:,}")
add_report(f"Variables: {df.width:,}")


print()
print("=" * 75)
print("1. IDENTIFIER AND STRATA VALIDATION")
print("=" * 75)

identifier_summary = df.select(
    [
        pl.len().alias("observations"),
        pl.col("idstd").n_unique().alias("unique_psus"),
        pl.col("idstd").null_count().alias("missing_psu"),
        pl.col("strata").n_unique().alias("unique_strata"),
        pl.col("strata").null_count().alias("missing_strata"),
    ]
)

print(identifier_summary)

identifier_summary.write_csv(
    output_dir / "identifier_summary.csv"
)

duplicate_psus = (
    df.group_by("idstd")
    .agg(pl.len().alias("observations"))
    .filter(pl.col("observations") > 1)
)

duplicate_psus.write_csv(
    output_dir / "duplicate_psu_audit.csv"
)

print(
    f"Duplicate PSU identifiers: {duplicate_psus.height:,}"
)

strata_summary = (
    df.group_by("strata")
    .agg(
        [
            pl.len().alias("observations"),
            pl.col("idstd").n_unique().alias("psu_count"),
            pl.col("a4a").n_unique().alias("industry_count"),
            pl.col("a2").n_unique().alias("region_count"),
            pl.col("a6a").n_unique().alias("size_category_count"),
            pl.col("stratificationpanelcode")
            .n_unique()
            .alias("panel_count"),
        ]
    )
    .sort("strata")
)

strata_summary.write_csv(
    output_dir / "strata_structure.csv"
)

singleton_summary = (
    strata_summary
    .group_by("psu_count")
    .agg(
        [
            pl.len().alias("number_of_strata"),
            pl.col("observations").sum().alias("number_of_observations"),
        ]
    )
    .sort("psu_count")
)

singleton_summary.write_csv(
    output_dir / "singleton_summary.csv"
)

singleton_strata = strata_summary.filter(
    pl.col("psu_count") == 1
)

print()
print(f"Number of strata: {strata_summary.height:,}")
print(f"Singleton strata: {singleton_strata.height:,}")
print(
    "Observations in singleton strata: "
    f"{singleton_strata.get_column('observations').sum():,}"
)

add_report("")
add_report("STRATA STRUCTURE")
add_report(f"Number of strata: {strata_summary.height:,}")
add_report(f"Singleton strata: {singleton_strata.height:,}")
add_report(
    "Observations in singleton strata: "
    f"{singleton_strata.get_column('observations').sum():,}"
)


print()
print("=" * 75)
print("2. STRATIFICATION VARIABLE CONSISTENCY")
print("=" * 75)

stratification_check = (
    df.group_by(
        [
            "strata",
            "a4a",
            "a2",
            "a6a",
            "stratificationpanelcode",
        ]
    )
    .agg(pl.len().alias("observations"))
    .group_by("strata")
    .agg(
        [
            pl.len().alias("combinations_per_stratum"),
            pl.col("observations").sum().alias("observations"),
        ]
    )
    .sort("strata")
)

stratification_check.write_csv(
    output_dir / "stratification_consistency.csv"
)

inconsistent_strata = stratification_check.filter(
    pl.col("combinations_per_stratum") > 1
)

print(
    "Strata with multiple stratification combinations: "
    f"{inconsistent_strata.height:,}"
)

panel_summary = (
    df.group_by("stratificationpanelcode")
    .agg(
        [
            pl.len().alias("observations"),
            pl.col("strata").n_unique().alias("strata_count"),
            pl.col("idstd").n_unique().alias("psu_count"),
        ]
    )
    .sort("stratificationpanelcode")
)

panel_summary.write_csv(
    output_dir / "panel_summary.csv"
)

print(panel_summary)


print()
print("=" * 75)
print("3. SURVEY WEIGHT VALIDATION")
print("=" * 75)

weight_summary_rows = []

for weight_column in ["wmedian", "wweak"]:
    weight_data = (
        df.get_column(weight_column)
        .cast(pl.Float64, strict=False)
    )

    summary = df.select(
        [
            pl.len().alias("observations"),
            pl.col(weight_column)
            .null_count()
            .alias("missing_values"),
            (
                pl.col(weight_column)
                .cast(pl.Float64, strict=False)
                .is_finite()
                .sum()
            ).alias("finite_values"),
            (
                pl.col(weight_column)
                .cast(pl.Float64, strict=False)
                .filter(
                    pl.col(weight_column)
                    .cast(pl.Float64, strict=False)
                    <= 0
                )
                .len()
            ).alias("nonpositive_values"),
            pl.col(weight_column)
            .cast(pl.Float64, strict=False)
            .min()
            .alias("minimum"),
            pl.col(weight_column)
            .cast(pl.Float64, strict=False)
            .median()
            .alias("median"),
            pl.col(weight_column)
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("mean"),
            pl.col(weight_column)
            .cast(pl.Float64, strict=False)
            .max()
            .alias("maximum"),
            pl.col(weight_column)
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("sum"),
        ]
    )

    weight_summary_rows.append(
        summary.with_columns(
            pl.lit(weight_column).alias("weight")
        )
    )

    print(f"\n{weight_column}")
    print(summary)

weight_summary = pl.concat(
    weight_summary_rows,
    how="vertical",
).select(
    [
        "weight",
        "observations",
        "missing_values",
        "finite_values",
        "nonpositive_values",
        "minimum",
        "median",
        "mean",
        "maximum",
        "sum",
    ]
)

weight_summary.write_csv(
    output_dir / "weight_summary.csv"
)

weight_by_panel = (
    df.group_by("stratificationpanelcode")
    .agg(
        [
            pl.col("wmedian")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("wmedian_mean"),
            pl.col("wweak")
            .cast(pl.Float64, strict=False)
            .mean()
            .alias("wweak_mean"),
            pl.col("wmedian")
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("wmedian_sum"),
            pl.col("wweak")
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("wweak_sum"),
        ]
    )
    .sort("stratificationpanelcode")
)

weight_by_panel.write_csv(
    output_dir / "weight_by_panel.csv"
)

print()
print("Weight summary by panel:")
print(weight_by_panel)


print()
print("=" * 75)
print("4. SVY PACKAGE VERSION AND API INSPECTION")
print("=" * 75)

try:
    svy_version = version("svy")
except PackageNotFoundError:
    svy_version = "Version metadata unavailable"

print(f"svy package version: {svy_version}")

add_report("")
add_report("SVY PACKAGE")
add_report(f"Version: {svy_version}")

design_constructor = getattr(svy, "Design", None)

if design_constructor is None:
    raise AttributeError(
        "svy.Design was not found in the installed package."
    )

try:
    constructor_signature = inspect.signature(
        design_constructor
    )
except Exception as error:
    constructor_signature = (
        f"Signature unavailable: {error}"
    )

print(
    f"Design constructor signature: {constructor_signature}"
)

add_report(
    f"Design constructor signature: {constructor_signature}"
)

try:
    package_file = inspect.getfile(svy)
except Exception as error:
    package_file = f"Unavailable: {error}"

print(f"Package file: {package_file}")
add_report(f"Package file: {package_file}")

try:
    constructor_source = inspect.getsource(
        design_constructor
    )
except Exception as error:
    constructor_source = (
        f"Source unavailable: {error}"
    )

add_report("")
add_report("DESIGN CONSTRUCTOR SOURCE")
add_report(constructor_source)

print()
print("Design constructor source:")
print(constructor_source[:5000])

print()
print("=" * 75)
print("5. DESIGN OBJECT INSPECTION")
print("=" * 75)

design_objects = {}

for singleton_method in ["scale", "center"]:
    print(f"\nCreating design with singleton='{singleton_method}'")

    try:
        design = svy.Design(
            stratum="strata",
            psu="idstd",
            wgt="wmedian",
            singleton=singleton_method,
        )

        design_objects[singleton_method] = design

        design_type = type(design)

        try:
            design_signature = inspect.signature(
                design_type
            )
        except Exception:
            design_signature = "Unavailable"

        print(f"Object type: {design_type}")
        print(f"Object representation: {repr(design)}")

        add_report("")
        add_report(
            f"DESIGN OBJECT: singleton={singleton_method}"
        )
        add_report(f"Object type: {design_type}")
        add_report(
            f"Object signature: {design_signature}"
        )
        add_report(
            f"Object representation: {repr(design)}"
        )

        try:
            object_attributes = vars(design)
        except Exception:
            object_attributes = {}

        print("Object attributes:")

        for attribute, value in object_attributes.items():
            line = f"{attribute}: {repr(value)}"
            print(line)
            add_report(line)

        print("Available callable methods:")

        for method_name, method in inspect.getmembers(
            design_type,
            predicate=callable,
        ):
            try:
                method_signature = inspect.signature(method)
            except Exception:
                method_signature = "Unavailable"

            line = (
                f"{method_name}{method_signature}"
            )

            print(line)
            add_report(line)

    except Exception as error:
        message = (
            f"Design construction failed for "
            f"singleton={singleton_method}: "
            f"{type(error).__name__}: {error}"
        )

        print(message)
        add_report(message)
        add_report(traceback.format_exc())


print()
print("=" * 75)
print("6. PACKAGE MODULE INSPECTION")
print("=" * 75)

module_members = []

for name, member in inspect.getmembers(svy):
    if name.startswith("_"):
        continue

    if inspect.isclass(member) or inspect.isfunction(member):
        try:
            signature = inspect.signature(member)
        except Exception:
            signature = "Unavailable"

        module_members.append(
            f"{name}{signature}"
        )

for item in module_members:
    print(item)
    add_report(item)


print()
print("=" * 75)
print("7. DESIGN INFORMATION LIMITATIONS")
print("=" * 75)

limitations = [
    "The current dataset audit does not establish whether the sample design is with replacement.",
    "The current dataset audit does not establish first-stage inclusion probabilities.",
    "The current dataset audit does not establish finite population corrections.",
    "The existence of singleton strata does not justify automatically dropping observations.",
    "Weight variables alone do not establish the complete variance estimation method.",
    "The svy package implementation must be checked before final inference.",
]

for item in limitations:
    print(f"- {item}")
    add_report(f"- {item}")


report_path = output_dir / "svy_package_and_design_report.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print()
print("=" * 75)
print("SCRIPT58 COMPLETED")
print("=" * 75)

print(f"Output directory: {output_dir}")
print(f"Technical report: {report_path}")

print()
print("No final survey standard errors or p-values were calculated.")
print(
    "Review the package report and survey design documentation "
    "before finalizing variance estimation."
)