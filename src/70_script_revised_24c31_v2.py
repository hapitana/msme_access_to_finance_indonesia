from pathlib import Path
import os
import re
import runpy
import shutil
import sys
import tempfile

import svy
import svy_rs

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import t as t_distribution
from scipy.stats import norm
from scipy.optimize import minimize


project_dir = Path(__file__).resolve().parents[1]

script55_path = (
    project_dir
    / "src"
    / "55_survey_weighted_logistic_regression.py"
)

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "outcome_robustness"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

if not script55_path.exists():
    raise FileNotFoundError(
        f"Script 55 not found: {script55_path}"
    )


print("=" * 78)
print("24C-31. ACTUAL-INDUSTRY REGIONAL PAIRWISE CONTRAST ANALYSIS")
print("=" * 78)


print()
print("STEP 1. VERIFY ENVIRONMENT AND REBUILD THE ORIGINAL GEOGRAPHIC MODELS")

expected_svy_version = "0.35.0"
expected_svy_rs_version = "0.19.1"

if getattr(svy, "__version__", None) != expected_svy_version:
    raise RuntimeError(
        "Unexpected svy version. "
        f"Expected {expected_svy_version}; found {getattr(svy, '__version__', 'unknown')}."
    )

if getattr(svy_rs, "__version__", None) != expected_svy_rs_version:
    raise RuntimeError(
        "Unexpected svy_rs version. "
        f"Expected {expected_svy_rs_version}; found {getattr(svy_rs, '__version__', 'unknown')}."
    )

print(f"Python version: {sys.version.split()[0]}")
print(f"svy version: {svy.__version__}")
print(f"svy_rs version: {svy_rs.__version__}")
print(f"Polars version: {pl.__version__}")
print(f"NumPy version: {np.__version__}")

# Script 55 writes many audited outputs while rebuilding its in-memory
# survey objects. Preserve the existing project outputs and restore them
# after Script 55 finishes, before writing the new 24C-31 results.
processed_dir = project_dir / "data" / "processed"
output_folders = [
    processed_dir / "script55_survey_weighted_logistic_regression",
    processed_dir / "outcome_robustness",
]

original_cwd = Path.cwd()
with tempfile.TemporaryDirectory(prefix="24c31_script55_backup_") as backup_name:
    backup_root = Path(backup_name)
    existing_folders = []

    for folder in output_folders:
        if folder.exists():
            backup_folder = backup_root / folder.name
            shutil.copytree(folder, backup_folder)
            existing_folders.append((folder, backup_folder))

    try:
        os.chdir(project_dir)
        namespace = runpy.run_path(str(script55_path))
    finally:
        os.chdir(original_cwd)
        for folder in output_folders:
            if folder.exists():
                shutil.rmtree(folder)
        for folder, backup_folder in existing_folders:
            shutil.copytree(backup_folder, folder)

required_objects = [
    "data_geo",
    "svy",
    "geo_designs",
    "geo_model1_predictors",
    "geo_model2_predictors",
]

missing_objects = [
    object_name
    for object_name in required_objects
    if object_name not in namespace
]

if missing_objects:
    raise KeyError(
        f"Required objects not found in Script 55 namespace: {missing_objects}"
    )

data_geo = namespace["data_geo"]
svy = namespace["svy"]
geo_designs = namespace["geo_designs"]

required_geo_columns = [
    "idstd", "strata", "a3a", "d1a2_v4",
    "major_obstacle", "any_obstacle",
    "model1_complete", "model2_complete", "wmedian", "wweak",
    "firm_age", "log_employees", "female_owner", "bank_account",
]
missing_geo_columns = [
    column for column in required_geo_columns
    if column not in data_geo.columns
]
if missing_geo_columns:
    raise KeyError(f"Required columns missing from Script 55 data_geo: {missing_geo_columns}")

print()
print("STEP 1A. MEASUREMENT AND SURVEY-DESIGN AUDITS")

# Audit the identifiers, official strata, and weights before sparse-region exclusion.
strata_audit = (
    data_geo.group_by("strata")
    .agg(
        pl.len().alias("establishment_n"),
        pl.col("idstd").n_unique().alias("psu_n"),
    )
    .sort("strata")
    .with_columns((pl.col("psu_n") == 1).alias("singleton_stratum"))
)
strata_audit.write_csv(output_dir / "24c31_survey_strata_psu_audit.csv")

audit_rows = []
for field_name in ["idstd", "strata", "a3a", "wmedian", "wweak"]:
    values = data_geo.get_column(field_name)
    audit_rows.append({
        "audit": "field_completeness",
        "field": field_name,
        "n_total": data_geo.height,
        "n_nonmissing": values.drop_nulls().len(),
        "n_missing": values.null_count(),
        "detail": "Pre-exclusion analytic frame",
    })
for weight_name in ["wmedian", "wweak"]:
    values = data_geo.get_column(weight_name).cast(pl.Float64, strict=False)
    valid = values.filter(values.is_finite() & (values > 0))
    audit_rows.append({
        "audit": "weight_distribution",
        "field": weight_name,
        "n_total": data_geo.height,
        "n_nonmissing": valid.len(),
        "n_missing": data_geo.height - valid.len(),
        "detail": f"min={valid.min()}; median={valid.median()}; max={valid.max()}; sum={valid.sum()}",
    })
for field_name in ["q_version", "stratificationpanelcode"]:
    if field_name in data_geo.columns:
        (data_geo.group_by(field_name).agg(pl.len().alias("n")).sort(field_name)
         .write_csv(output_dir / f"24c31_{field_name}_distribution.csv"))

audit_rows.extend([
    {
        "audit": "survey_design",
        "field": "idstd",
        "n_total": data_geo.height,
        "n_nonmissing": data_geo.get_column("idstd").drop_nulls().len(),
        "n_missing": data_geo.get_column("idstd").null_count(),
        "detail": f"unique PSU identifiers={data_geo.get_column('idstd').n_unique()}",
    },
    {
        "audit": "survey_design",
        "field": "strata",
        "n_total": data_geo.height,
        "n_nonmissing": data_geo.get_column("strata").drop_nulls().len(),
        "n_missing": data_geo.get_column("strata").null_count(),
        "detail": f"observed strata={strata_audit.height}; singleton strata={strata_audit.get_column('singleton_stratum').sum()}",
    },
])

# Retrieve the original ordinal k30 item. Never reconstruct its distribution
# from the two binary outcomes.
measurement_frame = data_geo if "k30" in data_geo.columns else None
if measurement_frame is None:
    for candidate_path in [
        project_dir / "data" / "processed" / "indonesia_2023_candidate_analytic.parquet",
        project_dir / "data" / "processed" / "indonesia_2023.parquet",
    ]:
        if candidate_path.exists():
            candidate_frame = pl.read_parquet(candidate_path)
            if "k30" in candidate_frame.columns:
                measurement_frame = candidate_frame
                break

if measurement_frame is not None:
    selected = ["k30"] + [c for c in ["a3a", "wmedian", "wweak"] if c in measurement_frame.columns]
    k30_frame = measurement_frame.select(selected).with_columns(
        pl.col("k30").cast(pl.Float64, strict=False).alias("k30_numeric")
    )
    observed = k30_frame.filter(
        pl.col("k30_numeric").is_not_null() & pl.col("k30_numeric").is_finite()
    )
    allowed_codes = [-9.0, -7.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    invalid = observed.filter(~pl.col("k30_numeric").is_in(allowed_codes))
    if invalid.height:
        invalid_values = invalid.get_column("k30_numeric").unique().sort().to_list()
        raise ValueError(
            f"Unexpected k30 codes {invalid_values}. Expected -9 (don't know), "
            "-7 (does not apply), or substantive values 0-4. Check questionnaire coding."
        )

    # WBES codebook: -9 = Don't know (spontaneous); -7 = Does not apply.
    # These are non-substantive response codes, not ordinal scale points.
    k30_frame = k30_frame.with_columns(
        pl.when(pl.col("k30_numeric") == -9).then(pl.lit("-9_dont_know"))
        .when(pl.col("k30_numeric") == -7).then(pl.lit("-7_does_not_apply"))
        .when(pl.col("k30_numeric") == 0).then(pl.lit("0_no_obstacle"))
        .when(pl.col("k30_numeric") == 1).then(pl.lit("1_minor_obstacle"))
        .when(pl.col("k30_numeric") == 2).then(pl.lit("2_moderate_obstacle"))
        .when(pl.col("k30_numeric") == 3).then(pl.lit("3_major_obstacle"))
        .when(pl.col("k30_numeric") == 4).then(pl.lit("4_very_severe_obstacle"))
        .otherwise(pl.lit("missing_or_non_numeric"))
        .alias("k30_category")
    )
    distribution = (
        k30_frame.group_by("k30_category").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / measurement_frame.height * 100).alias("percent"))
        .sort("k30_category")
    )
    distribution.write_csv(output_dir / "24c31_k30_ordinal_distribution.csv")

    weighted_k30 = k30_frame
    for weight_name in ["wmedian", "wweak"]:
        if weight_name in weighted_k30.columns:
            valid_weighted = weighted_k30.filter(
                pl.col(weight_name).is_not_null()
                & pl.col(weight_name).cast(pl.Float64, strict=False).is_finite()
                & (pl.col(weight_name).cast(pl.Float64, strict=False) > 0)
            )
            weighted_distribution = (
                valid_weighted.group_by("k30_category")
                .agg(pl.col(weight_name).cast(pl.Float64).sum().alias("weighted_total"))
                .with_columns(
                    (pl.col("weighted_total") / pl.col("weighted_total").sum() * 100)
                    .alias("weighted_percent")
                )
                .with_columns(pl.lit(weight_name).alias("weight"))
                .sort("k30_category")
            )
            weighted_distribution.write_csv(
                output_dir / f"24c31_k30_weighted_distribution_{weight_name}.csv"
            )
            if "a3a" in valid_weighted.columns:
                regional_weighted = (
                    valid_weighted.group_by(["a3a", "k30_category"])
                    .agg(pl.col(weight_name).cast(pl.Float64).sum().alias("weighted_total"))
                    .with_columns(
                        (pl.col("weighted_total") / pl.col("weighted_total").sum().over("a3a") * 100)
                        .alias("regional_weighted_percent"),
                        pl.lit(weight_name).alias("weight"),
                    )
                    .sort(["a3a", "k30_category"])
                )
                regional_weighted.write_csv(
                    output_dir / f"24c31_k30_weighted_by_region_{weight_name}.csv"
                )

    # Binary thresholds are defined only for substantive ordinal responses 0-4.
    valid_k30 = k30_frame.filter(pl.col("k30_numeric").is_in([0.0, 1.0, 2.0, 3.0, 4.0]))
    threshold_rows = []
    for label, expr in [
        ("any_obstacle_gt_0", pl.col("k30_numeric") > 0),
        ("moderate_or_major_ge_2", pl.col("k30_numeric") >= 2),
        ("major_obstacle_ge_3", pl.col("k30_numeric") >= 3),
        ("highest_category_eq_4", pl.col("k30_numeric") == 4),
    ]:
        events = valid_k30.filter(expr).height
        threshold_rows.append({
            "threshold": label,
            "valid_substantive_n": valid_k30.height,
            "excluded_dont_know_n": k30_frame.filter(pl.col("k30_numeric") == -9).height,
            "excluded_not_applicable_n": k30_frame.filter(pl.col("k30_numeric") == -7).height,
            "excluded_missing_or_non_numeric_n": k30_frame.filter(
                pl.col("k30_numeric").is_null() | ~pl.col("k30_numeric").is_finite()
            ).height,
            "event_n": events,
            "event_percent_unweighted": events / valid_k30.height * 100 if valid_k30.height else None,
        })
    pl.DataFrame(threshold_rows).write_csv(output_dir / "24c31_k30_threshold_sensitivity.csv")

    if "a3a" in k30_frame.columns:
        by_region = (
            k30_frame.filter(pl.col("a3a").is_not_null()).group_by("a3a")
            .agg(
                pl.len().alias("n_total"),
                pl.col("k30_numeric").is_in([0.0, 1.0, 2.0, 3.0, 4.0]).sum().alias("k30_substantive_n"),
                (pl.col("k30_numeric") == -9).sum().alias("k30_dont_know_n"),
                (pl.col("k30_numeric") == -7).sum().alias("k30_not_applicable_n"),
                (pl.col("k30_numeric") > 0).sum().alias("any_obstacle_n"),
                (pl.col("k30_numeric") >= 3).sum().alias("major_obstacle_n"),
                (pl.col("k30_numeric").is_null() | ~pl.col("k30_numeric").is_finite()).sum().alias("k30_missing_or_non_numeric_n"),
            )
            .with_columns(
                (pl.col("any_obstacle_n") / pl.col("k30_substantive_n") * 100).alias("any_obstacle_percent_valid"),
                (pl.col("major_obstacle_n") / pl.col("k30_substantive_n") * 100).alias("major_obstacle_percent_valid"),
            )
            .sort("a3a")
        )
        by_region.write_csv(output_dir / "24c31_k30_measurement_by_region.csv")

else:
    (output_dir / "24c31_k30_measurement_audit_NOT_AVAILABLE.txt").write_text(
        "Original k30 item not found in data_geo or the expected processed parquet files. "
        "No ordinal distribution was inferred from binary outcomes. Restore the original "
        "analytic parquet containing k30 and rerun.\n", encoding="utf-8"
    )

# Record the verified external-data sources and aggregation rules. The script
# does not inject incomplete provincial aggregates into a 15-region model.
context_sources = pl.DataFrame([
    {"indicator": "GRDP per capita at current prices", "source": "BPS", "year": 2023,
     "aggregation": "sum provincial GRDP divided by sum provincial population",
     "status": "province values found; full population weighting must be verified for every WBES region",
     "url": "https://www.bps.go.id/id/statistics-table/3/YWtoQlRVZzNiMU5qU1VOSlRFeFZiRTR4VDJOTVVUMDkjMw%3D%3D/produk-domestik-regional-bruto-per-kapita-atas-dasar-harga-berlaku-menurut-provinsi--ribu-rupiah---2022.html?year=2023"},
    {"indicator": "Poverty headcount ratio", "source": "BPS", "year": 2023,
     "aggregation": "sum poor population divided by total population",
     "status": "March 2023 provincial series identified; verify new-Papua coverage and denominators",
     "url": "https://www.bps.go.id/id/statistics-table/3/UkVkWGJVZFNWakl6VWxKVFQwWjVWeTlSZDNabVFUMDkjMw%3D%3D/jumlah-dan-persentase-penduduk-miskin-menurut-provinsi--2024.html?year=2023"},
    {"indicator": "Internet access among population aged 5+", "source": "BPS Susenas", "year": 2023,
     "aggregation": "population-weighted prevalence only with comparable denominators",
     "status": "38-province table has unavailable values for several new Papua provinces; not complete for WBES region 21",
     "url": "https://www.bps.go.id/id/statistics-table/2/MTY5OSMy/persentase-penduduk-usia-5-tahun-ke-atas--yang-pernah-mengakses-internet-dalam-3-bulan-terakhir-menurut-provinsi-dan-kemampuan-membaca.html"},
    {"indicator": "SME credit by project location", "source": "OJK", "year": 2023,
     "aggregation": "sum provincial credit balances; do not impute unreported provinces",
     "status": "province table does not cover all provinces in WBES 15-region geography; descriptive-only where complete",
     "url": "https://ojk.go.id/id/kanal/perbankan/data-dan-statistik/statistik-perbankan-indonesia/Pages/Statistik-Perbankan-Indonesia---Desember-2023.aspx"},
])
context_sources.write_csv(output_dir / "24c31_regional_context_source_register.csv")
pl.DataFrame(audit_rows).write_csv(output_dir / "24c31_measurement_survey_design_audit.csv")
print(f"Observed strata: {strata_audit.height}")
print(f"Singleton strata: {strata_audit.get_column('singleton_stratum').sum()}")
print("Measurement and design audit outputs written.")

if set(geo_designs) != {"wmedian", "wweak"}:
    raise ValueError(f"Unexpected survey design keys: {sorted(geo_designs)}")
for weight_name, design_obj in geo_designs.items():
    if getattr(design_obj, "wgt", None) != weight_name:
        raise ValueError(f"Survey design weight mismatch for {weight_name}")

# Reproduce the audited actual-industry mapping from analysis 24C-28.
industry_text = (
    pl.col("d1a2_v4").cast(pl.String).str.strip_chars()
    .str.replace(r"\.0$", "")
)
industry_division = (
    industry_text.str.extract(r"^(\d{1,4})", 1).str.zfill(2)
    .str.slice(0, 2).cast(pl.Int64, strict=False)
)
section_expr = (
    pl.when(industry_division.is_between(1, 3)).then(pl.lit("A"))
    .when(industry_division.is_between(5, 9)).then(pl.lit("B"))
    .when(industry_division.is_between(10, 33)).then(pl.lit("C"))
    .when(industry_division == 35).then(pl.lit("D"))
    .when(industry_division.is_between(36, 39)).then(pl.lit("E"))
    .when(industry_division.is_between(41, 43)).then(pl.lit("F"))
    .when(industry_division.is_between(45, 47)).then(pl.lit("G"))
    .when(industry_division.is_between(49, 53)).then(pl.lit("H"))
    .when(industry_division.is_between(55, 56)).then(pl.lit("I"))
    .when(industry_division.is_between(58, 63)).then(pl.lit("J"))
    .when(industry_division.is_between(64, 66)).then(pl.lit("K"))
    .when(industry_division == 68).then(pl.lit("L"))
    .when(industry_division.is_between(69, 75)).then(pl.lit("M"))
    .when(industry_division.is_between(77, 82)).then(pl.lit("N"))
    .when(industry_division == 84).then(pl.lit("O"))
    .when(industry_division == 85).then(pl.lit("P"))
    .when(industry_division.is_between(86, 88)).then(pl.lit("Q"))
    .when(industry_division.is_between(90, 93)).then(pl.lit("R"))
    .when(industry_division.is_between(94, 96)).then(pl.lit("S"))
    .when(industry_division.is_between(97, 98)).then(pl.lit("T"))
    .when(industry_division == 99).then(pl.lit("U"))
    .otherwise(None).alias("industry_section")
)
data_geo = data_geo.with_columns([
    industry_division.alias("industry_division"), section_expr,
])
invalid_industry = data_geo.filter(
    pl.col("d1a2_v4").is_not_null() & pl.col("industry_section").is_null()
)
if invalid_industry.height:
    raise ValueError(f"Unmapped nonmissing actual-industry codes: {invalid_industry.select('d1a2_v4').unique().head(30)}")

# Restrict to the same 15-region sensitivity set as the earlier pairwise analysis.
sparse_region_codes = [2, 5, 10, 13, 16, 18, 19]
original_region_counts = (
    data_geo.group_by("a3a").agg(pl.len().alias("n")).sort("a3a")
)
data_geo = data_geo.filter(~pl.col("a3a").cast(pl.Int64).is_in(sparse_region_codes))
remaining_regions = (
    data_geo.get_column("a3a").cast(pl.Int64).unique().sort().to_list()
)
if len(remaining_regions) != 15 or 1 not in remaining_regions:
    raise ValueError(f"Expected 15 retained regions including Jakarta; found {remaining_regions}")

# Preserve the original complete-case domains while requiring a valid actual ISIC section.
for outcome_name in ["any_obstacle", "major_obstacle"]:
    data_geo = data_geo.with_columns([
        pl.all_horizontal([
            pl.col("model1_complete"), pl.col("industry_section").is_not_null(),
            pl.col("a3a").is_not_null(), pl.col(outcome_name).is_not_null(),
        ]).alias(f"actual_m1_full_{outcome_name}"),
        pl.all_horizontal([
            pl.col("model2_complete"), pl.col("industry_section").is_not_null(),
            pl.col("a3a").is_not_null(), pl.col(outcome_name).is_not_null(),
        ]).alias(f"actual_m1_common_{outcome_name}"),
    ])

geo_samples = {
    weight_name: svy.Sample(data_geo, design_obj)
    for weight_name, design_obj in geo_designs.items()
}
geo_results = {}
for outcome_name in ["major_obstacle", "any_obstacle"]:
    for weight_name, sample_obj in geo_samples.items():
        predictors_m1 = [
            "firm_age", "log_employees", "female_owner",
            svy.Cat("industry_section"), svy.Cat("a3a"),
        ]
        predictors_m2 = predictors_m1 + ["bank_account"]
        geo_results[f"{outcome_name}_{weight_name}"] = {
            f"model1_full_{weight_name}": sample_obj.glm.fit(
                y=outcome_name, x=predictors_m1, family="binomial", link="logit",
                where=(svy.col(f"actual_m1_full_{outcome_name}") == True),
            ),
            f"model1_common_{weight_name}": sample_obj.glm.fit(
                y=outcome_name, x=predictors_m1, family="binomial", link="logit",
                where=(svy.col(f"actual_m1_common_{outcome_name}") == True),
            ),
            f"model2_common_{weight_name}": sample_obj.glm.fit(
                y=outcome_name, x=predictors_m2, family="binomial", link="logit",
                where=(svy.col(f"actual_m1_common_{outcome_name}") == True),
            ),
        }

print("Sparse-region-excluded geographic models re-estimated.")
print(f"Excluded a3a codes: {sparse_region_codes}")
print(f"Remaining a3a codes: {remaining_regions}")
print(f"Geographic dataset shape after exclusion: {data_geo.shape}")


print()
print("STEP 2. DEFINE ANALYSIS SETTINGS")

region_reference = 1

expected_regions = remaining_regions

outcomes = [
    "any_obstacle",
    "major_obstacle",
]

weights = [
    "wmedian",
    "wweak",
]

model_names = [
    "model1_full",
    "model1_common",
    "model2_common",
]

model_sample_flags = {
    "model1_full": "actual_m1_full_{outcome}",
    "model1_common": "actual_m1_common_{outcome}",
    "model2_common": "actual_m1_common_{outcome}",
}

region_values = (
    data_geo
    .get_column("a3a")
    .cast(pl.Float64)
    .unique()
    .sort()
    .to_list()
)

if region_values != [
    float(value)
    for value in expected_regions
]:
    raise ValueError(
        "Unexpected establishment-region categories. "
        f"Found: {region_values}"
    )

if region_reference not in expected_regions:
    raise ValueError(
        "The reference region is not present."
    )

print("Reference region: Jakarta (a3a = 1)")
print(f"Number of retained regions: {len(expected_regions)}")
print(f"Pairwise contrasts per family: {len(expected_regions) * (len(expected_regions) - 1) // 2}")


print()
print("STEP 2A. AUDIT ACTUAL INDUSTRY SECTION DISTRIBUTION")
industry_distribution = (
    data_geo.group_by("industry_section")
    .agg(pl.len().alias("observations"))
    .sort("industry_section")
)
industry_distribution_file = output_dir / "24c31_actual_industry_section_distribution_15_regions.csv"
industry_distribution.write_csv(industry_distribution_file)
print(industry_distribution)

# Each regional prediction uses the observed actual-industry mix within that region.
# It is not a common-covariate-distribution standardisation.


print()
print("STEP 3. DEFINE DESIGN MATRIX CONSTRUCTION")


def build_design_matrix(
    sample_frame,
    feature_names,
    counterfactual_region,
):
    matrix_columns = []

    intercept_names = {
        "_intercept_",
        "intercept",
        "const",
        "(intercept)",
    }

    for feature_name in feature_names:

        feature_name = str(feature_name)

        feature_lower = feature_name.lower()

        if feature_lower in intercept_names:

            column_values = np.ones(
                sample_frame.height,
                dtype=np.float64,
            )

        else:

            categorical_match = re.fullmatch(
                r"(a3a|industry_section)_(.+)",
                feature_name,
            )

            if categorical_match is not None:

                variable_name = (
                    categorical_match.group(1)
                )

                category_value = categorical_match.group(2)

                if variable_name == "a3a":

                    column_values = np.full(
                        sample_frame.height,
                        float(counterfactual_region),
                        dtype=np.float64,
                    )

                else:
                    if "industry_section" not in sample_frame.columns:
                        raise KeyError("industry_section not found in estimation data.")
                    section_values = sample_frame.get_column("industry_section").cast(pl.String).to_list()
                    column_values = np.asarray(
                        [1.0 if value == category_value else 0.0 for value in section_values],
                        dtype=np.float64,
                    )

                if variable_name == "a3a":
                    column_values = (
                        np.isclose(column_values, float(category_value))
                    ).astype(np.float64)

            else:

                if feature_name not in sample_frame.columns:
                    raise KeyError(
                        "Feature not found in estimation data: "
                        f"{feature_name}"
                    )

                column_values = np.asarray(
                    sample_frame
                    .get_column(feature_name)
                    .cast(pl.Float64)
                    .to_list(),
                    dtype=np.float64,
                )

        if column_values.shape != (
            sample_frame.height,
        ):
            raise ValueError(
                "Unexpected feature column dimensions: "
                f"{feature_name}"
            )

        if not np.isfinite(column_values).all():
            raise ValueError(
                f"Non-finite values in feature: {feature_name}"
            )

        matrix_columns.append(
            column_values
        )

    matrix = np.column_stack(
        matrix_columns
    )

    if matrix.shape != (
        sample_frame.height,
        len(feature_names),
    ):
        raise ValueError(
            "Unexpected design matrix dimensions."
        )

    return matrix


print("Design matrix construction function ready.")
print("Intercept handling includes _intercept_.")


print()
print("STEP 4. DEFINE WEIGHTED PREDICTION FUNCTION")


def calculate_weighted_prediction(
    matrix,
    beta,
    survey_weights,
    covariance,
):
    linear_predictor = matrix @ beta

    predicted_probability = expit(
        linear_predictor
    )

    weighted_probability = float(
        np.average(
            predicted_probability,
            weights=survey_weights,
        )
    )

    individual_gradient = (
        predicted_probability
        * (1.0 - predicted_probability)
    )[:, None] * matrix

    gradient = np.average(
        individual_gradient,
        axis=0,
        weights=survey_weights,
    )

    variance = float(
        gradient
        @ covariance
        @ gradient
    )

    if variance < -1e-10:
        raise ValueError(
            "Negative prediction variance detected."
        )

    standard_error = float(
        np.sqrt(
            max(variance, 0.0)
        )
    )

    return (
        weighted_probability,
        standard_error,
        gradient,
    )


print("Weighted prediction function ready.")
print("Estimand: within-region weighted average predicted probability.")


print()
print("STEP 5. ESTIMATE REGIONAL EFFECTS")

prediction_rows = []
odds_ratio_rows = []
sample_rows = []
pairwise_rows = []

for outcome_name in outcomes:

    for weight_name in weights:

        model_key = (
            f"{outcome_name}_{weight_name}"
        )

        if model_key not in geo_results:
            raise KeyError(
                f"Model group not found: {model_key}"
            )

        fitted_models = geo_results[
            model_key
        ]

        for model_name in model_names:

            fitted_model_key = (
                f"{model_name}_{weight_name}"
            )

            if fitted_model_key not in fitted_models:
                raise KeyError(
                    "Fitted model not found: "
                    f"{fitted_model_key}"
                )

            fitted_model = fitted_models[
                fitted_model_key
            ]

            fitted = fitted_model.fitted

            sample_flag = model_sample_flags[model_name].format(outcome=outcome_name)

            required_sample_columns = [
                sample_flag,
                outcome_name,
                weight_name,
                "a3a",
                "industry_section",
            ]

            missing_sample_columns = [
                column
                for column in required_sample_columns
                if column not in data_geo.columns
            ]

            if missing_sample_columns:
                raise KeyError(
                    "Required sample columns not found: "
                    f"{missing_sample_columns}"
                )

            sample_frame = (
                data_geo
                .filter(
                    pl.col(sample_flag)
                    & pl.col(outcome_name).is_not_null()
                    & pl.col(weight_name).is_not_null()
                    & pl.col(weight_name).is_finite()
                    & (pl.col(weight_name) > 0)
                )
            )

            if sample_frame.height == 0:
                raise ValueError(
                    "Empty estimation sample: "
                    f"{model_key}, {model_name}"
                )

            survey_weights = np.asarray(
                sample_frame
                .get_column(weight_name)
                .cast(pl.Float64)
                .to_list(),
                dtype=np.float64,
            )

            if not np.isfinite(survey_weights).all():
                raise ValueError(
                    f"Non-finite survey weights: {model_key}"
                )

            if (survey_weights <= 0).any():
                raise ValueError(
                    f"Non-positive survey weights: {model_key}"
                )

            weight_sum = float(
                survey_weights.sum()
            )

            kish_effective_n = float(
                weight_sum ** 2
                / np.sum(survey_weights ** 2)
            )

            feature_names = [
                str(feature)
                for feature in fitted.feature_names
            ]

            if not feature_names:
                raise ValueError(
                    f"No model features found: {model_name}"
                )

            coefficients = fitted.coefs

            coefficient_lookup = {
                str(coefficient.term): coefficient
                for coefficient in coefficients
            }

            missing_features = [
                feature
                for feature in feature_names
                if feature not in coefficient_lookup
            ]

            if missing_features:
                raise ValueError(
                    "Missing model coefficients: "
                    f"{missing_features}"
                )

            beta = np.asarray(
                [
                    coefficient_lookup[
                        feature
                    ].est
                    for feature in feature_names
                ],
                dtype=np.float64,
            )

            covariance = np.asarray(
                fitted.cov_matrix,
                dtype=np.float64,
            )

            if covariance.shape != (
                len(feature_names),
                len(feature_names),
            ):
                raise ValueError(
                    "Covariance matrix dimensions do not match "
                    f"the feature list in {model_name}."
                )

            covariance = (
                covariance + covariance.T
            ) / 2.0

            if not np.isfinite(beta).all():
                raise ValueError(
                    f"Non-finite coefficients in {model_name}."
                )

            if not np.isfinite(covariance).all():
                raise ValueError(
                    f"Non-finite covariance matrix in {model_name}."
                )

            df_den = float(
                fitted._design_df()
            )

            if df_den <= 0:
                raise ValueError(
                    f"Invalid design degrees of freedom: {df_den}"
                )

            critical_value = float(
                t_distribution.ppf(
                    0.975,
                    df_den,
                )
            )

            sample_rows.append(
                {
                    "outcome": outcome_name,
                    "weight": weight_name,
                    "model": model_name,
                    "sample_n": sample_frame.height,
                    "kish_effective_n": kish_effective_n,
                    "weight_sum": weight_sum,
                    "design_df": df_den,
                }
            )

            print()
            print("-" * 72)
            print(
                f"OUTCOME: {outcome_name} | "
                f"WEIGHT: {weight_name} | "
                f"MODEL: {model_name}"
            )
            print(
                f"Sample N: {sample_frame.height}"
            )
            print(
                f"Kish effective N: {kish_effective_n:.2f}"
            )
            print(
                f"Design df: {df_den:.0f}"
            )

            model_prediction_results = {}

            for region_code in expected_regions:

                regional_observed_sample = sample_frame.filter(
                    pl.col("a3a").cast(pl.Int64) == region_code
                )
                if regional_observed_sample.height == 0:
                    raise ValueError(
                        "No estimation-sample observations for region "
                        f"{region_code} in {model_key}/{model_name}."
                    )
                regional_observed_weights = np.asarray(
                    regional_observed_sample.get_column(weight_name).cast(pl.Float64).to_list(),
                    dtype=np.float64,
                )
                regional_observed_weight_sum = float(regional_observed_weights.sum())
                regional_observed_kish_n = float(
                    regional_observed_weight_sum ** 2 / np.sum(regional_observed_weights ** 2)
                )

                # Common-distribution standardization: every counterfactual
                # region is evaluated over the same pooled complete-case sample.
                standardization_sample = sample_frame
                standardization_weights = survey_weights
                matrix = build_design_matrix(
                    sample_frame=standardization_sample,
                    feature_names=feature_names,
                    counterfactual_region=region_code,
                )

                (
                    predicted_probability,
                    prediction_se,
                    gradient,
                ) = calculate_weighted_prediction(
                    matrix=matrix,
                    beta=beta,
                    survey_weights=standardization_weights,
                    covariance=covariance,
                )

                ci_low = max(
                    0.0,
                    predicted_probability
                    - critical_value * prediction_se,
                )

                ci_high = min(
                    1.0,
                    predicted_probability
                    + critical_value * prediction_se,
                )

                model_prediction_results[
                    region_code
                ] = {
                    "probability": predicted_probability,
                    "standard_error": prediction_se,
                    "gradient": gradient,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "regional_sample_n": standardization_sample.height,
                    "regional_kish_effective_n": kish_effective_n,
                    "regional_weight_sum": float(standardization_weights.sum()),
                    "regional_observed_n": regional_observed_sample.height,
                    "regional_observed_kish_n": regional_observed_kish_n,
                    "regional_observed_weight_sum": regional_observed_weight_sum,
                }

            reference_result = (
                model_prediction_results[
                    region_reference
                ]
            )

            # All unordered pairs among the 15 retained regions are tested within each
            # outcome x weight x model family.
            for first_index, first_region in enumerate(expected_regions):
                for second_region in expected_regions[first_index + 1:]:
                    first_result = model_prediction_results[first_region]
                    second_result = model_prediction_results[second_region]

                    pair_difference = (
                        first_result["probability"]
                        - second_result["probability"]
                    )
                    pair_gradient = (
                        first_result["gradient"]
                        - second_result["gradient"]
                    )
                    pair_variance = float(
                        pair_gradient @ covariance @ pair_gradient
                    )

                    if pair_variance < -1e-10:
                        raise ValueError(
                            "Negative pairwise contrast variance detected."
                        )

                    pair_se = float(np.sqrt(max(pair_variance, 0.0)))
                    pair_ci_low = pair_difference - critical_value * pair_se
                    pair_ci_high = pair_difference + critical_value * pair_se

                    if pair_se > 0:
                        pair_t = pair_difference / pair_se
                        pair_p = float(
                            2.0 * t_distribution.sf(abs(pair_t), df_den)
                        )
                    else:
                        pair_p = float("nan")

                    pairwise_rows.append(
                        {
                            "analysis": "actual_industry_sections_excluding_7_sparse_regions",
                            "excluded_region_codes": ";".join(str(x) for x in sparse_region_codes),
                            "retained_region_count": len(expected_regions),
                            "outcome": outcome_name,
                            "weight": weight_name,
                            "model": model_name,
                            "region_1_code": first_region,
                            "region_2_code": second_region,
                            "difference_probability": pair_difference,
                            "difference_percentage_points": pair_difference * 100.0,
                            "standard_error": pair_se,
                            "ci_low": pair_ci_low,
                            "ci_high": pair_ci_high,
                            "ci_low_percentage_points": pair_ci_low * 100.0,
                            "ci_high_percentage_points": pair_ci_high * 100.0,
                            "p_value_raw": pair_p,
                            "design_df": df_den,
                            "ci_method": "conditional_coefficient_covariance_delta_method",
                            "ci_scale": "probability; percentage-point bounds also provided",
                            "estimand": "difference_between_probabilities_standardized_to_common_pooled_covariate_distribution",
                            "inference_scope": "conditional_on_common_pooled_covariate_distribution_and_coefficient_covariance",
                        }
                    )

            for region_code in expected_regions:

                prediction_result = (
                    model_prediction_results[
                        region_code
                    ]
                )

                probability = (
                    prediction_result[
                        "probability"
                    ]
                )

                probability_se = (
                    prediction_result[
                        "standard_error"
                    ]
                )

                probability_gradient = (
                    prediction_result[
                        "gradient"
                    ]
                )

                probability_difference = (
                    probability
                    - reference_result["probability"]
                )

                difference_gradient = (
                    probability_gradient
                    - reference_result["gradient"]
                )

                difference_variance = float(
                    difference_gradient
                    @ covariance
                    @ difference_gradient
                )

                if difference_variance < -1e-10:
                    raise ValueError(
                        "Negative regional contrast variance detected."
                    )

                difference_se = float(
                    np.sqrt(
                        max(
                            difference_variance,
                            0.0,
                        )
                    )
                )

                difference_ci_low = (
                    probability_difference
                    - critical_value * difference_se
                )

                difference_ci_high = (
                    probability_difference
                    + critical_value * difference_se
                )

                if difference_se > 0:

                    difference_t = (
                        probability_difference
                        / difference_se
                    )

                    difference_p = float(
                        2.0
                        * t_distribution.sf(
                            abs(difference_t),
                            df_den,
                        )
                    )

                else:
                    difference_t = float("nan")
                    difference_p = float("nan")

                prediction_rows.append(
                    {
                        "outcome": outcome_name,
                        "weight": weight_name,
                        "model": model_name,
                        "region_code": region_code,
                        "reference_region": region_reference,
                        "sample_n": sample_frame.height,
                        "kish_effective_n": kish_effective_n,
                        "regional_sample_n": prediction_result["regional_sample_n"],
                        "regional_kish_effective_n": prediction_result["regional_kish_effective_n"],
                        "regional_weight_sum": prediction_result["regional_weight_sum"],
                        "regional_observed_n": prediction_result["regional_observed_n"],
                        "regional_observed_kish_n": prediction_result["regional_observed_kish_n"],
                        "regional_observed_weight_sum": prediction_result["regional_observed_weight_sum"],
                        "predicted_probability": probability,
                        "predicted_percentage": (
                            probability * 100.0
                        ),
                        "prediction_se": probability_se,
                        "ci_low": prediction_result["ci_low"],
                        "ci_high": prediction_result["ci_high"],
                        "ci_low_percentage": prediction_result["ci_low"] * 100.0,
                        "ci_high_percentage": prediction_result["ci_high"] * 100.0,
                        "difference_vs_jakarta": probability_difference,
                        "difference_vs_jakarta_pp": (
                            probability_difference * 100.0
                        ),
                        "difference_se": difference_se,
                        "difference_ci_low": difference_ci_low,
                        "difference_ci_high": difference_ci_high,
                        "difference_ci_low_pp": difference_ci_low * 100.0,
                        "difference_ci_high_pp": difference_ci_high * 100.0,
                        "difference_p_value": difference_p,
                        "design_df": df_den,
                        "ci_method": "conditional_coefficient_covariance_delta_method",
                        "estimand": "probability_standardized_to_common_pooled_covariate_distribution",
                        "inference_scope": "conditional_on_common_pooled_covariate_distribution_and_coefficient_covariance",
                    }
                )

                if region_code == region_reference:

                    odds_ratio_rows.append(
                        {
                            "outcome": outcome_name,
                            "weight": weight_name,
                            "model": model_name,
                            "region_code": region_code,
                            "reference_region": region_reference,
                            "log_odds_ratio": 0.0,
                            "odds_ratio": 1.0,
                            "or_ci_low": 1.0,
                            "or_ci_high": 1.0,
                            "p_value": None,
                            "comparison": "Reference",
                        }
                    )

                else:

                    matching_terms = [
                        term
                        for term in feature_names
                        if re.fullmatch(
                            rf"a3a_{re.escape(str(float(region_code)))}",
                            term,
                        )
                    ]

                    if len(matching_terms) != 1:
                        raise ValueError(
                            "Could not uniquely identify the regional "
                            f"coefficient for region {region_code}: "
                            f"{matching_terms}"
                        )

                    regional_coefficient = (
                        coefficient_lookup[
                            matching_terms[0]
                        ]
                    )

                    log_or = float(
                        regional_coefficient.est
                    )

                    log_or_low = float(
                        regional_coefficient.lci
                    )

                    log_or_high = float(
                        regional_coefficient.uci
                    )

                    odds_ratio_rows.append(
                        {
                            "outcome": outcome_name,
                            "weight": weight_name,
                            "model": model_name,
                            "region_code": region_code,
                            "reference_region": region_reference,
                            "log_odds_ratio": log_or,
                            "odds_ratio": float(
                                np.exp(log_or)
                            ),
                            "or_ci_low": float(
                                np.exp(log_or_low)
                            ),
                            "or_ci_high": float(
                                np.exp(log_or_high)
                            ),
                            "p_value": float(
                                regional_coefficient.wald.p_value
                            ),
                            "comparison": "Compared with Jakarta",
                        }
                    )

            print(
                "Regional predictions completed:",
                len(model_prediction_results),
            )


print()
print("STEP 6. BUILD OUTPUT DATAFRAMES")

expected_pair_count = len(expected_regions) * (len(expected_regions) - 1) // 2
if len(pairwise_rows) != len(outcomes) * len(weights) * len(model_names) * expected_pair_count:
    raise ValueError(
        "Unexpected number of pairwise regional contrasts: "
        f"{len(pairwise_rows)}; expected "
        f"{len(outcomes) * len(weights) * len(model_names) * expected_pair_count}"
    )

# Holm step-down adjustment, applied separately within each
# outcome x weight x model family (105 hypotheses per family).
for group_key in sorted({
    (row["outcome"], row["weight"], row["model"])
    for row in pairwise_rows
}):
    family = [
        row for row in pairwise_rows
        if (row["outcome"], row["weight"], row["model"]) == group_key
        and np.isfinite(row["p_value_raw"])
    ]
    if len(family) != expected_pair_count:
        raise ValueError(
            f"A Holm family must contain {expected_pair_count} finite pairwise p-values; "
            f"found {len(family)} for {group_key}."
        )
    family_sorted = sorted(family, key=lambda row: row["p_value_raw"])
    running_max = 0.0
    family_size = len(family_sorted)
    for rank, row in enumerate(family_sorted):
        adjusted = min(1.0, (family_size - rank) * row["p_value_raw"])
        running_max = max(running_max, adjusted)
        row["p_value_holm"] = running_max
    for row in pairwise_rows:
        if (row["outcome"], row["weight"], row["model"]) == group_key:
            row.setdefault("p_value_holm", float("nan"))
            row["holm_family_size"] = family_size

print()
print("STEP 6A. RIDGE-PENALIZED LOGISTIC SENSITIVITY MODEL")

# This is a supplementary model-based sensitivity analysis, not a replacement
# for the design-based svy estimates. Ridge regularization stabilizes regional
# coefficients under sparse events. Its Hessian intervals are approximate and
# are not design-based survey intervals.
ridge_lambda = 1.0
ridge_rows = []
ridge_pair_rows = []


def build_ridge_matrix(frame, region_values, industry_values, numeric_scalers,
                       counterfactual_region=None):
    n = frame.height
    columns = [np.ones(n, dtype=np.float64)]
    names = ["intercept"]
    for variable_name in ["firm_age", "log_employees", "female_owner"]:
        values = np.asarray(frame.get_column(variable_name).cast(pl.Float64).to_list(), dtype=np.float64)
        mean_value, scale_value = numeric_scalers[variable_name]
        columns.append((values - mean_value) / scale_value)
        names.append(variable_name)
    if counterfactual_region is None:
        regions = frame.get_column("a3a").cast(pl.Int64).to_list()
    else:
        regions = [int(counterfactual_region)] * n
    for region_code in region_values:
        if region_code == 1:
            continue
        columns.append(np.asarray([float(value == region_code) for value in regions]))
        names.append(f"region_{region_code}")
    sections = frame.get_column("industry_section").cast(pl.String).to_list()
    for section in industry_values:
        if section == industry_values[0]:
            continue
        columns.append(np.asarray([float(value == section) for value in sections]))
        names.append(f"industry_{section}")
    return np.column_stack(columns), names


for outcome_name in outcomes:
    for weight_name in weights:
        sample_flag = f"actual_m1_common_{outcome_name}"
        ridge_sample = data_geo.filter(
            pl.col(sample_flag)
            & pl.col(outcome_name).is_not_null()
            & pl.col("industry_section").is_not_null()
            & pl.col(weight_name).is_not_null()
            & pl.col(weight_name).is_finite()
            & (pl.col(weight_name) > 0)
        )
        if ridge_sample.height == 0:
            raise ValueError(f"Empty ridge sample: {outcome_name}/{weight_name}")
        raw_weights = np.asarray(ridge_sample.get_column(weight_name).cast(pl.Float64).to_list())
        normalized_weights = raw_weights / raw_weights.mean()
        y = np.asarray(ridge_sample.get_column(outcome_name).cast(pl.Float64).to_list())
        numeric_scalers = {}
        for variable_name in ["firm_age", "log_employees", "female_owner"]:
            values = np.asarray(ridge_sample.get_column(variable_name).cast(pl.Float64).to_list())
            mean_value = float(np.average(values, weights=normalized_weights))
            variance_value = float(np.average((values - mean_value) ** 2, weights=normalized_weights))
            scale_value = float(np.sqrt(variance_value))
            if not np.isfinite(scale_value) or scale_value < 1e-10:
                scale_value = 1.0
            numeric_scalers[variable_name] = (mean_value, scale_value)
        industry_values = sorted(ridge_sample.get_column("industry_section").unique().to_list())
        x, feature_names = build_ridge_matrix(
            ridge_sample, expected_regions, industry_values, numeric_scalers
        )
        penalty = np.eye(x.shape[1], dtype=np.float64) * ridge_lambda
        penalty[0, 0] = 0.0

        def ridge_objective(beta):
            eta = x @ beta
            probabilities = expit(eta)
            objective = np.sum(normalized_weights * (np.logaddexp(0.0, eta) - y * eta))
            objective += 0.5 * float(beta @ penalty @ beta)
            gradient = x.T @ (normalized_weights * (probabilities - y)) + penalty @ beta
            return objective, gradient

        fit = minimize(
            ridge_objective,
            np.zeros(x.shape[1], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not fit.success and np.linalg.norm(fit.jac) > 1e-4:
            raise RuntimeError(f"Ridge logistic optimization failed: {outcome_name}/{weight_name}: {fit.message}")
        beta = fit.x
        fitted_probability = expit(x @ beta)
        hessian = x.T @ ((normalized_weights * fitted_probability * (1.0 - fitted_probability))[:, None] * x) + penalty
        covariance = np.linalg.pinv(hessian, hermitian=True)
        model_predictions = {}
        for region_code in expected_regions:
            x_cf, _ = build_ridge_matrix(
                ridge_sample, expected_regions, industry_values, numeric_scalers,
                counterfactual_region=region_code,
            )
            probabilities = expit(x_cf @ beta)
            mean_probability = float(np.average(probabilities, weights=normalized_weights))
            gradient = np.average(
                (probabilities * (1.0 - probabilities))[:, None] * x_cf,
                axis=0,
                weights=normalized_weights,
            )
            variance = max(float(gradient @ covariance @ gradient), 0.0)
            standard_error = float(np.sqrt(variance))
            model_predictions[region_code] = {
                "probability": mean_probability,
                "se": standard_error,
                "gradient": gradient,
            }
            ridge_rows.append({
                "outcome": outcome_name,
                "weight": weight_name,
                "model": "ridge_model1_common",
                "region_code": region_code,
                "sample_n": ridge_sample.height,
                "kish_effective_n": float(raw_weights.sum() ** 2 / np.sum(raw_weights ** 2)),
                "predicted_probability": mean_probability,
                "predicted_percentage": mean_probability * 100.0,
                "ci_low": max(0.0, mean_probability - 1.96 * standard_error),
                "ci_high": min(1.0, mean_probability + 1.96 * standard_error),
                "ridge_lambda": ridge_lambda,
                "penalized_objective_converged": bool(fit.success),
                "uncertainty_type": "approximate_penalized_hessian_model_based_not_design_based",
                "standardization": "common_pooled_complete_case_covariate_distribution",
            })
        for i, region_1 in enumerate(expected_regions):
            for region_2 in expected_regions[i + 1:]:
                first = model_predictions[region_1]
                second = model_predictions[region_2]
                difference = first["probability"] - second["probability"]
                gradient = first["gradient"] - second["gradient"]
                variance = max(float(gradient @ covariance @ gradient), 0.0)
                se = float(np.sqrt(variance))
                z_value = difference / se if se > 0 else float("nan")
                p_value = float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan")
                ridge_pair_rows.append({
                    "outcome": outcome_name,
                    "weight": weight_name,
                    "model": "ridge_model1_common",
                    "region_1_code": region_1,
                    "region_2_code": region_2,
                    "difference_probability": difference,
                    "difference_percentage_points": difference * 100.0,
                    "standard_error": se,
                    "ci_low_percentage_points": (difference - 1.96 * se) * 100.0,
                    "ci_high_percentage_points": (difference + 1.96 * se) * 100.0,
                    "p_value_raw": p_value,
                    "ridge_lambda": ridge_lambda,
                    "uncertainty_type": "approximate_penalized_hessian_model_based_not_design_based",
                    "standardization": "common_pooled_complete_case_covariate_distribution",
                })

# Holm correction is applied separately to each outcome x weight ridge family.
for family_key in sorted({(r["outcome"], r["weight"]) for r in ridge_pair_rows}):
    family = [r for r in ridge_pair_rows if (r["outcome"], r["weight"]) == family_key]
    family.sort(key=lambda r: r["p_value_raw"] if np.isfinite(r["p_value_raw"]) else 1.0)
    running = 0.0
    m = len(family)
    for rank, row in enumerate(family):
        adjusted = min(1.0, (m - rank) * row["p_value_raw"]) if np.isfinite(row["p_value_raw"]) else float("nan")
        running = max(running, adjusted) if np.isfinite(adjusted) else running
        row["p_value_holm"] = running if np.isfinite(adjusted) else float("nan")
        row["holm_family_size"] = m

ridge_predictions_df = pl.DataFrame(ridge_rows).sort(["outcome", "weight", "region_code"])
ridge_pairwise_df = pl.DataFrame(ridge_pair_rows).sort(["outcome", "weight", "p_value_holm"])
ridge_predictions_df.write_csv(output_dir / "24c31_ridge_standardized_regional_predictions.csv")
ridge_pairwise_df.write_csv(output_dir / "24c31_ridge_standardized_pairwise_holm.csv")
print(f"Ridge standardized predictions: {ridge_predictions_df.height}")
print(f"Ridge pairwise contrasts: {ridge_pairwise_df.height}")

pairwise_df = pl.DataFrame(pairwise_rows).sort(
    ["outcome", "model", "weight", "p_value_holm"]
)


predictions_df = (
    pl.DataFrame(prediction_rows)
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "region_code",
        ]
    )
)

odds_ratios_df = (
    pl.DataFrame(odds_ratio_rows)
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "region_code",
        ]
    )
)

samples_df = (
    pl.DataFrame(sample_rows)
    .sort(
        [
            "outcome",
            "model",
            "weight",
        ]
    )
)

expected_prediction_rows = (
    len(outcomes)
    * len(weights)
    * len(model_names)
    * len(expected_regions)
)

expected_or_rows = (
    len(outcomes)
    * len(weights)
    * len(model_names)
    * len(expected_regions)
)

# Compact primary manuscript extracts: wmedian, model1_full, both outcomes.
primary_predictions_df = predictions_df.filter(
    (pl.col("weight") == "wmedian") & (pl.col("model") == "model1_full")
).select(
    "outcome", "region_code", "reference_region", "sample_n",
    "regional_observed_n", "predicted_percentage", "ci_low_percentage",
    "ci_high_percentage", "difference_vs_jakarta_pp",
    "difference_ci_low_pp", "difference_ci_high_pp",
)
primary_predictions_df.write_csv(
    output_dir / "24c31_manuscript_primary_effect_sizes.csv"
)
primary_holm_pairs_df = pairwise_df.filter(
    (pl.col("weight") == "wmedian")
    & (pl.col("model") == "model1_full")
    & (pl.col("p_value_holm") < 0.05)
).select(
    "outcome", "region_1_code", "region_2_code",
    "difference_percentage_points", "ci_low_percentage_points",
    "ci_high_percentage_points", "p_value_raw", "p_value_holm",
)
primary_holm_pairs_df.write_csv(
    output_dir / "24c31_manuscript_primary_holm_pairs.csv"
)

if predictions_df.height != expected_prediction_rows:
    raise ValueError(
        "Unexpected number of predicted probability rows: "
        f"{predictions_df.height}"
    )

if odds_ratios_df.height != expected_or_rows:
    raise ValueError(
        "Unexpected number of odds ratio rows: "
        f"{odds_ratios_df.height}"
    )

if predictions_df.select(
    pl.col("predicted_probability").is_finite().all()
).item() is not True:
    raise ValueError(
        "Non-finite predicted probabilities detected."
    )

if predictions_df.filter(
    (pl.col("predicted_probability") < 0)
    | (pl.col("predicted_probability") > 1)
).height > 0:
    raise ValueError(
        "Predicted probabilities outside the [0, 1] range."
    )


print()
print("STEP 7. WEIGHT SENSITIVITY COMPARISON")

median_predictions = (
    predictions_df
    .filter(
        pl.col("weight") == "wmedian"
    )
    .select(
        "outcome",
        "model",
        "region_code",
        pl.col(
            "predicted_probability"
        ).alias("probability_wmedian"),
    )
)

weak_predictions = (
    predictions_df
    .filter(
        pl.col("weight") == "wweak"
    )
    .select(
        "outcome",
        "model",
        "region_code",
        pl.col(
            "predicted_probability"
        ).alias("probability_wweak"),
    )
)

weight_comparison_df = (
    median_predictions
    .join(
        weak_predictions,
        on=[
            "outcome",
            "model",
            "region_code",
        ],
        how="inner",
        validate="1:1",
    )
    .with_columns(
        (
            pl.col("probability_wmedian")
            - pl.col("probability_wweak")
        ).alias("difference"),
        (
            (
                pl.col("probability_wmedian")
                - pl.col("probability_wweak")
            ).abs()
            * 100.0
        ).alias("absolute_difference_pp"),
    )
    .sort(
        [
            "outcome",
            "model",
            "absolute_difference_pp",
        ],
        descending=[
            False,
            False,
            True,
        ],
    )
)

expected_weight_rows = (
    len(outcomes)
    * len(model_names)
    * len(expected_regions)
)

if weight_comparison_df.height != expected_weight_rows:
    raise ValueError(
        "Unexpected number of weight comparison rows."
    )


print()
print("STEP 8. CREATE REGIONAL RANKINGS")

ranking_df = (
    predictions_df
    .with_columns(
        pl.col(
            "predicted_probability"
        )
        .rank(
            method="ordinal",
            descending=True,
        )
        .over(
            [
                "outcome",
                "model",
                "weight",
            ]
        )
        .alias("risk_rank")
    )
    .sort(
        [
            "outcome",
            "model",
            "weight",
            "risk_rank",
        ]
    )
)


print()
print("STEP 8A. SUMMARIZE HOLM-SIGNIFICANT PAIRWISE CONTRASTS")

significance_threshold = 0.05
expected_model_weight_combinations = len(model_names) * len(weights)

significant_pairwise_df = (
    pairwise_df
    .filter(
        pl.col("p_value_holm").is_finite()
        & (pl.col("p_value_holm") < significance_threshold)
    )
    .with_columns(
        pl.col("difference_percentage_points")
        .abs()
        .alias("absolute_difference_percentage_points"),
        pl.concat_str(
            [pl.col("model"), pl.col("weight")],
            separator="|",
        ).alias("model_weight_combination"),
    )
)

# Count appearances in significant pairs; each pair contributes to both regions.
region_1_appearances = significant_pairwise_df.select(
    "outcome",
    pl.col("region_1_code").alias("region_code"),
    pl.col("region_2_code").alias("counterpart_region_code"),
    "model_weight_combination",
)
region_2_appearances = significant_pairwise_df.select(
    "outcome",
    pl.col("region_2_code").alias("region_code"),
    pl.col("region_1_code").alias("counterpart_region_code"),
    "model_weight_combination",
)

region_significance_frequency_df = (
    pl.concat([region_1_appearances, region_2_appearances], how="vertical")
    .group_by(["outcome", "region_code"])
    .agg(
        pl.len().alias("significant_pair_appearances"),
        pl.col("counterpart_region_code").n_unique().alias("unique_significant_counterparts"),
        pl.col("model_weight_combination").n_unique().alias("model_weight_combinations_with_significance"),
    )
    .sort(
        ["outcome", "significant_pair_appearances", "region_code"],
        descending=[False, True, False],
    )
)

# A robust pair must be Holm-significant in all six combinations and retain
# the same direction in every combination.
pairwise_robustness_df = (
    pairwise_df
    .group_by(["outcome", "region_1_code", "region_2_code"])
    .agg(
        pl.len().alias("model_weight_combinations_tested"),
        (pl.col("p_value_holm") < significance_threshold).sum().alias("holm_significant_combinations"),
        pl.col("difference_percentage_points").mean().alias("mean_difference_percentage_points"),
        pl.col("difference_percentage_points").min().alias("minimum_signed_difference_percentage_points"),
        pl.col("difference_percentage_points").max().alias("maximum_signed_difference_percentage_points"),
        pl.col("difference_percentage_points").abs().mean().alias("mean_absolute_difference_percentage_points"),
        pl.col("difference_percentage_points").abs().min().alias("minimum_absolute_difference_percentage_points"),
        pl.col("difference_percentage_points").abs().max().alias("maximum_absolute_difference_percentage_points"),
        (pl.col("difference_percentage_points") > 0).sum().alias("positive_direction_combinations"),
        (pl.col("difference_percentage_points") < 0).sum().alias("negative_direction_combinations"),
    )
    .with_columns(
        (
            (pl.col("model_weight_combinations_tested") == expected_model_weight_combinations)
            & (pl.col("holm_significant_combinations") == expected_model_weight_combinations)
            & (
                (pl.col("positive_direction_combinations") == expected_model_weight_combinations)
                | (pl.col("negative_direction_combinations") == expected_model_weight_combinations)
            )
        ).alias("robust_significant_all_six")
    )
    .sort(
        ["outcome", "robust_significant_all_six", "mean_absolute_difference_percentage_points"],
        descending=[False, True, True],
    )
)

robust_pairwise_df = pairwise_robustness_df.filter(pl.col("robust_significant_all_six"))

# Ten largest Holm-significant contrasts within each outcome x weight x model family.
largest_significant_pairwise_df = (
    significant_pairwise_df
    .sort(
        ["outcome", "weight", "model", "absolute_difference_percentage_points"],
        descending=[False, False, False, True],
    )
    .group_by(["outcome", "weight", "model"], maintain_order=True)
    .head(10)
    .sort(
        ["outcome", "weight", "model", "absolute_difference_percentage_points"],
        descending=[False, False, False, True],
    )
)

region_frequency_file = output_dir / "24c31_region_holm_significance_frequency.csv"
robust_pairs_file = output_dir / "24c31_pairwise_robust_all_six_combinations.csv"
largest_pairs_file = output_dir / "24c31_largest_holm_significant_pairwise_contrasts.csv"

region_significance_frequency_df.write_csv(region_frequency_file)
pairwise_robustness_df.write_csv(robust_pairs_file)
largest_significant_pairwise_df.write_csv(largest_pairs_file)

print(f"Holm-significant pairwise rows: {significant_pairwise_df.height}")
print(f"Robust pairs significant in all six combinations: {robust_pairwise_df.height}")
print("Region frequencies count appearances in significant pairs, not unique regions.")

print()
print("ROBUST PAIRWISE CONTRASTS BY OUTCOME")
for outcome_name in outcomes:
    robust_outcome = robust_pairwise_df.filter(pl.col("outcome") == outcome_name).sort(
        "mean_absolute_difference_percentage_points", descending=True
    )
    print(f"{outcome_name}: {robust_outcome.height} robust pairs")
    print(robust_outcome.select(
        "region_1_code", "region_2_code", "mean_difference_percentage_points",
        "minimum_absolute_difference_percentage_points", "maximum_absolute_difference_percentage_points",
    ).head(10))

print()
print("REGIONS MOST FREQUENTLY INVOLVED IN HOLM-SIGNIFICANT PAIRS")
for outcome_name in outcomes:
    print(outcome_name)
    print(region_significance_frequency_df.filter(pl.col("outcome") == outcome_name).head(10))

print()
print("MANUSCRIPT-READY SUMMARY TEXT (REGION CODES REQUIRE LABEL CROSSWALK)")
for outcome_name in outcomes:
    robust_outcome = robust_pairwise_df.filter(pl.col("outcome") == outcome_name).sort(
        "mean_absolute_difference_percentage_points", descending=True
    )
    if robust_outcome.height == 0:
        print(
            f"For {outcome_name.replace('_', ' ')}, no regional pair remained "
            "statistically significant after Holm adjustment in all six model-weight combinations."
        )
    else:
        example_text = "; ".join(
            f"a3a {int(row['region_1_code'])} versus a3a {int(row['region_2_code'])} "
            f"(mean absolute difference {row['mean_absolute_difference_percentage_points']:.2f} percentage points)"
            for row in robust_outcome.head(3).iter_rows(named=True)
        )
        print(
            f"For {outcome_name.replace('_', ' ')}, {robust_outcome.height} regional pairs "
            "remained Holm-significant in all six model-weight combinations. "
            f"The largest robust contrasts included: {example_text}."
        )


print()
print("STEP 9. SAVE OUTPUT FILES")

prediction_file = (
    output_dir
    / "24c31_regional_adjusted_predicted_probabilities.csv"
)

odds_ratio_file = (
    output_dir
    / "24c31_regional_odds_ratios.csv"
)

sample_file = (
    output_dir
    / "24c31_regional_prediction_sample_summary.csv"
)

weight_comparison_file = (
    output_dir
    / "24c31_regional_weight_sensitivity.csv"
)

ranking_file = (
    output_dir
    / "24c31_regional_risk_rankings.csv"
)

pairwise_file = (
    output_dir
    / "24c31_regional_all_pairwise_contrasts_holm.csv"
)

predictions_df.write_csv(
    prediction_file
)

odds_ratios_df.write_csv(
    odds_ratio_file
)

samples_df.write_csv(
    sample_file
)

weight_comparison_df.write_csv(
    weight_comparison_file
)

ranking_df.write_csv(
    ranking_file
)

pairwise_df.write_csv(
    pairwise_file
)

# Descriptive forest plot for the prespecified primary display:
# major_obstacle, wmedian, model1_full. Each regional prediction uses its
# common pooled covariate distribution. Error bars are conditional
# coefficient-covariance delta-method intervals, not full design CIs.
forest_data = (
    predictions_df
    .filter(
        (pl.col("outcome") == "major_obstacle")
        & (pl.col("weight") == "wmedian")
        & (pl.col("model") == "model1_full")
    )
    .sort("predicted_probability")
)

forest_codes = forest_data.get_column("region_code").to_list()
forest_estimates = (
    forest_data.get_column("predicted_percentage").to_numpy()
)
forest_low = (
    forest_data.get_column("ci_low_percentage").to_numpy()
)
forest_high = (
    forest_data.get_column("ci_high_percentage").to_numpy()
)

fig, ax = plt.subplots(figsize=(9, 9))
y_positions = np.arange(len(forest_codes))
ax.errorbar(
    forest_estimates,
    y_positions,
    xerr=[forest_estimates - forest_low, forest_high - forest_estimates],
    fmt="o",
    capsize=3,
)
ax.set_yticks(y_positions)
ax.set_yticklabels([str(code) for code in forest_codes])
ax.set_xlabel("Adjusted predicted probability (%)")
ax.set_ylabel("Establishment region code (a3a)")
ax.set_title(
    "Major financing obstacle by establishment region\n"
    "wmedian; model1_full; common-distribution standardized predictions; conditional 95% CIs"
)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
forest_plot_file = output_dir / "24c31_major_obstacle_regional_forest_plot.png"
fig.savefig(forest_plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)


print()
print("STEP 10. FINAL SUMMARY")

print()
print("PAIRWISE CONTRAST SUMMARY")
print(f"Total pairwise rows: {pairwise_df.height}")
print(f"Holm families: outcome x weight x model; {expected_pair_count} tests per family")
print("Confidence intervals: conditional coefficient-covariance delta method")
print("Regional predictions use one common pooled covariate distribution within each model sample.")
print("A ridge-penalized logistic sensitivity model is also reported with approximate model-based intervals.")
print("Probability CIs are stored in both 0-1 and 0-100 scales.")
print("Pairwise CIs are stored in probability and percentage-point scales.")
print("Intervals are conditional delta-method intervals; they do not include covariate-distribution uncertainty or replicate-weight variance estimation.")

print()
print("MODEL SAMPLE SUMMARY")
print(samples_df)

print()
print("TOP FIVE REGIONS BY PREDICTED PROBABILITY")

for outcome_name in outcomes:

    for weight_name in weights:

        for model_name in model_names:

            top_regions = (
                ranking_df
                .filter(
                    (pl.col("outcome") == outcome_name)
                    & (pl.col("weight") == weight_name)
                    & (pl.col("model") == model_name)
                )
                .head(5)
                .select(
                    "region_code",
                    "predicted_percentage",
                    "ci_low_percentage",
                    "ci_high_percentage",
                    "risk_rank",
                )
            )

            print()
            print(
                f"{outcome_name} | "
                f"{weight_name} | "
                f"{model_name}"
            )

            print(top_regions)


print()
print("OUTPUT FILES")

for output_file in [
    prediction_file,
    odds_ratio_file,
    sample_file,
    weight_comparison_file,
    ranking_file,
    pairwise_file,
    region_frequency_file,
    robust_pairs_file,
    largest_pairs_file,
    forest_plot_file,
    industry_distribution_file,
    output_dir / "24c31_ridge_standardized_regional_predictions.csv",
    output_dir / "24c31_ridge_standardized_pairwise_holm.csv",
    output_dir / "24c31_measurement_survey_design_audit.csv",
    output_dir / "24c31_regional_context_source_register.csv",
    output_dir / "24c31_manuscript_primary_effect_sizes.csv",
    output_dir / "24c31_manuscript_primary_holm_pairs.csv",
]:
    print(output_file)


print()
print("=" * 78)
print("24C-31 COMPLETED")
print("=" * 78)

region_exclusion_audit = original_region_counts.with_columns(
    pl.col("a3a").cast(pl.Int64).alias("region_code")
).with_columns(
    pl.col("region_code").is_in(sparse_region_codes).alias("excluded")
).drop("a3a")

region_exclusion_audit.write_csv(
    output_dir / "24c31_sparse_region_exclusion_audit.csv"
)

print("Sparse-region exclusion audit saved.")
