from pathlib import Path
import os
import runpy
import shutil
import tempfile
import polars as pl

project_dir = Path(__file__).resolve().parents[1]
script55_path = project_dir / "src" / "55_survey_weighted_logistic_regression.py"
output_dir = project_dir / "data" / "processed" / "outcome_robustness"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("24C-28: ACTUAL INDUSTRY CLASSIFICATION ROBUSTNESS (d1a2_v4)")
print("=" * 78)
print("The original script is executed unchanged. This script adds a separate")
print("robustness specification using actual ISIC Rev. 4 industry sections.")

# Script 55 is executed to rebuild its in-memory survey objects. Preserve and
# restore its existing output folders so this robustness run does not overwrite
# the user's previously audited results.
processed_dir = project_dir / "data" / "processed"
output_folders = [
    processed_dir / "script55_survey_weighted_logistic_regression",
    processed_dir / "outcome_robustness",
]
original_cwd = Path.cwd()
with tempfile.TemporaryDirectory(prefix="24c28_backup_") as backup_name:
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

required_objects = ["data_geo", "geo_designs", "svy"]
missing_objects = [name for name in required_objects if name not in namespace]
if missing_objects:
    raise RuntimeError(f"Required objects missing from script 55: {missing_objects}")

pl = namespace["pl"]
svy = namespace["svy"]
data_geo = namespace["data_geo"]
geo_designs = namespace["geo_designs"]

required_columns = [
    "idstd", "strata", "wmedian", "wweak", "d1a2_v4", "a3a",
    "major_obstacle", "any_obstacle", "firm_age", "log_employees",
    "female_owner", "bank_account", "model1_complete", "model2_complete",
]
missing_columns = [name for name in required_columns if name not in data_geo.columns]
if missing_columns:
    raise ValueError(f"Required columns missing from geographic analytic data: {missing_columns}")

# Convert the reported ISIC Rev. 4 code to a two-digit division.
# The first two digits identify the ISIC division; divisions are then grouped
# into the standard ISIC Rev. 4 sections to avoid highly sparse detailed codes.
industry_text = (
    pl.col("d1a2_v4")
    .cast(pl.String)
    .str.strip_chars()
    .str.replace(r"\.0$", "")
)
industry_division = (
    industry_text
    .str.extract(r"^(\d{1,4})", 1)
    .str.zfill(2)
    .str.slice(0, 2)
    .cast(pl.Int64, strict=False)
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
    .otherwise(None)
    .alias("industry_section")
)

data_alt = data_geo.with_columns([
    industry_division.alias("industry_division"),
    section_expr,
])

invalid_industry = data_alt.filter(
    pl.col("d1a2_v4").is_not_null() & pl.col("industry_section").is_null()
)
if invalid_industry.height:
    invalid_values = (
        invalid_industry
        .select("d1a2_v4")
        .unique()
        .head(30)
        .to_series()
        .to_list()
    )
    raise ValueError(
        "Some nonmissing d1a2_v4 values could not be mapped to an ISIC Rev. 4 "
        f"section. Inspect these values before estimation: {invalid_values}"
    )

print("\nISIC SECTION DISTRIBUTION")
section_distribution = (
    data_alt.group_by("industry_section")
    .agg(pl.len().alias("observations"))
    .sort("industry_section")
)
print(section_distribution)
section_distribution.write_csv(output_dir / "24c28_actual_industry_section_distribution.csv")

base_vars = ["firm_age", "log_employees", "female_owner", "industry_section", "a3a", "idstd", "strata"]
model2_vars = base_vars + ["bank_account"]
for outcome in ["major_obstacle", "any_obstacle"]:
    # Preserve the original model's complete-case domains, then require a valid
    # actual-industry code. This isolates the industry-classification change.
    data_alt = data_alt.with_columns([
        pl.all_horizontal([
            pl.col("model1_complete"),
            pl.col("d1a2_v4").is_not_null(),
            pl.col("industry_section").is_not_null(),
            pl.col("a3a").is_not_null(),
            pl.col(outcome).is_not_null(),
        ]).alias(f"alt_m1_{outcome}_complete"),
        pl.all_horizontal([
            pl.col("model2_complete"),
            pl.col("d1a2_v4").is_not_null(),
            pl.col("industry_section").is_not_null(),
            pl.col("a3a").is_not_null(),
            pl.col(outcome).is_not_null(),
        ]).alias(f"alt_m2_{outcome}_complete"),
    ])

alt_samples = {}
for weight_name, design_obj in geo_designs.items():
    alt_samples[weight_name] = svy.Sample(data_alt, design_obj)

coefficient_rows = []
sample_rows = []
wald_rows = []

for outcome in ["major_obstacle", "any_obstacle"]:
    for weight_name, sample_obj in alt_samples.items():
        predictors_m1 = [
            "firm_age", "log_employees", "female_owner",
            svy.Cat("industry_section"), svy.Cat("a3a"),
        ]
        predictors_m2 = predictors_m1 + ["bank_account"]
        model_specs = [
            ("model1_full", predictors_m1, f"alt_m1_{outcome}_complete"),
            ("model1_common", predictors_m1, f"alt_m2_{outcome}_complete"),
            ("model2_common", predictors_m2, f"alt_m2_{outcome}_complete"),
        ]
        for model_name, predictors, flag_name in model_specs:
            fitted_model = sample_obj.glm.fit(
                y=outcome,
                x=predictors,
                family="binomial",
                link="logit",
                where=(svy.col(flag_name) == True),
            )
            fitted = fitted_model.fitted
            for coef in fitted.coefs:
                coefficient_rows.append({
                    "outcome": outcome,
                    "weight": weight_name,
                    "model": model_name,
                    "term": coef.term,
                    "estimate": coef.est,
                    "std_error": coef.se,
                    "conf_low": coef.lci,
                    "conf_high": coef.uci,
                    "p_value": coef.wald.p_value,
                })
            try:
                wald_result = fitted.term_test("a3a")
                wald_text = str(wald_result)
            except Exception as exc:
                wald_text = f"Wald test unavailable: {type(exc).__name__}: {exc}"
            wald_rows.append({
                "outcome": outcome,
                "weight": weight_name,
                "model": model_name,
                "region_joint_test": wald_text,
            })
            flag = flag_name
            subset = data_alt.filter(pl.col(flag) == True)
            weight_col = pl.col(weight_name)
            event_col = pl.col(outcome)
            totals = subset.select([
                pl.len().alias("n"),
                event_col.sum().alias("events"),
                (weight_col * event_col).sum().alias("weighted_events"),
                weight_col.sum().alias("weight_total"),
                pl.col("a3a").n_unique().alias("n_regions"),
            ]).row(0, named=True)
            sample_rows.append({
                "outcome": outcome,
                "weight": weight_name,
                "model": model_name,
                "observations": totals["n"],
                "events": totals["events"],
                "weighted_prevalence": totals["weighted_events"] / totals["weight_total"],
                "n_regions": totals["n_regions"],
                "industry_sections": subset.get_column("industry_section").n_unique(),
            })

coefficients = pl.DataFrame(coefficient_rows)
samples = pl.DataFrame(sample_rows)
wald = pl.DataFrame(wald_rows)
coefficients.write_csv(output_dir / "24c28_actual_industry_section_coefficients.csv")
samples.write_csv(output_dir / "24c28_actual_industry_section_sample_audit.csv")
wald.write_csv(output_dir / "24c28_actual_industry_section_region_wald_tests.csv")

print("\nMODEL SAMPLE AUDIT")
print(samples.sort(["outcome", "weight", "model"]))
print("\nREGIONAL COEFFICIENTS")
print(coefficients.filter(pl.col("term").str.starts_with("a3a_")).sort(["outcome", "weight", "model", "term"]))
print("\nREGIONAL JOINT TESTS")
print(wald)
print("\n24C-28 COMPLETED")
print(f"Outputs saved under: {output_dir}")
