library(arrow)
library(survey)

project_root <- normalizePath(
  "D:/portofolio/02_msme_access_to_finance",
  winslash = "/",
  mustWork = TRUE
)

setwd(project_root)

input_path <- file.path(
  project_root,
  "data",
  "processed",
  "outcome_robustness",
  "k30_k162_raoscott_input.parquet"
)

output_dir <- file.path(
  project_root,
  "data",
  "processed",
  "outcome_robustness"
)

if (!file.exists(input_path)) {
  stop("Input Parquet file does not exist: ", input_path)
}

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

data <- read_parquet(input_path)

# Explicitly define the singleton-PSU treatment.
options(survey.lonely.psu = "adjust")

# Record the actual setting used.
cat(
  "R survey lonely PSU setting:",
  getOption("survey.lonely.psu"),
  "\n"
)

cat(
  "Survey package version:",
  as.character(packageVersion("survey")),
  "\n"
)

# Verify establishment identifiers.
if (anyDuplicated(data$idstd) > 0) {
  stop("idstd is not unique. Verify the sampling unit.")
}

# Construct the survey design.
design <- svydesign(
  ids = ~idstd,
  strata = ~strata,
  weights = ~wmedian,
  data = data,
  nest = TRUE
)

# Define the complete-response analytical domain.
analysis_design <- subset(
  design,
  !is.na(obstacle_major) &
  !is.na(credit_application)
)

# Weighted prevalence.
prevalence <- svyby(
  ~obstacle_major,
  ~credit_application,
  design = analysis_design,
  FUN = svymean,
  vartype = c("se", "ci"),
  na.rm = TRUE
)

# Rao-Scott adjusted F test.
rao_scott <- svychisq(
  ~obstacle_major + credit_application,
  design = analysis_design,
  statistic = "F"
)

# Unweighted cross-tabulation.
unweighted_table <- with(
  analysis_design$variables,
  table(credit_application, obstacle_major)
)

cat("\nUNWEIGHTED CROSS-TABULATION\n")
print(unweighted_table)

cat("\nWEIGHTED OBSTACLE PREVALENCE\n")
print(prevalence)

cat("\nRAO-SCOTT TEST\n")
print(rao_scott)

cat("\nANALYTICAL SAMPLE SIZE\n")
print(nrow(analysis_design$variables))

# Save results.
write.csv(
  as.data.frame(prevalence),
  file.path(
    output_dir,
    "k30_k162_weighted_prevalence_adjust.csv"
  ),
  row.names = FALSE
)

write.csv(
  as.data.frame(unweighted_table),
  file.path(
    output_dir,
    "k30_k162_unweighted_crosstab_adjust.csv"
  )
)

capture.output(
  rao_scott,
  file = file.path(
    output_dir,
    "k30_k162_rao_scott_adjust.txt"
  )
)

capture.output(
  sessionInfo(),
  file = file.path(
    output_dir,
    "k30_k162_session_info.txt"
  )
)

cat("\nAnalysis completed.\n")