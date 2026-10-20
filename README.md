# Indonesia 2023 WBES: MSME Access to Finance

A data analysis project examining access-to-finance obstacles among Indonesian firms using the **2023 World Bank Enterprise Surveys (WBES)**. The project uses Python and Polars to explore survey-weighted estimates, compare perceived financing obstacles by credit status, and assess the sensitivity of descriptive findings to alternative survey-weight specifications.

## Project Overview

Access to finance is an important consideration for firms seeking to operate, invest, and grow. However, firms' perceived financing obstacles may differ according to their existing credit arrangements and the survey-weight specification used to estimate population-level indicators.

This project investigates the prevalence of major or very severe access-to-finance obstacles and examines how that prevalence differs between firms with and without an existing loan or line of credit.

The project emphasizes:

- Data ingestion and validation using Python and Polars.
- Descriptive analysis of access-to-finance variables.
- Survey-weighted prevalence estimates.
- Sensitivity analysis using alternative survey-weight specifications.
- Comparison of perceived financing obstacles by existing credit status.
- Transparent documentation of sample restrictions and methodological limitations.

The analysis is descriptive and exploratory. Associations should not be interpreted as causal effects.

## Research Questions

1. What proportion of surveyed firms report access to finance as a major or very severe obstacle?
2. How sensitive is the estimated prevalence to strict-, median-, and weak-eligibility survey weights?
3. How does the prevalence of major or very severe financing obstacles differ between firms with and without an existing loan or line of credit?
4. Does the direction of the observed association remain consistent across alternative weighting specifications?

## Data Source and Scope

- **Source:** [World Bank Enterprise Surveys — Indonesia 2023](https://microdata.worldbank.org/catalog/6449/)
- **Original dataset:** `Indonesia-2023-full-data.dta`
- **Original sample:** 2,955 observations.
- **Original variables:** 431 columns.
- **Source format:** Stata `.dta`.
- **Working format:** Parquet.
- **Main tools:** Python, Polars, Pyreadstat, scikit-learn, and NumPy.

The survey implementation covers formal or registered private-sector businesses with at least five employees in eligible industries. Based on the local dataset's `stratificationpanelcode`, 2,684 observations are classified as fresh interviews and 271 as panel interviews. The relationship between this variable and the `panel` variable described in the official survey documentation has not yet been independently verified.

The survey universe should not be treated as equivalent to all Indonesian MSMEs. Informal businesses, firms with fewer than five employees, and businesses outside the eligible industries may fall outside the survey's sampling frame. Therefore, the findings should not automatically be generalized to every MSME in Indonesia.

Raw survey data are not included in this repository. Users must obtain the dataset from the official survey catalog and comply with the applicable access conditions and data-use requirements.

## Key Variables

| Variable | Interpretation | Coding used in this project |
|---|---|---|
| `k82` | Existing loan or line of credit | Codes 1–3 indicate some form of credit; code 4 indicates no credit; `-9` means don't know. |
| `k162` | Application for a loan or line of credit | Codes 1–3 indicate an application; code 4 indicates no application; `-9` means don't know. |
| `k20a1` | Outcome of the most recent credit application | Code 1: fully approved; 2: partly approved; 3: rejected; 4: withdrawn; `-6`: in process; `-9`: don't know. |
| `k17` | Main reason for not applying for credit | Codes 1–7 cover no need or sufficient capital, complex procedures, unfavorable rates, collateral requirements, insufficient size or maturity, expected rejection, and other reasons. |
| `k30` | Perceived access-to-finance obstacle | 0: no obstacle; 1: minor; 2: moderate; 3: major; 4: very severe; `-7`: does not apply; `-9`: don't know. |
| `wstrict` | Strict-eligibility survey weight | Alternative weight specification used for sensitivity analysis. |
| `wmedian` | Median-eligibility survey weight | Provisional primary specification for the main descriptive comparisons. |
| `wweak` | Weak-eligibility survey weight | Alternative weight specification used for sensitivity analysis. |

Variable definitions and special response codes should be checked against the official questionnaire, data dictionary, and implementation documentation before extending the analysis.

Negative response codes must be interpreted variable by variable. They should not automatically be converted to zero or treated as interchangeable missing values.

## Outcome Definition

The main outcome is whether a firm reports access to finance as a major or very severe obstacle.

The analysis uses the following response categories:

| `k30` code | Interpretation | Binary outcome |
|---|---|---:|
| 0 | No obstacle | 0 |
| 1 | Minor obstacle | 0 |
| 2 | Moderate obstacle | 0 |
| 3 | Major obstacle | 1 |
| 4 | Very severe obstacle | 1 |
| -9 | Don't know | Excluded |
| -7 | Does not apply | Excluded |

The binary outcome, `major_obstacle`, equals 1 when `k30` is 3 or 4 and equals 0 when `k30` is 0, 1, or 2.

This outcome measures the prevalence of firms reporting access to finance as a major or very severe obstacle. It does not measure the amount of unmet credit demand or identify the specific cause of financing difficulties.

## Methodology

### 1. Data Preparation and Validation

The analysis uses a processed Parquet dataset generated from the original survey data.

The processing workflow is intended to preserve the original observations and variables while providing a consistent input for subsequent analyses.

The analytical scripts validate the presence of required variables, exclude responses that are invalid for the relevant analysis, and apply eligibility criteria to survey weights.

For the overall finance-obstacle prevalence analysis, the eligible sample includes valid `k30` responses and positive, non-missing values for all three weight specifications.

For the credit-status comparison, the analysis uses a common complete-case sample with valid responses for both `k82` and `k30`, as well as positive, non-missing values for all three weights.

Using a common eligible sample allows the alternative weight specifications to be compared on the same observations.

### 2. Weighted Estimation

The weighted prevalence of major or very severe financing obstacles is calculated as:

\[
\widehat{P} =
\frac{\sum_{i=1}^{n} w_i y_i}
{\sum_{i=1}^{n} w_i}
\]

where:

- \(y_i\) is the binary outcome for firm \(i\);
- \(w_i\) is the selected survey weight;
- \(n\) is the number of eligible observations.

Weighted estimates are compared with unweighted sample proportions to examine how weighting affects the descriptive results.

### 3. Weight Sensitivity Analysis

Three survey-weight specifications are evaluated:

- `wstrict`: strict-eligibility weight.
- `wmedian`: median-eligibility weight.
- `wweak`: weak-eligibility weight.

The median-eligibility weight is used provisionally as the primary specification. The strict- and weak-eligibility weights provide sensitivity checks.

This is an analytical choice that should be confirmed against the official survey documentation and the target population of the research question. The use of a provisional primary weight does not establish that it is necessarily the most appropriate specification for every analysis.

### 4. Credit-Status Comparison

Existing credit status is defined using `k82`:

- **Has credit:** codes 1, 2, or 3.
- **No credit:** code 4.

The analysis estimates the weighted prevalence of major or very severe financing obstacles separately for these two groups.

The between-group difference is defined as:

\[
\Delta =
\widehat{P}_{\text{Has credit}}
-
\widehat{P}_{\text{No credit}}
\]

A positive difference indicates a higher estimated prevalence of major or very severe financing obstacles among firms with existing credit.

This comparison is descriptive. It does not identify the causal effect of obtaining credit on firms' financing constraints.

## Results

### 1. Finance-Obstacle Prevalence and Weight Sensitivity

Of the 2,955 observations in the original dataset, 2,833 have valid responses for the finance-obstacle outcome and satisfy the common weight-eligibility criteria. The remaining 122 observations are excluded from this analysis.

The unweighted proportion reporting access to finance as a major or very severe obstacle is **14.68%**.

| Weight specification | Observations | Weighted estimate |
|---|---:|---:|
| `wstrict` | 2,833 | 12.67% |
| `wmedian` | 2,833 | 14.94% |
| `wweak` | 2,833 | 11.66% |

The estimates range from 11.66% to 14.94%, a difference of **3.28 percentage points** between the highest and lowest estimates.

These results show that the estimated prevalence is sensitive to the choice of survey-weight specification. The median-eligibility estimate is close to the unweighted proportion, but this similarity does not establish that `wmedian` is necessarily the most appropriate weight.

The choice of the primary weight should be justified by the survey documentation and the population to which the analysis intends to generalize.

### 2. Credit Status and Perceived Finance Obstacles

The credit-status comparison uses a common analytical sample of **2,491 firms** with valid responses to both `k82` and `k30` and positive, non-missing values for all three survey weights. This represents 464 fewer observations than the original dataset.

Under the provisional primary specification, `wmedian`, the weighted distribution of perceived finance obstacles is:

| Perceived obstacle | Has credit (%) | No credit (%) |
|---|---:|---:|
| No obstacle | 35.13 | 39.57 |
| Minor obstacle | 16.95 | 27.37 |
| Moderate obstacle | 23.06 | 20.73 |
| Major obstacle | 16.52 | 7.26 |
| Very severe obstacle | 8.34 | 5.07 |

The weighted proportion reporting a major or very severe obstacle is **24.86%** among firms with credit, compared with **12.33%** among firms without credit.

The estimated difference, calculated before rounding, is approximately **12.54 percentage points**.

This pattern suggests that having an existing loan or line of credit does not necessarily mean that a firm experiences no financing constraints. Firms with existing credit may still encounter difficulties obtaining additional financing or securing more favorable terms. These are possible explanations, not mechanisms directly tested by the current analysis.

### 3. Sensitivity of the Credit-Status Comparison

To evaluate whether the observed difference depends on the selected survey weight, the credit-status comparison is repeated under all three weight specifications.

| Weight specification | Has credit (%) | No credit (%) | Difference (percentage points) |
|---|---:|---:|---:|
| `wstrict` | 23.19 | 9.63 | 13.56 |
| `wmedian` | 24.86 | 12.33 | 12.54 |
| `wweak` | 21.71 | 8.73 | 12.98 |

The direction of the association is consistent across all three specifications: firms with credit have a higher estimated prevalence of major or very severe financing obstacles than firms without credit.

The estimated differences range from approximately 12.54 to 13.56 percentage points. The direction and approximate magnitude of the descriptive association are relatively stable across the tested weight specifications, although the individual group estimates vary.

These results support reporting the association as a descriptive pattern that is consistent in direction across the three tested weights. They do not demonstrate statistical significance or establish a causal relationship.

## Limitations

Several limitations should be considered when interpreting these findings.

### Survey-Weight Selection

The analysis compares strict-, median-, and weak-eligibility weights. The median-eligibility weight is currently used as the provisional primary specification, but its suitability should be confirmed against the official survey documentation and the intended target population.

### Sample Restrictions

The overall prevalence analysis includes 2,833 observations, while the credit-status comparison includes 2,491 complete-case observations. The exclusion of observations with invalid or missing responses may affect the composition of the analytical sample and the resulting estimates.

### Descriptive, Not Causal

The analysis identifies an association between credit status and perceived finance obstacles. It does not establish that having credit causes firms to experience more severe financing constraints. Differences in firm characteristics, financing demand, credit conditions, or other factors may contribute to the observed pattern.

### Statistical Uncertainty

Survey-design-adjusted standard errors and confidence intervals have not yet been calculated. The reported differences should therefore not be interpreted as statistically significant.

### Generalizability

The results relate to the surveyed firms represented by the relevant WBES sample and analytical eligibility criteria. They should not automatically be generalized to all Indonesian MSMEs, including firms outside the survey's sampling frame.

### Scope of the Outcome

The binary outcome combines major and very severe obstacles into one category. It does not distinguish the underlying reasons for financing difficulties or measure the amount of unmet credit demand.

### Survey Design and Stratification

The dataset contains 1,111 distinct `strata` values, including many strata represented by a single sampled observation. Diagnostics indicate that the combination of `a4a`, `a2`, `a6a`, and `stratificationpanelcode` maps one-to-one to `strata` in this dataset.

This is a consistency check, not proof that all variance-estimation details are correctly specified. Design-based standard errors and confidence intervals should not be reported until the sampling design, strata, panel treatment, and variance-estimation method have been validated against the official survey documentation.

### Firm-Size Coding

The inspected implementation documentation describes `a6a` as the sampling-size category for small, medium, and large firms. The variable `a6b` contains screener categories for micro, small, medium, and large firms, but no observations with `a6b = 0` occur in this dataset.

The inspected official dictionary does not label codes 4–6 in `stratificationsizecode`. Their meanings remain unresolved and should not be guessed or silently recoded.

## Project Structure

The repository contains Python scripts for data inspection, data preparation, exploratory analysis, survey-weight evaluation, and exploratory modeling.

```text
02_msme_access_to_finance/
├── data/
│   ├── raw/                         # Source data; excluded from version control
│   └── processed/                   # Processed data and analytical outputs
├── src/
│   ├── 01_inspect_dataset.py
│   ├── 02_find_finance_variables.py
│   ├── 03_inspect_finance_value_labels.py
│   ├── 04_create_variable_dictionary.py
│   ├── 05_load_stata_to_polars.py
│   ├── 06_save_to_parquet.py
│   ├── 07_eda_finance_access.py
│   ├── 08_clean_eda_metrics.py
│   ├── 09_credit_vs_finance_obstacle.py
│   ├── 10_inspect_survey_weights.py
│   ├── 11_weighted_credit_vs_obstacle.py
│   ├── 12_inspect_firm_characteristics.py
│   ├── 13_inspect_size_labels.py
│   ├── 14_size_vs_finance_obstacle.py
│   ├── 15_check_size_distribution.py
│   ├── 16_prepare_model_dataset.py
│   ├── 17_train_logistic_regression.py
│   ├── 18_analyze_logistic_coefficients.py
│   ├── 19_inspect_model_features.py
│   ├── 20_weighted_obstacle_by_region_industry.py
│   ├── 21_weight_sensitivity_check.py
│   ├── 22_create_weight_sensitivity_report.py
│   ├── 23_audit_sensitivity_sample.py
│   ├── 24_inspect_survey_design.py
│   ├── 25_validate_strata_structure.py
│   ├── 26_validate_strata_with_panel.py
│   ├── 27_inspect_size_code_mapping.py
│   ├── 28_inspect_original_size_labels.py
│   ├── 29_inspect_sampling_metadata.py
│   ├── 30_inspect_size_code_crosswalk.py
│   ├── 31_weighted_credit_obstacle_analysis.py
│   └── 32_credit_obstacle_weight_sensitivity.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `src/` directory contains scripts covering the following stages:

1. **Data inspection and preparation:** examining the source dataset, identifying variables, documenting value labels, and converting Stata data to Parquet.
2. **Exploratory analysis:** examining credit access, financing obstacles, and firm characteristics.
3. **Exploratory modeling:** preparing a model dataset, training a logistic regression classifier, and examining model features and coefficients.
4. **Survey-weight analysis:** comparing weighted estimates, evaluating sensitivity to alternative weight specifications, and auditing analytical samples.
5. **Survey-design diagnostics:** inspecting stratification metadata and validating the consistency of survey-design variables.

The raw dataset and processed Parquet files are excluded from version control. The exact contents of `data/processed/` may vary depending on which scripts have been executed.

## Reproducibility and Setup

### Requirements

- Python
- Polars
- Pyreadstat
- scikit-learn
- NumPy

The project dependencies are listed in `requirements.txt`.

### 1. Clone the Repository

Run the following commands from the directory where you want to store the project:

```powershell
git clone https://github.com/hapitana/msme_access_to_finance_indonesia.git
cd msme_access_to_finance_indonesia
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly or follow the appropriate environment-activation instructions for your system.

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the Data

Obtain the Indonesia 2023 WBES dataset from the [official survey catalog](https://microdata.worldbank.org/catalog/6449/) and place the Stata file at:

```text
data/raw/Indonesia-2023-full-data.dta
```

Run the following script from the project root to convert the Stata dataset into Parquet format:

```powershell
python src/06_save_to_parquet.py
```

The script saves the processed dataset to:

```text
data/processed/indonesia_2023.parquet
```

It then reloads the saved Parquet file and verifies that its row and column counts match those of the original dataset. This verifies the dataset dimensions, but does not by itself establish that every value and data type is identical.

The separate `05_load_stata_to_polars.py` script is intended for inspecting the dataset after loading it into Polars. It does not save the Parquet output.

### 5. Run the Analysis Scripts

Run scripts from the project root. For example:

```powershell
python src/10_inspect_survey_weights.py
python src/11_weighted_credit_vs_obstacle.py
python src/21_weight_sensitivity_check.py
python src/22_create_weight_sensitivity_report.py
python src/23_audit_sensitivity_sample.py
python src/31_weighted_credit_obstacle_analysis.py
python src/32_credit_obstacle_weight_sensitivity.py
```

The scripts produce descriptive summaries and, where configured, CSV reports in `data/processed/`.

Run the relevant scripts again when the underlying data, analysis logic, or methodological choices change. Verify that the values reported in this README match the latest generated outputs before publishing updated results.

## Reproducibility and Data Management

- Raw survey data are excluded from version control.
- The processed Parquet dataset is excluded from version control.
- Analysis scripts document the calculations used to produce the reported estimates.
- Sensitivity analyses use a common eligible sample to facilitate comparisons across weights.
- Valid-response denominators and sample restrictions should be stated when reporting percentages.
- Weighted and unweighted results should be clearly distinguished.
- Unresolved survey-design codes should not be interpreted without supporting documentation.
- Reported percentages are descriptive estimates and should be interpreted alongside the limitations described above.

## Future Improvements

Potential next steps include:

1. Confirm the intended target population and primary weight against the official WBES documentation.
2. Validate the survey's strata and other design variables for statistical inference.
3. Calculate survey-design-adjusted standard errors and confidence intervals.
4. Assess whether the complete-case sample differs systematically from excluded observations.
5. Extend the analysis to firm characteristics, financing applications, and reasons for not applying for credit, where the data and research question support such comparisons.
6. Add automated checks to ensure reported results remain consistent with regenerated outputs.
7. Improve data-validation checks for missing, non-finite, or otherwise invalid survey-weight values.

## Conclusion

The analysis finds that the estimated prevalence of major or very severe access-to-finance obstacles varies across survey-weight specifications. In the credit-status comparison, firms with an existing loan or line of credit consistently show a higher weighted prevalence of major or very severe obstacles than firms without credit across the three tested weights.

The direction of this descriptive association is stable across the tested specifications, while its estimated magnitude varies modestly. These findings motivate further investigation into financing constraints among firms with existing credit, while emphasizing the need for validated survey-design inference before making claims about statistical significance.

## License and Data Use

The survey data remain subject to the access conditions and terms specified by the World Bank Enterprise Surveys. Users should consult the [official survey catalog](https://microdata.worldbank.org/catalog/6449/) for applicable documentation and data-use requirements.

This repository presents an independent analytical project and should not be interpreted as an official World Bank publication or endorsement.