# Indonesia 2023 WBES — MSME Access to Finance

## Project overview

This project uses the World Bank Enterprise Surveys (WBES) Indonesia 2023 dataset to examine firms' access to formal credit and their reported access-to-finance obstacles. The workflow is designed as a reproducible portfolio project demonstrating data ingestion, data-quality checks, exploratory analysis, survey-weighted descriptive statistics, and a baseline machine-learning model using **Python and Polars**.

**Research question:** How are firms' reported access-to-finance obstacles associated with existing formal credit access, firm size, industry, and region in the Indonesia 2023 WBES sample?

The analysis is descriptive and exploratory. Associations should not be interpreted as causal effects.

## Data source and scope

- **Source:** World Bank Enterprise Surveys, Indonesia 2023.
- **Fieldwork:** December 2022–September 2023.
- **Sample:** 2,955 interviewed establishments; 2,684 fresh interviews and 271 panel interviews.
- **Raw format:** Stata `.dta`.
- **Working format:** Parquet.
- **Core tools:** Python, Polars, pyreadstat, scikit-learn, Matplotlib (where used).

The WBES universe is not the same as all Indonesian MSMEs. The implementation report defines the universe as formal/registered private-sector businesses with at least 1% private ownership and at least five employees, within eligible industries. Consequently, informal businesses and firms with fewer than five employees are outside the intended universe of inference. The observed WBES sample should not be described as representative of every MSME in Indonesia.

### Obtaining the data

Download the Indonesia 2023 WBES data and its documentation from the official World Bank Microdata Library catalog:

- https://microdata.worldbank.org/catalog/6449/

Do not commit the raw survey file or the processed row-level Parquet file to a public repository unless the data license and redistribution terms explicitly permit it. This repository's `.gitignore` should exclude `data/raw/` and `data/processed/*.parquet`.

## Project structure

```text
02_msme_access_to_finance/
├── data/
│   ├── raw/                  # Local-only source data; not committed
│   └── processed/            # Local-only Parquet and row-level outputs
├── notebooks/                # Optional notebooks and supporting notes
├── reports/
│   └── 23_project_findings.md
├── src/
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
│   ├── 24_inspect_survey_design.py
│   ├── 25_validate_strata_structure.py
│   ├── 26_validate_strata_with_panel.py
│   ├── 27_inspect_size_code_mapping.py
│   └── 28_inspect_original_size_labels.py
└── README.md
```

The exact set of scripts may differ if the repository has additional earlier scripts or local working files. The numbering follows the current analysis sequence; script 23's findings are a Markdown report under `reports/`, not a Python script.

## Setup

Use Python 3.14 or a compatible Python version with the dependencies listed in `requirements.txt`. The workflow was developed with Python 3.14 and Polars 2.0.0; compatibility may vary across environments.

### Installation

From the project directory, run:

```powershell
python -m pip install -r requirements.txt
```

If using Windows PowerShell, navigate to the project directory first:

```powershell
cd D:\portofolio\02_msme_access_to_finance
```

Obtain the raw survey data separately according to the instructions in the **Obtaining the data** section. Raw survey data are not included in this repository.

The project intentionally uses **Polars rather than pandas** for data manipulation.

## Main variables

| Variable | Interpretation used in this project | Notes |
|---|---|---|
| `k82` | Whether the establishment has a line of credit or loan from a financial institution | Codes 1–3 indicate some form of credit; code 4 indicates no; `-9` is don't know |
| `k162` | Whether the establishment applied for credit | Codes 1–3 indicate application; code 4 indicates no; `-9` is don't know |
| `k20a1` | Outcome of the latest credit application | Includes full approval, partial approval, rejection, and withdrawal; skip patterns matter |
| `k17` | Main reason for not applying for credit | Includes no need/sufficient capital, procedures, interest rates, collateral, expected rejection, and other reasons |
| `k30` | Establishment's perceived obstacle to access to finance | 0 = no obstacle; 1 = minor; 2 = moderate; 3 = major; 4 = very severe; negative codes are non-substantive response codes |
| `a6a` | Sampling-size category | 1 = small, 2 = medium, 3 = large according to the implementation report |
| `a6b` | Screener size category | 0 = micro, 1 = small, 2 = medium, 3 = large; code 0 does not appear in the observed data |
| `a4a` | Sampling-frame industry stratum | 19 categories |
| `a2` | Sampling region | 22 categories |
| `strata` | Survey stratum identifier | Validated against the documented stratification components and panel status |
| `stratificationpanelcode` | Panel status | 0 = fresh; 1 = panel |
| `wstrict`, `wmedian`, `wweak` | Standard survey weights under strict, median, and weak eligibility assumptions | The implementation report says WBES team analyses use median-assumption weights |
| `wstrict_BR`, `wmedian_BR`, `wweak_BR` | Weights for the additional B-Ready / Green Economy and Taxation module questions | Do not substitute these for standard-question weights without a reason |

Use the official questionnaire and data dictionary to confirm variable meaning and skip patterns before adding further variables. In particular, `a4a` is the sampling-frame industry classification; the report notes that it can differ from realized industry information such as `d1a2_v4`.

## Methodology decisions

### Missing and special codes

The analysis treats `-9` as a non-substantive “don't know” response for the variables listed above. Other negative codes (for example, “not applicable” or “still in process”) must be interpreted variable by variable. Do not automatically convert every negative code to zero or treat every negative code as the same type of missingness.

For each estimate, state the denominator used. A percentage among valid answers is not necessarily a percentage of all 2,955 interviewed firms.

### Survey weights

The Indonesia 2023 WBES implementation report states that the Enterprise Surveys team uses the **median-assumption sampling weights** for its standard indicators. Accordingly, `wmedian` is the primary weight considered for standard questions such as `k82` and `k30`. The `_BR` weight family is intended for the additional module questions, not the standard questions.

The strict, median, and weak weights represent alternative eligibility assumptions. A sensitivity analysis across them is useful, but it does not establish which estimate is correct for every research question. Weighted percentages are descriptive estimates; they are not, by themselves, survey-design-adjusted standard errors or confidence intervals.

The data contain 1,111 distinct `strata` values, including many strata represented by one sampled observation. Diagnostics verified that the combination of `a4a`, `a2`, `a6a`, and `stratificationpanelcode` maps one-to-one to `strata` in this file. This is a consistency check, not proof that all variance-estimation details are correctly specified. Do not report design-based standard errors or confidence intervals until the design, strata, panel treatment, and variance method have been validated against the official sampling documentation.

### Firm-size coding

The implementation report documents `a6a` as small, medium, and large. `a6b` has screener labels for micro, small, medium, and large, but no `a6b = 0` observations occur in this dataset. The variable `stratificationsizecode` contains codes 4–6 that are not labeled in the inspected official dictionary. Their meanings are unresolved. Do not guess their meaning, silently recode them, or delete the 330 observations where `a6a` and `stratificationsizecode` differ.

For comparisons using documented categories, use the variable and coding whose meaning is appropriate to the research question, explain the choice, and retain an audit of any excluded observations.

## Preliminary results

These are results from the current exploratory workflow and should be reproduced from the scripts before external publication.

- Of valid responses to `k82` (n = 2,559), 20.24% reported having some form of formal credit (codes 1–3); 79.76% reported no credit (code 4).
- Of valid responses to `k162` (n = 2,564), 16.30% reported applying for credit; 83.70% reported no application.
- Among valid `k20a1` application outcomes (n = 389), 82.78% were fully approved, 13.37% partly approved, 2.57% rejected, and 1.29% withdrawn. This is conditional on the observed latest-application outcome responses, not all establishments.
- Of valid `k17` responses (n = 2,012), 67.54% cited no need/sufficient capital as the main reason for not applying.
- Of valid `k30` responses (n = 2,833), 14.69% rated access to finance a major or very severe obstacle.
- In the complete model sample (n = 2,491), 14.93% reported a major/very severe obstacle. This is an analytical subset, not the full survey.
- In that model sample, the unweighted major/severe-obstacle rate was 21.83% among establishments with credit and 13.14% among those without credit, a difference of 8.69 percentage points. The corresponding `wmedian` weighted estimates were 24.86% and 12.33%, a difference of 12.53 percentage points. These are associations and do not establish that credit causes firms to perceive fewer or more obstacles.
- A preliminary logistic-regression holdout evaluation reported ROC-AUC 0.7318, recall 0.6133, precision 0.2805, F1 0.3849, and accuracy 0.7054. The test confusion matrix was `[[306, 118], [29, 46]]`. This model is an exploratory classifier, not a lending decision system. Metrics are not survey-weighted and reflect the selected features, target definition, split, and class-balancing setting.

See [`reports/23_project_findings.md`](reports/23_project_findings.md) for a more detailed findings and limitations note.

## Key limitations

1. **Universe limitation:** The WBES covers formal businesses with at least five employees in eligible industries, not all MSMEs or informal enterprises.
2. **Item nonresponse and skip patterns:** Valid-response denominators differ by question; some questions are only asked after specific prior answers.
3. **Survey design:** Sampling weights matter. A weighted point estimate is not a substitute for design-aware uncertainty estimation.
4. **Strata and size-code ambiguity:** Codes 4–6 in `stratificationsizecode` are unlabeled in the official dictionary inspected for this project. Their meaning remains unresolved.
5. **Associational analysis:** Cross-tabulations and logistic regression do not identify causal effects.
6. **Predictive model limits:** The model uses survey variables and a single stratified holdout split. It has not been externally validated, and its evaluation metrics are not survey-weighted.
7. **Potential selection and measurement issues:** Perceived obstacles and current credit status may be jointly related to firm characteristics, demand for credit, and other unobserved factors.

## Responsible use

This is an educational and research-portfolio project. It does not support direct approval or rejection of real credit applications. Any future credit-risk work should distinguish credit access from repayment/default outcomes and evaluate fairness, explainability, privacy, leakage, and the availability of features at the point a real decision would be made.

## Reproducibility checklist

- [ ] Keep raw data local and out of version control.
- [ ] Re-run the import and Parquet conversion scripts.
- [ ] Re-run the EDA and modeling scripts after changes.
- [ ] Record the exact valid-response denominator for each percentage.
- [ ] Keep weighted and unweighted results clearly separated.
- [ ] Do not interpret undocumented size codes 4–6.
- [ ] Verify any design-based standard errors or confidence intervals against official survey documentation.
- [ ] Update `reports/23_project_findings.md` when results or methodological decisions change.
