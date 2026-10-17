# Project Findings — Indonesia 2023 WBES: Access to Finance

**Status:** Exploratory findings; survey-design validation is still in progress.

## 1. Research question

This analysis examines how formal credit access is associated with firms' perceived access-to-finance obstacles in the World Bank Enterprise Surveys (WBES) Indonesia 2023 data. It also documents firm-size, regional, industry, weighting, and model-evaluation checks relevant to a reproducible research portfolio.

The results are descriptive and associational. They do not show that credit access causes an increase or decrease in perceived obstacles.

## 2. Data source and universe of inference

The Indonesia 2023 WBES implementation report describes fieldwork conducted from December 2022 to September 2023 and a total achieved sample of 2,955 establishments: 2,684 fresh interviews and 271 panel interviews.

The report defines the universe of inference as formal/registered private-sector businesses with at least 1% private ownership and at least five employees, within eligible industries. Therefore, the findings should not be generalized to every Indonesian MSME, especially informal enterprises and firms with fewer than five employees.

The survey uses stratified random sampling by industry, establishment size, and region. The implementation report emphasizes the need to apply sampling weights when estimating population indicators.

## 3. Main descriptive results

### 3.1 Existing formal credit (`k82`)

Among 2,559 valid responses:
- Some form of credit (codes 1–3): 518 establishments, or **20.24%**.
- No credit (code 4): 2,041 establishments, or **79.76%**.

The valid-response denominator excludes 396 “don't know” responses (`-9`). The result is not the percentage of all 2,955 interviews without qualification.

### 3.2 Credit applications (`k162`)

Among 2,564 valid responses:
- Applied for some form of credit (codes 1–3): 418, or **16.30%**.
- Did not apply (code 4): 2,146, or **83.70%**.

The valid-response denominator excludes 391 “don't know” responses (`-9`). Non-application should not be equated automatically with being denied credit; reasons for not applying are examined separately.

### 3.3 Outcome of latest application (`k20a1`)

Among 389 valid recorded outcomes:
- Fully approved: 322 (**82.78%**).
- Partly approved: 52 (**13.37%**).
- Rejected: 10 (**2.57%**).
- Withdrawn: 5 (**1.29%**).

These percentages describe valid recorded latest-application outcomes only. The variable follows questionnaire routing, so this distribution must not be interpreted as an approval rate among all establishments or all firms needing credit.

### 3.4 Main reason for not applying (`k17`)

Among 2,012 valid responses:
- No need / sufficient capital: 1,359 (**67.54%**).
- Complex procedures: 134 (**6.66%**).
- Unfavorable interest rates: 144 (**7.16%**).
- Collateral too high: 109 (**5.42%**).
- Loan size or maturity insufficient: 38 (**1.89%**).
- Expected rejection: 34 (**1.69%**).
- Other: 194 (**9.64%**).

The leading reported reason is no need or sufficient capital. This should not be described as evidence that all non-applicants have no financing constraints: the measure is self-reported, conditional on the question's applicability, and the other reasons remain important.

### 3.5 Perceived access-to-finance obstacle (`k30`)

Among 2,833 valid responses:
- No obstacle: 1,145 (**40.42%**).
- Minor obstacle: 658 (**23.23%**).
- Moderate obstacle: 614 (**21.67%**).
- Major obstacle: 269 (**9.50%**).
- Very severe obstacle: 147 (**5.19%**).

Combining major and very severe responses gives 416 of 2,833 valid responses, or **14.69%**. This combined binary outcome is used for the exploratory comparison and baseline classifier. The original five-level response should be retained for further ordinal analysis.

## 4. Credit status and perceived obstacle

The complete analytical sample used in the current comparison/model contains **2,491 establishments**. In this subset, 14.93% report a major or very severe access-to-finance obstacle.

### Unweighted comparison

| Credit status | Major/very severe obstacle rate |
|---|---:|
| Has some form of credit | 21.83% |
| No credit | 13.14% |
| Difference | +8.69 percentage points |

### `wmedian`-weighted comparison

| Credit status | Weighted major/very severe obstacle rate |
|---|---:|
| Has some form of credit | 24.86% |
| No credit | 12.33% |
| Difference | +12.53 percentage points |

In this sample, firms that report having credit are also more likely to rate access to finance as a major or very severe obstacle. This may seem counterintuitive, but the measures capture different concepts: a firm can have a loan and still experience financing constraints, or firms facing a strong need for finance may be more likely to obtain some credit while continuing to perceive obstacles. The cross-sectional comparison cannot distinguish these explanations or establish causal direction.

The weighted figures are point estimates. Design-adjusted standard errors, confidence intervals, and significance tests have not been established in this workflow.

## 5. Firm-size patterns

A preliminary weighted comparison in the complete model sample produced the following major/very severe obstacle rates:

| Firm-size group used in the current script | Sample count | Weighted rate |
|---|---:|---:|
| Small | 2,361 | 15.91% |
| Medium | 305 | 7.95% |
| Large | 167 | 7.23% |

The sample counts shown here are from the size-comparison output and may differ from the complete model-sample size because the exact subset depends on variable availability and coding. The estimates suggest lower reported major/severe-obstacle rates for medium and large firms than for small firms in this analysis. Before treating this as a population finding, confirm the size variable and its labels, the analysis denominator, the weight specification, and survey-design uncertainty.

### Size coding decision

The official implementation report documents `a6a` as:
- 1 = Small (5–19 employees)
- 2 = Medium (20–99 employees)
- 3 = Large (100 or more employees)

The data dictionary labels `a6b` as screener size: 0 = Micro, 1 = Small, 2 = Medium, 3 = Large. No `a6b = 0` observations appear in the observed dataset.

`stratificationsizecode` has labeled codes 1–3 in the inspected official dictionary, while codes 4–6 appear in the data without labels. The project has not inferred the meaning of codes 4–6, removed affected records, or treated those codes as panel status. There are 330 records where `a6a` and `stratificationsizecode` differ; this mismatch remains documented and unresolved.

## 6. Survey-weight and design validation

### 6.1 Weight choice

The Indonesia 2023 WBES implementation report says that Enterprise Surveys team indicators and analyses use weights based on the **median eligibility assumption**. Accordingly:
- `wmedian` is the primary weight considered for standard questions such as `k82` and `k30`.
- `wstrict` and `wweak` represent alternative eligibility assumptions and can be used for sensitivity checks.
- `wmedian_BR`, `wstrict_BR`, and `wweak_BR` apply to questions in the additional B-Ready / Green Economy and Taxation modules. They should not be substituted for standard-question weights without a specific methodological reason.

The implementation report's narrative refers to the standard weight family as `wstrong` in one database-structure passage, but the actual file inspected for this project contains `wstrict`, `wmedian`, and `wweak`, not `wstrong`. The analysis uses the actual variable names in the file and does not invent a `wstrong` variable.

### 6.2 Weight sensitivity

For the current 2,491-row model sample, the observed weighted rates of major/very severe obstacles were:

| Estimation approach | Rate |
|---|---:|
| Unweighted | 14.93% |
| `wstrict` | 12.38% |
| `wmedian` | 14.91% |
| `wweak` | 11.43% |

The sensitivity results vary materially under alternative eligibility assumptions. This is a reason to report the chosen weight and sensitivity transparently, not to select whichever estimate best fits a preferred narrative. These are not design-adjusted confidence intervals.

### 6.3 Strata consistency checks

The dataset contains 2,955 observations and 1,111 distinct `strata` values. The official report says the stratum is formed from industry, region, establishment size, and panel status. Validation scripts found:
- 956 distinct combinations of `a4a`, `a2`, and `a6a`.
- 1,111 distinct combinations after adding `stratificationpanelcode`.
- Each `strata` value maps to exactly one combination of `a4a`, `a2`, `a6a`, and `stratificationpanelcode`.
- Each such combination maps to exactly one `strata` value in the observed file.
- `a4a` matches `stratificationsectorcode` for all 2,955 observations.
- `a2` matches `stratificationregioncode` for all 2,955 observations.
- `a6a` matches `stratificationsizecode` for 2,625 observations and differs for 330.

These are useful internal consistency checks. They do not alone establish a correct variance-estimation procedure. The `strata` variable has many singleton strata (584 strata have one observed record in the file), so variance estimation requires particular care and should follow the official survey design guidance. Do not present standard errors, confidence intervals, or design-based tests until the exact design and appropriate variance method have been validated.

## 7. Baseline logistic-regression model

The baseline classifier predicts the binary outcome “major or very severe obstacle” using:
- Firm-size code
- Existing credit status
- Industry code
- Region code

The current workflow uses one-hot encoding, `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`, and a stratified 80/20 train-test split (`random_state=42`).

### Holdout metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.7054 |
| Precision for positive class | 0.2805 |
| Recall for positive class | 0.6133 |
| F1-score | 0.3849 |
| ROC-AUC | 0.7318 |

Confusion matrix (actual rows; predicted columns, with positive class = major/very severe obstacle):

```text
[[306, 118],
 [ 29,  46]]
```

The model identifies a portion of firms reporting major/severe obstacles, but precision is modest, so many predicted positives are false positives at the selected threshold. Accuracy alone is not an adequate measure, especially with an imbalanced target. These metrics are not survey-weighted and come from one holdout split; they do not demonstrate external validity or suitability for operational decisions.

The coefficient inspection also produced large odds ratios for some regional categories. Such estimates can be unstable when categories have small sample counts or when category interpretation/reference levels are not documented. Coefficients should not be presented as causal effects or policy rankings.

## 8. Key limitations and interpretation

1. **Universe of inference:** WBES Indonesia 2023 does not cover all MSMEs, informal enterprises, or firms with fewer than five employees.
2. **Nonresponse and questionnaire routing:** Each variable has its own valid-response denominator; some questions apply only to a subset.
3. **Self-reported obstacle:** `k30` measures the firm's perceived obstacle, not an objective loan-denial event or realized credit default.
4. **Cross-sectional association:** Existing credit and perceived obstacles may be jointly related to demand for finance, firm characteristics, and unobserved constraints.
5. **Weights and survey design:** Weighted point estimates are reported, but design-based uncertainty has not yet been validated.
6. **Unresolved size codes:** The meanings of `stratificationsizecode` 4–6 are not documented in the inspected official dictionary.
7. **Model limitations:** One holdout split, no external validation, no survey-weighted model evaluation, and no claim of causal inference.
8. **Potential measurement and selection effects:** Credit status, application decisions, and reported obstacles capture different stages and perceptions of firms' financing experiences.

## 9. Next steps

1. Re-run scripts and verify all reported denominators from the current data.
2. Add an explicit variable-definition and missing-code table to the data dictionary.
3. Confirm the exact survey design and appropriate variance estimator from the official WBES sampling note or with a survey-statistics package that supports the required design.
4. Keep the standard weights and `_BR` weights separate.
5. Reconcile size coding with authoritative documentation before producing size-specific population claims.
6. Add confidence intervals only after the survey design is correctly specified.
7. Consider ordinal models for the original five-level `k30` outcome, with appropriate survey-design treatment.
8. Improve predictive evaluation with repeated or cross-validation splits, calibration checks, threshold analysis, and clearly separated survey inference from prediction.
9. Preserve an auditable record of all recoding and exclusions.

## 10. Official documentation

- World Bank Microdata Library, Indonesia 2023 Enterprise Surveys: https://microdata.worldbank.org/catalog/6449/
- Indonesia 2023 WBES Implementation Report and accompanying tables, available from the official catalog.

This report is an exploratory portfolio artifact, not an official World Bank analysis and not a lending recommendation.
