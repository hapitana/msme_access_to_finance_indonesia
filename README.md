# Indonesia 2023 WBES: Firm-Level Financing Constraints and Access to Finance

**Repository:** [hapitana/msme_access_to_finance_indonesia](https://github.com/hapitana/msme_access_to_finance_indonesia)  
**Research status:** Ongoing, exploratory and methodological validation in progress  
**Primary data source:** World Bank Enterprise Survey 2023 — Indonesia  
**Main tools:** Python, Polars, NumPy, scikit-learn, and R for selected survey-design checks

## Project purpose

This repository contains code and documentation for an independent analysis of firm-level access to finance and perceived financing obstacles in Indonesia. The project uses the World Bank Enterprise Survey (WBES) 2023 for Indonesia and develops a transparent workflow for data inspection, analytical-sample construction, descriptive estimation, survey-weight sensitivity analysis, exploratory regression, and regional/industry robustness checks.

The project is concerned with firms' reported experiences and perceptions. It does **not** estimate the causal effect of receiving credit, does not constitute a credit-scoring product, and should not be interpreted as an assessment of the creditworthiness of individual firms.

## Research questions

The current analytical programme includes the following questions:

1. How common are major or very severe perceived access-to-finance obstacles among firms represented by the WBES sample?
2. How sensitive are descriptive estimates to alternative survey-weight assumptions?
3. How do perceived financing obstacles vary by existing credit status and selected firm characteristics?
4. How much regional and industry heterogeneity is visible in the observed outcomes?
5. How robust are regional comparisons to sparse-event categories, alternative samples, and model specifications?
6. Which findings remain suitable for descriptive reporting, and which can support design-based statistical inference after the survey design has been validated?

These questions may be refined as the codebook, survey design, and analytical decisions are further validated.

## Data source and population

The source is the **World Bank Enterprise Survey 2023 — Indonesia**, study reference `IDN_2023_WBES_v01_M`, catalog number 6449:

- [World Bank Microdata Library — study page](https://microdata.worldbank.org/catalog/6449)
- [World Bank Data Catalog — dataset record](https://datacatalog.worldbank.org/search/dataset/0066646/indonesia-world-bank-enterprise-survey-2023)
- [World Bank Enterprise Surveys methodology manual](https://www.worldbank.org/content/dam/enterprisesurveys/documents/methodology/Enterprise%20Surveys_Manual%20and%20Guide.pdf)

The source file used in the local workflow is named `Indonesia-2023-full-data.dta`. The catalog describes a file with 2,955 cases and 431 variables. The survey represents the eligible formal/private-sector establishment population defined by the WBES sampling frame; it is **not** a census of all Indonesian MSMEs. Informal businesses, firms outside the eligible industries, and establishments outside the survey's size/frame criteria should not be assumed to be represented.

The data must be obtained from the official World Bank source. It is not redistributed in this repository. Users are responsible for complying with the source's access conditions, citation requirements, and terms of use.

## Analytical scope and current workflow

The `src/` directory contains scripts developed iteratively. Script names and numbering reflect the research and diagnostic history; they are not necessarily a single, fully automated pipeline. In broad terms, the scripts cover:

- source-data inspection, variable/value-label exploration, and candidate-variable selection;
- analytical dataset preparation and sample/missingness diagnostics;
- weighted descriptive estimates and weight-sensitivity comparisons;
- exploratory logistic-regression specifications and coefficient/separation diagnostics;
- survey-design, stratification, singleton-stratum, and variance-implementation audits;
- geographic outcome concordance, joint Wald-test checks, regional predictions, and pairwise effect-size comparisons;
- industry-section and sparse-region robustness checks; and
- preparation and verification of selected cross-tabulations and design-based estimates.

Read the relevant script before running it. Some scripts are diagnostic, depend on outputs from earlier scripts, or use local input/output paths. The script sequence should not be interpreted as a promise that every file can be run independently or in numerical order.

## Variables and coding

The following variable names are used in parts of the current project. Confirm their labels, universe, skip patterns, and special codes against the exact Indonesia 2023 questionnaire and data dictionary before reusing them.

| Variable | Project role | Important caution |
|---|---|---|
| `k30` | Perceived access-to-finance obstacle | The binary `major_obstacle` outcome groups major and very severe responses only where the underlying response is valid. |
| `k82` | Existing loan/line-of-credit status in earlier project specifications | Confirm the exact label and valid codes against the current data dictionary before interpreting. |
| `k162` | Credit-application indicator in earlier project specifications | Confirm question universe, module, and response codes. |
| `k20a1` | Outcome of the most recent credit application in earlier project specifications | Respect skip patterns and special response codes. |
| `k17` | Main reason for not applying for credit in earlier project specifications | Distinguish no credit demand from discouraged or constrained non-applicants. |
| `wmedian`, `wweak`, `wstrict` | Weight names used in local analysis scripts | See the weight-metadata reconciliation note below. |

Negative and special response codes are variable-specific. They must not automatically be recoded to zero or treated as interchangeable missing values. Empty values may also reflect questionnaire skip patterns rather than item non-response.

### Important weight-metadata reconciliation

The local project has used the names `wstrict`, `wmedian`, and `wweak`. The current World Bank Indonesia 2023 catalog description refers to `wweak`, `wmedian`, and `wstrong` for weak, median, and strong eligibility assumptions. Do **not** assume that `wstrict` and `wstrong` are equivalent. Before presenting weight-sensitivity results as final, reconcile the variable names and definitions against the exact downloaded file, its value labels, the Indonesia 2023 implementation report, and the current catalog metadata.

The WBES methodology explains that eligibility assumptions affect weight construction and that the median assumption is used for published Enterprise Surveys indicators. This does not, by itself, prove that a particular weight is appropriate for every custom analytical sample or estimand.

## Outcome definition

One recurring outcome is whether a firm reports access to finance as a **major or very severe obstacle**. In specifications based on `k30`, codes 3 and 4 have been grouped as the positive category, while valid responses 0–2 form the comparison category. Special codes and inapplicable responses must be excluded or handled according to the documented variable-specific rules.

This outcome measures a respondent's reported perception. It is not a direct measure of the amount of unmet credit demand, loan rejection probability, financing gap, or objective creditworthiness.

## Methods and interpretation

The project distinguishes between:

1. **Unweighted sample summaries**, which describe the eligible analytic sample;
2. **Weighted descriptive estimates**, which apply the selected survey weight;
3. **Exploratory regression**, which examines conditional associations under stated model specifications; and
4. **Design-based inference**, which requires a defensible specification of the sampling design and variance estimator.

For a binary outcome \(y_i\) and selected weight \(w_i\), a weighted descriptive proportion is:

\[
\widehat{P}_w = \frac{\sum_{i \in S} w_i y_i}{\sum_{i \in S} w_i}.
\]

This ratio alone does not provide a design-correct standard error or confidence interval. A valid inferential analysis must account for the survey's sampling design, including the correct strata, primary sampling units where applicable, weights, finite population corrections if required, and the treatment of singleton strata.

Differences between firms with and without existing credit are descriptive unless a credible causal identification strategy is separately developed. Reverse causality, selection into borrowing, firm heterogeneity, financing demand, and credit supply conditions may all contribute to observed differences.

## Current methodological cautions

The project has progressed beyond the initial descriptive analysis and includes extensive diagnostics. The following issues must remain visible in any interpretation:

- **Weight labels:** the local `wstrict` naming and the catalog's `wstrong` naming require explicit reconciliation.
- **Survey-design inference:** internal audits have identified singleton-stratum and design-degrees-of-freedom issues. A discrepancy between calculated and model-reported design degrees of freedom remains to be resolved against the official design documentation and the variance-estimation implementation.
- **Sparse outcomes:** at least one severe-obstacle model specification exhibited quasi-complete separation. Its inferential use was flagged as unsuitable; results from such specifications should remain descriptive or exploratory.
- **Regional comparisons:** computational validation of a set of joint Wald calculations is not, on its own, proof that the underlying survey-design variance estimator is valid.
- **Sample selection:** complete-case and common-sample restrictions can change the composition of the analytic sample. Report denominators and exclusions alongside estimates.
- **External validity:** do not generalize findings to every Indonesian MSME or informal enterprise.
- **Causality:** regression coefficients and group differences are associations unless a separate identification design justifies causal language.

Do not report statistical significance, confidence intervals, or population-level inferential conclusions from a specification until the design and variance estimator have been validated.

## Repository structure

```text
msme_access_to_finance_indonesia/
├── data/
│   ├── raw/                 # Local source data; do not commit
│   └── processed/           # Local/intermediate data; inspect before publishing
├── docs/
│   ├── data_documentation.md
│   ├── methodology.md
│   └── reproducibility.md
├── src/                     # Research, analysis, and diagnostic scripts
├── reports/                 # Project reports, if maintained
├── README.md
├── requirements.txt
└── .gitignore
```

The exact set of directories and outputs may vary. Some aggregate CSV outputs may already be tracked in Git. A `.gitignore` rule does not remove files that were committed previously. Review existing tracked outputs individually before adding new data products.

## Reproducibility

Start with [`docs/reproducibility.md`](docs/reproducibility.md). In brief:

1. Clone the repository.
2. Create an isolated Python environment.
3. Install the dependencies listed in `requirements.txt`.
4. Obtain the source data independently from the official WBES catalog.
5. Place the source file in the documented local data directory.
6. Inspect each script's inputs, outputs, package requirements, and path assumptions.
7. Run the relevant scripts in the documented dependency order.
8. Record software versions, sample definitions, weight choices, and output checksums for any research release.

The repository is an evolving research workspace. A successful run of an individual script does not establish end-to-end reproducibility of every result.

## Data ethics and responsible publication

- Do not commit raw microdata, respondent-level records, firm identifiers, row-level predictions, or confidential/restricted files.
- Do not publish logs or diagnostic dumps that contain record-level data, local file paths, credentials, or other sensitive information.
- Publish only aggregate outputs that have been reviewed for disclosure risk and are permitted by the source's terms.
- Preserve source attribution and the World Bank's required dataset citation.
- Do not imply World Bank authorship, endorsement, or validation of this independent analysis.
- Do not use the results to make decisions about identifiable firms or individuals.
- Retain enough methodological detail for scrutiny without redistributing source data.

## Dataset citation

The World Bank Group. *Indonesia - World Bank Enterprise Survey (WBES) 2023*, Ref. IDN_2023_WBES_v01_M. World Bank Microdata Library, catalog 6449. Cite the actual download date in any manuscript or report and follow the citation text and conditions displayed on the official catalog page.

## Research status and disclaimer

This repository documents an ongoing independent research project. The scripts, data preparation, variable crosswalks, survey-design implementation, and analytical conclusions remain subject to review and revision. Results should not be described as peer-reviewed, causal, or fully design-validated unless and until the corresponding evidence is available.

The project is not an official World Bank product and does not represent the views of the World Bank Group.
