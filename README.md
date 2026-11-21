# Indonesia 2023 World Bank Enterprise Survey: Access to Finance

An independent research project examining firms' perceived access-to-finance obstacles in Indonesia using the 2023 World Bank Enterprise Survey (WBES). The repository contains Python and R scripts for data preparation, descriptive analysis, exploratory modelling, survey-design diagnostics, and robustness analysis.

**Repository:** https://github.com/hapitana/msme_access_to_finance_indonesia
**Data source:** World Bank Enterprise Surveys, Indonesia 2023
**Research status:** Ongoing; methodological validation is not complete.

## Research focus

The project investigates how perceived financing obstacles vary across firm characteristics, credit-related measures, industries, and geographic areas. The analytical workflow has developed through several stages rather than one final, pre-registered specification.

The codebase includes:

- data inspection, validation, and preparation;
- descriptive and survey-weighted analysis;
- comparisons across alternative weight specifications;
- exploratory logistic regression;
- missingness and analytical-sample diagnostics;
- sparse-event and separation diagnostics;
- geographic and industry heterogeneity analyses;
- survey-design and variance-estimation diagnostics; and
- sensitivity analyses involving sparse-category exclusions and regional pooling.

The scripts are research materials in development. Their numbering reflects the order in which analyses were developed, not necessarily a complete or automatic execution sequence.

## Data and scope

The project uses the Indonesia 2023 WBES establishment-level survey. The official source record is available through the [World Bank Microdata Library](https://microdata.worldbank.org/catalog/6449).

The source microdata are not distributed in this repository. Researchers must obtain the data from the official source and comply with its access conditions and citation requirements. Raw data, processed microdata, firm-level extracts, and row-level predictions should remain outside this public repository.

WBES coverage is defined by its sampling frame, eligibility criteria, industries, and establishment requirements. The results must not be described as representative of every Indonesian MSME, particularly informal businesses and establishments outside the survey frame.

## Methodological status and limitations

The project deliberately distinguishes descriptive analysis from statistical inference. The following issues remain important:

1. **Survey-weight reconciliation:** local scripts have used `wstrict`, `wmedian`, and `wweak`. The catalog describes `wstrong`, `wmedian`, and `wweak`. The equivalence of `wstrict` and `wstrong` has not been established and must not be assumed.
2. **Survey-design inference:** strata, sampling units, panel treatment, singleton-stratum handling, variance estimation, and degrees of freedom require validation against the Indonesia 2023 implementation documentation.
3. **Sparse outcomes:** some outcome/model combinations have sparse events, zero-event categories, or quasi-complete separation. Models flagged as inferentially unsuitable should be treated as descriptive or exploratory only.
4. **Sample selection:** full, common, and model-specific analytical samples can differ. Every reported estimate must state its denominator and exclusion rules.
5. **Interpretation:** observational associations are not causal effects. Do not claim statistical significance or report design-based confidence intervals until the survey design and variance procedure have been validated.

For the detailed analytical rules, see:

- [Methodology and statistical analysis notes](docs/methodology.md)
- [Data documentation and provenance](docs/data_documentation.md)
- [Reproducibility guide](docs/reproducibility.md)

## Repository structure

```text
.
├── docs/
│   ├── data_documentation.md
│   ├── methodology.md
│   ├── reproducibility.md
│   ├── wbes_indonesia_2023_implementation_report.pdf
│   └── wbes_indonesia_2023_implementation_report_tables.xlsx
├── reports/
│   └── Reviewed research reports and presentation outputs
├── src/
│   ├── Python analysis and diagnostic scripts
│   ├── R survey-analysis scripts
│   └── PowerShell inspection utilities
├── .gitignore
├── README.md
└── requirements.txt
```

The exact files in `docs/` and `src/` are visible in the repository. Not every script is intended to be run in numeric order; inspect each script's inputs, outputs, and dependencies before execution.

## Reproducibility

### 1. Clone the repository

```powershell
git clone https://github.com/hapitana/msme_access_to_finance_indonesia.git
cd msme_access_to_finance_indonesia
```

### 2. Create a Python environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The baseline `requirements.txt` lists Polars, Pyreadstat, scikit-learn, and NumPy. Individual scripts may require additional packages. Selected analyses also use R packages such as `arrow` and `survey`; these must be installed in a separate R environment.

### 3. Obtain the source data

Download the relevant Indonesia 2023 WBES dataset from the [official catalog](https://microdata.worldbank.org/catalog/6449). Place the local Stata file at:

```text
data/raw/Indonesia-2023-full-data.dta
```

Do not commit the raw dataset or processed microdata.

### 4. Run individual scripts

Run scripts from the repository root unless a script documents otherwise. Before execution, inspect its imports, input paths, required variables, sample restrictions, output paths, and overwrite behaviour. Some scripts depend on intermediate files generated by earlier workflow stages.

The repository is not currently a one-command replication package. Successful execution alone does not establish statistical validity or full reproducibility.

## Responsible reporting

For every result intended for publication, document:

- source dataset and version;
- exact outcome and predictor definitions;
- sample size, event count, and exclusions;
- exact weight variable and official interpretation;
- survey-design and variance-estimation specification;
- missing-data treatment;
- sparse-event and separation diagnostics;
- robustness specifications; and
- limitations on inference and generalizability.

Use cautious language such as “associated with” and “weighted descriptive estimate.” Avoid causal claims and unsupported claims of statistical significance.

## Data citation

Follow the citation instructions in the [official WBES catalog](https://microdata.worldbank.org/catalog/6449). Record the actual data download date and dataset version in any manuscript or replication materials.

This repository is an independent research project. It is not an official World Bank publication and does not imply World Bank endorsement.
