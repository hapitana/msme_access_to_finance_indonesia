# Data Documentation and Provenance

## 1. Source dataset

**Dataset:** World Bank Enterprise Survey 2023 — Indonesia  
**Study reference:** `IDN_2023_WBES_v01_M`  
**Catalog:** 6449  
**Source file used locally:** `Indonesia-2023-full-data.dta`  
**Catalog dimensions:** 2,955 cases and 431 variables

Official records:

- [World Bank Microdata Library — Indonesia 2023 WBES](https://microdata.worldbank.org/catalog/6449)
- [World Bank Data Catalog record](https://datacatalog.worldbank.org/search/dataset/0066646/indonesia-world-bank-enterprise-survey-2023)
- [World Bank Enterprise Surveys Manual and Guide](https://www.worldbank.org/content/dam/enterprisesurveys/documents/methodology/Enterprise%20Surveys_Manual%20and%20Guide.pdf)

The source catalog provides the data, questionnaires, metadata, and study documentation. Users should consult the current official record rather than relying solely on variable descriptions copied into this repository.

## 2. Unit, coverage, and representativeness

The WBES is an establishment-level survey of the eligible private-sector business population specified by the survey's sampling frame and methodology. Its results must not be described as covering every Indonesian MSME. In particular, informal firms and establishments outside the survey's industry, size, or frame criteria may not be represented.

The exact population interpretation must follow the Indonesia 2023 implementation report and sampling documentation.

## 3. Data access and storage

The original Stata dataset is not included in this repository. Obtain it directly from the official World Bank Microdata Library and comply with the access terms and citation instructions.

Expected local path used by the project:

```text
data/raw/Indonesia-2023-full-data.dta
```

The project has also used a processed Parquet working file. Processed microdata should remain local and should not be committed to the public repository unless the source terms and disclosure review explicitly permit it.

## 4. Variables used in project scripts

The following list records variable names used in project scripts and earlier documentation. It is a project-level working map, not a substitute for the official data dictionary.

| Variable | Working analytical role | Verification required |
|---|---|---|
| `k30` | Perceived access-to-finance obstacle | Confirm question wording, valid categories, special codes, and module applicability. |
| `k82` | Existing loan/line-of-credit status in earlier specifications | Confirm exact variable label and response codes. |
| `k162` | Credit-application status in earlier specifications | Confirm universe, questionnaire module, and codes. |
| `k20a1` | Outcome of latest credit application in earlier specifications | Confirm skip pattern and special codes. |
| `k17` | Main reason for not applying in earlier specifications | Distinguish discouraged firms from firms with no credit demand. |
| `wmedian` | Median-eligibility weight name used locally | Confirm exact metadata and target population. |
| `wweak` | Weak-eligibility weight name used locally | Confirm exact metadata and target population. |
| `wstrict` | Strict-eligibility weight name used locally | Reconcile with catalog's `wstrong` terminology; do not assume equivalence. |

### Weight-name discrepancy

The local project has used `wstrict`, `wmedian`, and `wweak`. The official Indonesia 2023 catalog description refers to `wweak`, `wmedian`, and `wstrong` for weak, median, and strong eligibility assumptions. This naming difference must be resolved by inspecting the actual data file's variable labels and the official implementation report. Do not relabel or reinterpret the local variable without evidence.

## 5. Special values and missingness

WBES data may contain:

- valid substantive response codes;
- “don't know” or refusal codes;
- “does not apply” codes; and
- empty values caused by questionnaire skip patterns or questions not administered to a respondent.

The meaning of a special value depends on the variable. Preserve the original codes in the source data and create explicit, documented analytical recodes. Do not globally convert all negative codes to zero or to one generic missing value without checking variable-level metadata.

## 6. Survey design metadata

The official WBES documentation describes stratified sampling and provides survey weights. For Indonesia 2023, the catalog describes `strata` as the stratum identifier and discusses size, region, and industry sampling information. The catalog also notes the panel variable and the combination of sampling variables used to define strata.

The local project has performed consistency checks on stratification fields and singleton-stratum structure. A consistency check is not proof that the full variance-estimation specification is correct. The primary sampling unit, panel treatment, singleton handling, and degrees of freedom must be verified against the official Indonesia 2023 implementation documentation before design-based inference is reported.

## 7. Derived data and outputs

Derived files may include:

- analytic Parquet datasets;
- weighted and unweighted aggregate CSV tables;
- regression coefficient summaries;
- regional and industry summaries;
- predicted values and pairwise comparisons; and
- diagnostic outputs.

Only reviewed, non-disclosive aggregate outputs should be considered for public release. Row-level predictions, firm identifiers, microdata extracts, and diagnostic dumps should remain private unless there is a clear legal and ethical basis for publication.

A `.gitignore` entry does not remove a file that is already tracked by Git. Existing tracked outputs must be reviewed separately.

## 8. Citation

Use the official catalog citation and insert the actual date on which the dataset was downloaded:

> The World Bank Group. Indonesia - World Bank Enterprise Survey (WBES) 2023, Ref. IDN_2023_WBES_v01_M. Dataset downloaded from the World Bank Enterprise Surveys / Microdata Library on [insert download date].

Check the catalog for any updated citation wording and access requirements before manuscript submission.

## 9. Provenance record recommended for each analysis release

Record the following in a release note or analysis manifest:

- catalog reference and dataset version;
- date of download;
- source filename and SHA-256 checksum (stored locally if the data cannot be redistributed);
- software and package versions;
- script names and commit hash;
- variable crosswalk and recoding decisions;
- sample inclusion and exclusion rules;
- selected weight and its official definition; and
- output filenames and checksums for released aggregate tables.
