# Reproducibility Guide

## 1. Scope

This repository is an evolving research workspace, not yet a single-click replication package. The scripts are numbered and named by development stage; some are diagnostics or depend on local intermediate outputs. Read each script's input/output section before execution.

## 2. Clone the repository

```powershell
git clone https://github.com/hapitana/msme_access_to_finance_indonesia.git
cd msme_access_to_finance_indonesia
```

## 3. Python environment

The repository currently includes a `requirements.txt` file. Create an isolated environment and install the listed baseline dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, call the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The currently listed baseline packages are Polars, Pyreadstat, scikit-learn, and NumPy. Later Python scripts may require additional packages. Inspect imports in the specific script and record any additional package and version before execution. Do not assume `requirements.txt` already captures every dependency of scripts added later.

Selected R scripts use R and packages including `arrow` and `survey`; install and record the package versions required by the relevant R script. Python and R environments are separate.

## 4. Obtain and place the data

Download the Indonesia 2023 WBES dataset from the official catalog:

https://microdata.worldbank.org/catalog/6449

Place the Stata file at:

```text
data/raw/Indonesia-2023-full-data.dta
```

Do not commit the source file or processed microdata to GitHub. The dataset is obtained separately by each researcher under the source's terms.

## 5. Prepare the local working data

The initial project workflow has used a Stata-to-Parquet conversion script. The original README identifies:

```powershell
python src/06_save_to_parquet.py
```

as the initial conversion step, producing a local Parquet file. Confirm that this script and its expected output path still match the current checkout before running it. Later scripts may use a differently named candidate analytic dataset or a revised path.

Never overwrite the original source dataset. Preserve an untouched copy and record the source file checksum locally.

## 6. Run analysis scripts

There is no guarantee that every script can be executed in numeric order. Some scripts:

- inspect or validate data without creating downstream inputs;
- expect intermediate Parquet/CSV files from earlier stages;
- write diagnostic outputs to local folders;
- use hard-coded paths that may need adaptation; or
- depend on R packages and a separate R environment.

Before running a script:

1. inspect its imports;
2. identify its input paths;
3. identify required columns and sample filters;
4. inspect its output paths;
5. check whether it modifies or overwrites an existing file; and
6. record the command, environment, and commit hash.

Run from the repository root unless the script explicitly documents another working directory.

## 7. Suggested run log

For a research-grade run, maintain a private run log with:

| Field | Record |
|---|---|
| Git commit | `git rev-parse HEAD` |
| Date/time | Local run timestamp and timezone |
| Python | `python --version` |
| R | `R --version`, if applicable |
| Packages | Installed versions |
| Source data | Filename, catalog version, SHA-256 |
| Script | Relative path and command |
| Inputs | File names and checksums |
| Sample | N, event count, exclusions |
| Weight | Exact variable name and official definition |
| Outputs | File names and checksums |
| Notes | Warnings, errors, deviations |

Example commands:

```powershell
git rev-parse HEAD
python --version
Get-FileHash data/raw/Indonesia-2023-full-data.dta -Algorithm SHA256
```

Do not publish the raw-data checksum if doing so conflicts with the source's terms; it is normally safe to retain it in a private run manifest.

## 8. Reproducibility limitations

A successful script execution does not establish that the research result is fully reproducible or statistically valid. Reproducibility requires the same source data version, environment, scripts, input files, and analytical decisions.

In particular, the project still requires explicit resolution of:

- the local `wstrict` versus official catalog `wstrong` naming discrepancy;
- the survey-design variance specification and degrees-of-freedom discrepancy;
- singleton-stratum handling;
- sparse-event and quasi-complete-separation limitations; and
- the dependencies and path assumptions of the full set of later scripts.

Do not present design-based p-values, confidence intervals, or population-level inference as final until these issues have been resolved and documented.

## 9. Public-release checklist

Before publishing a code or results update:

- [ ] No raw microdata or respondent/firm-level extracts are staged.
- [ ] No row-level predictions or identifiers are staged.
- [ ] No logs, temporary diagnostics, credentials, or local personal paths are staged.
- [ ] Every aggregate output has been reviewed for disclosure risk and source-term compliance.
- [ ] New dependencies are recorded.
- [ ] Script inputs, outputs, and execution order are documented.
- [ ] Weight labels and survey design decisions are traceable to official documentation.
- [ ] README claims match the latest regenerated outputs.
- [ ] Git status and staged file list have been reviewed before commit.
