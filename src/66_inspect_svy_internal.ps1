$ErrorActionPreference = "Continue"

$project_dir = "D:\portofolio\02_msme_access_to_finance"
$python_exe = (Get-Command python -ErrorAction Stop).Source

$site_packages = (& $python_exe -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" | Select-Object -Last 1).Trim()

$svy_dir = Join-Path $site_packages "svy"
$svy_rs_dir = Join-Path $site_packages "svy_rs"

$base_file = Join-Path $svy_dir "regression\base.py"
$glm_file = Join-Path $svy_dir "regression\glm.py"
$design_file = Join-Path $svy_dir "core\design.py"
$sample_file = Join-Path $svy_dir "core\sample.py"
$variance_file = Join-Path $svy_dir "core\variance_method.py"
$rs_init_file = Join-Path $svy_rs_dir "__init__.py"

$script55 = Join-Path $project_dir "src\55_survey_weighted_logistic_regression.py"

$output_dir = Join-Path $project_dir "data\processed\survey_design_validation"
$output_file = Join-Path $output_dir "66_svy_internal_inspection.txt"

New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

Start-Transcript -Path $output_file -Force

function Show-FileInfo {
    param([string]$Path)

    Write-Output ""
    Write-Output "============================================================"
    Write-Output "FILE: $Path"
    Write-Output "============================================================"

    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path
        Write-Output "STATUS: FOUND"
        Write-Output "SIZE: $($item.Length) bytes"
        Write-Output "LAST MODIFIED: $($item.LastWriteTime)"
    }
    else {
        Write-Output "STATUS: NOT FOUND"
    }
}

function Show-Section {
    param(
        [string]$Path,
        [int]$Start,
        [int]$Count,
        [string]$Label
    )

    Write-Output ""
    Write-Output "------------------------------------------------------------"
    Write-Output $Label
    Write-Output "FILE: $Path"
    Write-Output "LINES: $Start to $($Start + $Count - 1)"
    Write-Output "------------------------------------------------------------"

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Output "FILE NOT FOUND"
        return
    }

    $content = @(Get-Content -LiteralPath $Path)

    if ($Start -lt 1 -or $Start -gt $content.Count) {
        Write-Output "REQUESTED LINE RANGE IS OUTSIDE FILE"
        return
    }

    $end = [Math]::Min($Start + $Count - 1, $content.Count)

    for ($i = $Start; $i -le $end; $i++) {
        Write-Output ("{0,6}: {1}" -f $i, $content[$i - 1])
    }
}

function Show-Matches {
    param(
        [string]$Path,
        [string[]]$Patterns,
        [int]$Before = 4,
        [int]$After = 8,
        [string]$Label = "PATTERN SEARCH"
    )

    Write-Output ""
    Write-Output "============================================================"
    Write-Output $Label
    Write-Output "FILE: $Path"
    Write-Output "============================================================"

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Output "FILE NOT FOUND"
        return
    }

    $matches = Select-String `
        -LiteralPath $Path `
        -Pattern $Patterns `
        -Context $Before, $After

    if ($null -eq $matches) {
        Write-Output "NO MATCHES FOUND"
        return
    }

    foreach ($match in $matches) {
        Write-Output ""
        Write-Output "MATCH AT LINE $($match.LineNumber)"
        Write-Output $match.Line

        foreach ($line in $match.Context.PreContext) {
            Write-Output $line
        }

        foreach ($line in $match.Context.PostContext) {
            Write-Output $line
        }
    }
}

Write-Output "SVY INTERNAL IMPLEMENTATION INSPECTION"
Write-Output "DATE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "PROJECT: $project_dir"
Write-Output "PYTHON: $python_exe"
Write-Output "SITE PACKAGES: $site_packages"

Write-Output ""
Write-Output "IMPORTANT:"
Write-Output "This script only reads source files."
Write-Output "It does not execute Script 55."
Write-Output "It does not modify Script 55."
Write-Output "It does not rerun any regression."

Write-Output ""
Write-Output "1. PACKAGE FILE CHECK"

$files_to_check = @(
    $base_file,
    $glm_file,
    $design_file,
    $sample_file,
    $variance_file,
    $rs_init_file,
    $script55
)

foreach ($file in $files_to_check) {
    Show-FileInfo -Path $file
}

Write-Output ""
Write-Output "2. PYTHON PACKAGE AND BACKEND INFORMATION"

$python_inspection = @'
import importlib
import importlib.util
import pathlib
import sys
import svy

print("Python version:", sys.version)
print("svy version:", getattr(svy, "__version__", "not exposed"))
print("svy package:", pathlib.Path(svy.__file__).resolve())

try:
    import svy_rs
    print("svy_rs package:", pathlib.Path(svy_rs.__file__).resolve())
except Exception as exc:
    print("svy_rs import error:", repr(exc))

try:
    spec = importlib.util.find_spec("svy_rs._internal")
    print("_internal spec:", spec)
    if spec is not None:
        print("_internal origin:", spec.origin)
        print("_internal loader:", spec.loader)
except Exception as exc:
    print("_internal lookup error:", repr(exc))

try:
    module = importlib.import_module("svy_rs._internal")
    print("_internal imported:", module)
    print("_internal file:", getattr(module, "__file__", None))
    print("_internal functions:", [
        name for name in dir(module)
        if not name.startswith("__")
    ])
except Exception as exc:
    print("_internal import error:", repr(exc))
'@

$python_inspection | & $python_exe -

Write-Output ""
Write-Output "3. BACKEND FILE DISCOVERY"

if (Test-Path -LiteralPath $svy_rs_dir) {
    Get-ChildItem -LiteralPath $svy_rs_dir -Recurse -File |
        Where-Object {
            $_.Extension -in @(
                ".pyd",
                ".dll",
                ".rs",
                ".py",
                ".toml",
                ".lock"
            )
        } |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize
}
else {
    Write-Output "svy_rs directory not found"
}

Write-Output ""
Write-Output "4. BASE.PY: KNOWN IMPLEMENTATION SECTIONS"

Show-Section -Path $base_file -Start 800 -Count 110 `
    -Label "BASE.PY SECTION 800-909"

Show-Section -Path $base_file -Start 900 -Count 100 `
    -Label "BASE.PY SECTION 900-999"

Show-Section -Path $base_file -Start 1625 -Count 40 `
    -Label "BASE.PY SECTION 1625-1664"

Show-Section -Path $base_file -Start 1660 -Count 75 `
    -Label "BASE.PY SECTION 1660-1734"

Write-Output ""
Write-Output "5. GLM.PY: GLM FITTING AND RESULTS"

Show-Section -Path $glm_file -Start 255 -Count 100 `
    -Label "GLM.PY SECTION 255-354"

Show-Matches -Path $glm_file `
    -Patterns @(
        "fit_glm",
        "cov_matrix",
        "wald",
        "df_resid",
        "df_den",
        "df_num",
        "GLMResults",
        "RegressionResults"
    ) `
    -Before 5 -After 10 `
    -Label "GLM.PY IMPLEMENTATION SEARCH"

Write-Output ""
Write-Output "6. BASE.PY: COVARIANCE AND DEGREES OF FREEDOM"

Show-Matches -Path $base_file `
    -Patterns @(
        "cov_matrix",
        "cov_mat",
        "replicate",
        "rep_wgts",
        "df_resid",
        "design_df",
        "df_den",
        "df_num",
        "class GLMResults",
        "class RegressionResults",
        "wald"
    ) `
    -Before 4 -After 8 `
    -Label "BASE.PY COVARIANCE AND WALD SEARCH"

Write-Output ""
Write-Output "7. DESIGN.PY: STRATA, PSU AND SINGLETON DESIGN"

Show-Matches -Path $design_file `
    -Patterns @(
        "singleton",
        "stratum",
        "psu",
        "variance",
        "Taylor",
        "scale",
        "center",
        "degrees",
        "df"
    ) `
    -Before 4 -After 8 `
    -Label "DESIGN.PY DESIGN IMPLEMENTATION"

Write-Output ""
Write-Output "8. SAMPLE.PY: SAMPLE CONSTRUCTION AND SINGLETONS"

Show-Matches -Path $sample_file `
    -Patterns @(
        "singleton",
        "stratum",
        "psu",
        "design",
        "variance",
        "df",
        "n_psus",
        "singletons"
    ) `
    -Before 4 -After 8 `
    -Label "SAMPLE.PY SAMPLE IMPLEMENTATION"

Write-Output ""
Write-Output "9. VARIANCE_METHOD.PY: VARIANCE ESTIMATION"

Show-Matches -Path $variance_file `
    -Patterns @(
        "singleton",
        "scale",
        "center",
        "Taylor",
        "variance",
        "covariance",
        "stratum",
        "psu",
        "replicate"
    ) `
    -Before 5 -After 10 `
    -Label "VARIANCE_METHOD.PY IMPLEMENTATION"

Write-Output ""
Write-Output "10. BASE.PY: BACKEND CALL"

Show-Matches -Path $base_file `
    -Patterns @(
        "fit_glm_rs",
        "svy_rs",
        "cov_matrix",
        "wald_adj",
        "df_resid"
    ) `
    -Before 6 -After 12 `
    -Label "RUST BACKEND CALL AND RESULT HANDLING"

Write-Output ""
Write-Output "11. SVY_RS PYTHON WRAPPER"

Show-Section -Path $rs_init_file -Start 1 -Count 300 `
    -Label "SVY_RS INIT SOURCE"

Write-Output ""
Write-Output "12. SCRIPT 55: SECTION 24C-23 AND RELATED DIAGNOSTICS"

Show-Matches -Path $script55 `
    -Patterns @(
        "24C-23",
        "a3a_",
        "covariance_region",
        "cov_matrix",
        "eigval",
        "matrix_rank",
        "condition",
        "wald_adj",
        "df_den",
        "design_df",
        "WALD"
    ) `
    -Before 5 -After 12 `
    -Label "SCRIPT 55 READ-ONLY SEARCH"

Write-Output ""
Write-Output "13. SCRIPT 55: SURVEY DESIGN AND MODEL FITTING"

Show-Matches -Path $script55 `
    -Patterns @(
        "svy.Design",
        "svy.Sample",
        "singleton",
        "sample.glm",
        "fitted =",
        "geo_results",
        "wmedian",
        "strata",
        "idstd"
    ) `
    -Before 4 -After 8 `
    -Label "SCRIPT 55 DESIGN AND FITTING SEARCH"

Write-Output ""
Write-Output "14. INSPECTION COMPLETED"

Write-Output "Output file:"
Write-Output $output_file

Write-Output ""
Write-Output "No regression was executed."
Write-Output "No source file was modified."

Stop-Transcript

Write-Host ""
Write-Host "INSPECTION FINISHED" -ForegroundColor Green
Write-Host "Report saved to:"
Write-Host $output_file