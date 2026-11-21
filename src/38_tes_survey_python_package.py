from pathlib import Path
from importlib.metadata import version

import polars as pl
import pyreadstat
import svy

project_dir = Path(__file__).resolve().parent.parent

data_path = (
    project_dir
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

if not data_path.exists():
    raise FileNotFoundError(data_path)

print("=" * 65)
print("WBES INDONESIA 2023: PYTHON SURVEY PACKAGE TEST")
print("=" * 65)

print("\n1. Package versions")

print("Python survey package:", version("svy"))
print("Polars:", pl.__version__)

print("\n2. Reading original Stata dataset")

data, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

df = pl.DataFrame(data)

print("Rows:", df.height)
print("Columns:", df.width)

print("\n3. Defining survey design")

design = svy.Design(
    stratum="strata",
    psu="idstd",
    wgt="wmedian",
)

sample = svy.Sample(
    data=df,
    design=design,
)

print(sample)

print("\n4. Survey design diagnostics")

print("Number of strata:", sample.n_strata)
print("Number of PSUs:", sample.n_psus)
print("Number of singleton strata:", sample.n_singletons)

print("\n5. Singleton details")

print(sample.singletons)

print("\nPackage compatibility test completed.")