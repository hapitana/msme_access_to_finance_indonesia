from pathlib import Path

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

data, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

df = pl.DataFrame(data)

print("=" * 65)
print("WBES INDONESIA 2023: SINGLETON METHOD COMPARISON")
print("=" * 65)

methods = ["scale", "center"]

for method in methods:
    print(f"\n1. Testing singleton method: {method}")

    design = svy.Design(
        stratum="strata",
        psu="idstd",
        wgt="wmedian",
        singleton=method,
    )

    sample = svy.Sample(
        data=df,
        design=design,
    )

    print("Strata:", sample.n_strata)
    print("PSUs:", sample.n_psus)
    print("Singletons:", sample.n_singletons)

    print("\nSingleton handling summary:")

    handling_summary = (
        sample.singletons
        .group_by("handled")
        .agg(
            pl.len().alias("number_of_strata")
        )
    )

    print(handling_summary)

    print("\nDesign configuration:")
    print(sample.design)

    if sample.n_singletons != 584:
        raise ValueError(
            f"Unexpected singleton count for {method}"
        )

    if sample.singletons["handled"].null_count() != 0:
        raise ValueError(
            f"Some singleton strata were not handled by {method}"
        )

print("\nBoth singleton methods passed the configuration test.")
print("No regression estimates were calculated.")

print("\nTest completed.")