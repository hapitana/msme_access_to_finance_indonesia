from pathlib import Path

import polars as pl
import pyreadstat

project_dir = Path(__file__).resolve().parent.parent

data_path = (
    project_dir
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

data, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

df = pl.DataFrame(data)

base_columns = ["a4a", "a2", "a6a"]
panel_column = "stratificationpanelcode"

required = base_columns + [panel_column, "strata"]

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing variables: {missing}")

print("=" * 65)
print("WBES INDONESIA 2023: PANEL AND SINGLETON STRUCTURE")
print("=" * 65)

panel_cells = (
    df.group_by(base_columns + [panel_column])
    .agg(
        pl.len().alias("panel_cell_size"),
        pl.col("strata").n_unique().alias("strata_codes"),
    )
)

base_cells = (
    df.group_by(base_columns)
    .agg(
        pl.len().alias("total_observations"),
        pl.col(panel_column).n_unique().alias("panel_categories"),
    )
)

combined = panel_cells.join(
    base_cells,
    on=base_columns,
    how="left",
)

print("\n1. Base sampling cells")

print(
    "Distinct industry-region-size combinations:",
    base_cells.height,
)

print(
    "Base cells represented in both panel categories:",
    base_cells.filter(
        pl.col("panel_categories") == 2
    ).height,
)

print("\n2. Panel-specific singleton cells")

panel_singletons = combined.filter(
    pl.col("panel_cell_size") == 1
)

print(
    "Panel-specific singleton cells:",
    panel_singletons.height,
)

print(
    "Singleton cells with more than one observation "
    "across panel categories:",
    panel_singletons.filter(
        pl.col("total_observations") > 1
    ).height,
)

print(
    "Singleton cells with only one observation "
    "across all panel categories:",
    panel_singletons.filter(
        pl.col("total_observations") == 1
    ).height,
)

print("\n3. Base cells split across panel categories")

split_cells = base_cells.filter(
    pl.col("panel_categories") == 2
)

print("Cells split across panels:", split_cells.height)

split_with_singleton = (
    combined
    .filter(
        (pl.col("panel_cell_size") == 1)
        & (pl.col("panel_categories") == 2)
    )
)

print(
    "Panel-specific singleton groups in split cells:",
    split_with_singleton.height,
)

print("\n4. Examples of split cells")

print(
    split_with_singleton
    .sort("total_observations", descending=True)
    .head(20)
)

print("\n5. Save outputs")

combined.write_csv(
    output_dir / "panel_sampling_cell_details.csv"
)

panel_singletons.write_csv(
    output_dir / "panel_singleton_cells.csv"
)

split_with_singleton.write_csv(
    output_dir / "panel_split_singleton_cells.csv"
)

split_cells.write_csv(
    output_dir / "sampling_cells_split_across_panels.csv"
)

print("Files saved to:", output_dir)

print("\nThis is a structural diagnostic, not a justification "
      "for changing the official survey strata.")

print("\nAudit completed.")