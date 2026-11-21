from pathlib import Path

import pyreadstat

project_dir = Path(__file__).resolve().parent.parent

data_path = (
    project_dir
    / "data"
    / "raw"
    / "Indonesia-2023-full-data.dta"
)

if not data_path.exists():
    raise FileNotFoundError(data_path)

_, metadata = pyreadstat.read_dta(
    str(data_path),
    output_format="dict",
    apply_value_formats=False,
)

print("=" * 60)
print("WBES INDONESIA 2023: SAMPLING METADATA")
print("=" * 60)

print("\n1. Dataset notes")

notes = getattr(metadata, "notes", None)

if notes:
    for index, note in enumerate(notes, start=1):
        print(f"\nNote {index}:")
        print(str(note))
else:
    print("No dataset notes found in PyReadStat metadata.")

print("\n2. Strata variable label")

labels = metadata.column_names_to_labels

print(labels.get("strata", "No label available"))

print("\n3. Relevant variable labels")

for variable in [
    "idstd",
    "strata",
    "a4a",
    "a2",
    "a6a",
    "panel",
    "wmedian",
    "wweak",
    "wstrong",
]:
    label = labels.get(variable)

    if label is not None:
        print(f"{variable}: {label}")
    else:
        print(f"{variable}: NOT FOUND")

print("\n4. Dataset file label")
print(getattr(metadata, "file_label", None))

print("\nMetadata inspection completed.")