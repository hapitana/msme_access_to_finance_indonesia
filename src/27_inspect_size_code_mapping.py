
from pathlib import Path
import polars as pl

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "indonesia_2023.parquet"
)
df = pl.read_parquet(data_path)

print("=" * 70)
print("INSPECT SIZE CODE MAPPING")
print("=" * 70)

print("\n1. DISTRIBUTION OF a6a")
print(
    df.group_by("a6a")
    .len()
    .sort("a6a")
)

print("\n2. DISTRIBUTION OF stratificationsizecode")
print(
    df.group_by("stratificationsizecode")
    .len()
    .sort("stratificationsizecode")
)

print("\n3. CROSS-TABULATION: a6a vs stratificationsizecode")
cross_tab = (
    df.group_by(["a6a", "stratificationsizecode"])
    .len()
    .sort(["a6a", "stratificationsizecode"])
)

print(cross_tab)

print("\n4. SIZE CODE MAPPING BY PANEL STATUS")
panel_mapping = (
    df.group_by([
        "stratificationpanelcode",
        "a6a",
        "stratificationsizecode",
    ])
    .len()
    .sort([
        "stratificationpanelcode",
        "a6a",
        "stratificationsizecode",
    ])
)

print(panel_mapping)

print("\nInspection completed.")
