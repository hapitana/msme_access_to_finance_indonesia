
from pathlib import Path

import polars as pl


project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / "indonesia_2023.parquet"
)

if not data_path.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {data_path}"
    )

df = pl.read_parquet(data_path)

required_columns = [
    "k30",
    "k82",
    "wstrict",
    "wmedian",
    "wweak",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

weights = ["wstrict", "wmedian", "wweak"]

valid_outcome = pl.col("k30").is_in([0, 1, 2, 3, 4])
valid_credit = pl.col("k82").is_in([1, 2, 3, 4])

valid_weights = pl.all_horizontal([
    pl.col(weight).is_not_null()
    & pl.col(weight).is_finite()
    & (pl.col(weight) > 0)
    for weight in weights
])

print("=" * 65)
print("SENSITIVITY SAMPLE AUDIT")
print("=" * 65)
print(f"Original observations: {df.height:,}")

checks = [
    ("Valid obstacle response (k30)", valid_outcome),
    ("Valid credit status (k82)", valid_credit),
    (
        "Valid obstacle AND credit status",
        valid_outcome & valid_credit,
    ),
    (
        "Valid obstacle AND all three weights",
        valid_outcome & valid_weights,
    ),
    (
        "Valid obstacle, credit status, and all weights",
        valid_outcome & valid_credit & valid_weights,
    ),
]

for label, condition in checks:
    count = df.filter(condition).height
    percentage = count / df.height * 100 if df.height else 0
    print(f"{label}: {count:,} ({percentage:.2f}%)")

print("\nK30 RESPONSE CODES")
print(
    df.group_by("k30")
    .len()
    .sort("k30")
)

print("\nK82 RESPONSE CODES")
print(
    df.group_by("k82")
    .len()
    .sort("k82")
)

print("\nWEIGHT VALIDITY")
for weight in weights:
    valid_count = df.filter(
        pl.col(weight).is_not_null()
        & pl.col(weight).is_finite()
        & (pl.col(weight) > 0)
    ).height

    print(
        f"{weight}: {valid_count:,} valid "
        f"({valid_count / df.height * 100:.2f}%)"
        if df.height
        else f"{weight}: 0 observations"
    )
