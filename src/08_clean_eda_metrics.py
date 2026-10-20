from pathlib import Path

import polars as pl


project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root / "data" / "processed" / "indonesia_2023.parquet"
)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pl.read_parquet(data_path)

# Substantive response codes documented in the dataset's value labels.
VARIABLES = [
    ("k82", "Existing loan or line of credit", [1, 2, 3, 4]),
    ("k162", "Applied for a loan or line of credit", [1, 2, 3, 4]),
    ("k20a1", "Outcome of most recent credit application", [1, 2, 3, 4]),
    ("k17", "Main reason for not applying for credit", [1, 2, 3, 4, 5, 6, 7]),
    ("k30", "Access to finance as an obstacle", [0, 1, 2, 3, 4]),
]

# Special codes confirmed in the embedded value labels.
KNOWN_SPECIAL_CODES = {
    "k82": {-9},
    "k162": {-9},
    "k20a1": {-9, -6},
    "k17": {-9},
    "k30": {-9, -7},
}

TOTAL_ROWS = df.height


def report_valid_responses(variable, description, valid_codes):
    print("\n" + "=" * 72)
    print(f"{variable}: {description}")
    print("=" * 72)

    if variable not in df.columns:
        raise KeyError(f"Variable not found in dataset: {variable}")

    valid = df.filter(pl.col(variable).is_in(valid_codes))
    excluded = df.filter(
    pl.col(variable).is_null()
    | ~pl.col(variable).is_in(valid_codes)
)

    valid_n = valid.height
    excluded_n = excluded.height

    print(f"Total dataset rows: {TOTAL_ROWS:,}")
    print(f"Valid responses: {valid_n:,}")
    print(f"Excluded rows: {excluded_n:,}")

    valid_share = valid_n / TOTAL_ROWS * 100 if TOTAL_ROWS else 0
    print(f"Valid responses as % of all rows: {valid_share:.2f}%")

    print("\nExcluded values (including nulls):")
    excluded_counts = (
        excluded
        .group_by(variable)
        .len(name="count")
        .sort(variable)
    )
    print(excluded_counts)

    observed_special_codes = set(
        excluded
        .filter(pl.col(variable).is_not_null())
        .get_column(variable)
        .unique()
        .to_list()
    )
    unexpected_codes = observed_special_codes - KNOWN_SPECIAL_CODES[variable]

    if unexpected_codes:
        print(
            "WARNING: Undocumented excluded codes detected: "
            f"{sorted(unexpected_codes)}"
        )
    else:
        print("No undocumented non-null excluded codes detected.")

    if valid_n == 0:
        print("No valid responses; category percentages not calculated.")
        return

    result = (
        valid
        .group_by(variable)
        .len(name="count")
        .with_columns(
            (pl.col("count") / valid_n * 100)
            .round(2)
            .alias("percent_of_valid")
        )
        .sort(variable)
    )

    print("\nValid response distribution:")
    print(result)
    print(f"Category percentage denominator: {valid_n:,}")


if __name__ == "__main__":
    print("MSME ACCESS TO FINANCE: DESCRIPTIVE DATA AUDIT")
    print(f"Dataset rows: {TOTAL_ROWS:,}")
    print(f"Dataset columns: {len(df.columns):,}")

    for variable, description, valid_codes in VARIABLES:
        report_valid_responses(
            variable,
            description,
            valid_codes,
        )

    print("\nDescriptive data audit completed.")
