from pathlib import Path
import polars as pl

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "processed" / "indonesia_2023.parquet"

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pl.read_parquet(data_path)

def report_valid_responses(variable, description, valid_codes):
    print("\n" + "=" * 70)
    print(f"{variable}: {description}")
    print("=" * 70)

    valid = df.filter(pl.col(variable).is_in(valid_codes))
    denominator = valid.height

    print(f"Valid responses: {denominator:,}")
    print(f"Excluded rows (special codes or null): {df.height - denominator:,}")

    if denominator == 0:
        print("No valid responses.")
        return

    result = (
        valid.group_by(variable)
        .len(name="count")
        .with_columns(
            (pl.col("count") / denominator * 100)
            .round(2)
            .alias("percent_of_valid")
        )
        .sort(variable)
    )

    print(result)
    print(f"Percentage denominator: {denominator:,}")

report_valid_responses(
    "k82",
    "Existing loan or line of credit",
    [1, 2, 3, 4]
)

report_valid_responses(
    "k162",
    "Applied for a loan or line of credit",
    [1, 2, 3, 4]
)

report_valid_responses(
    "k20a1",
    "Outcome of most recent credit application",
    [1, 2, 3, 4]
)

report_valid_responses(
    "k17",
    "Main reason for not applying for credit",
    [1, 2, 3, 4, 5, 6, 7]
)

report_valid_responses(
    "k30",
    "Access to finance as an obstacle",
    [0, 1, 2, 3, 4]
)

print("\nAll corrected descriptive tables completed.")
