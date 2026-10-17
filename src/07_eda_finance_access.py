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
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pl.read_parquet(data_path)

print("=" * 65)
print("MSME ACCESS TO FINANCE — INITIAL EDA")
print("=" * 65)
print(f"Total observations: {df.height:,}")
print(f"Total variables: {df.width:,}")

variables = {
    "k82": "Existing loan or line of credit",
    "k162": "Applied for a loan or line of credit",
    "k20a1": "Outcome of most recent credit application",
    "k17": "Main reason for not applying",
    "k30": "Access to finance as an obstacle",
}

for variable, description in variables.items():
    print("\n" + "-" * 65)
    print(f"{variable}: {description}")
    print("-" * 65)

    series = df.get_column(variable)
    non_null_count = series.len() - series.null_count()

    print(f"Total rows: {series.len():,}")
    print(f"Null values: {series.null_count():,}")
    print(f"Non-null values: {non_null_count:,}")

    frequency = (
        df.group_by(variable)
        .len(name="count")
        .sort(variable)
    )

    if non_null_count > 0:
        frequency = frequency.with_columns(
            (
                pl.col("count")
                / non_null_count
                * 100
            ).round(2).alias("pct_of_non_null")
        )

    print(frequency)

print("\n" + "=" * 65)
print("EDA FREQUENCY CHECK COMPLETED")
print("No rows or columns were modified.")
print("=" * 65)
