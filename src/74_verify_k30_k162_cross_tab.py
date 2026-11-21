import polars as pl

input_path = (
    "data/processed/outcome_robustness/"
    "k30_k162_raoscott_input.parquet"
)

df = pl.read_parquet(input_path)

print("COLUMNS")
print(df.columns)

print("\nK30 DISTRIBUTION IN ANALYTICAL SAMPLE")
print(
    df.group_by("k30")
    .len()
    .sort("k30")
)

print("\nOBSTACLE_MAJOR DISTRIBUTION")
print(
    df.group_by("obstacle_major")
    .len()
    .sort("obstacle_major")
)

print("\nK30 BY OBSTACLE_MAJOR")
print(
    df.group_by(["k30", "obstacle_major"])
    .len()
    .sort(["k30", "obstacle_major"])
)

print("\nVALIDATION")
print(
    "K30 codes 3-4:",
    df.filter(pl.col("k30").is_in([3, 4])).height
)

print(
    "obstacle_major = 1:",
    df.filter(pl.col("obstacle_major") == 1).height
)

print(
    "Mismatch between K30 3-4 and obstacle_major:",
    df.filter(
        (
            pl.col("k30").is_in([3, 4])
        ) != (
            pl.col("obstacle_major") == 1
        )
    ).height
)

print("\nK162 DISTRIBUTION")
print(
    df.group_by("k162")
    .len()
    .sort("k162")
)

print("\nK30 BY K162")
print(
    df.group_by(["k30", "k162"])
    .len()
    .sort(["k30", "k162"])
)

print("\nK30 MAJOR EVENTS BY K162")
print(
    df.filter(pl.col("k30").is_in([3, 4]))
    .group_by("k162")
    .len()
    .sort("k162")
)

print("\nCREDIT APPLICATION CODING CHECK")
print(
    df.group_by(["k162", "credit_application"])
    .len()
    .sort(["k162", "credit_application"])
)