
from pathlib import Path

import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root / "data" / "processed" / "model_dataset.parquet"
)
output_dir = project_root / "data" / "processed"

df = pl.read_parquet(data_path)

target = "major_obstacle"
features = [
    "firm_size_code",
    "credit_status",
    "industry_code",
    "region_code",
]

X = df.select(features).to_numpy()
y = df.get_column(target).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
            list(range(len(features))),
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

encoder = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = encoder.get_feature_names_out()
coefficients = classifier.coef_[0]

results = pl.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
    }
).with_columns(
    pl.col("coefficient").abs().alias("absolute_coefficient")
).sort(
    "absolute_coefficient",
    descending=True,
)

print("=" * 60)
print("LOGISTIC REGRESSION COEFFICIENT ANALYSIS")
print("=" * 60)

print("\nLargest absolute coefficients:")
print(results.head(20))

output_path = output_dir / "logistic_regression_coefficients.csv"
results.write_csv(output_path)

print(f"\nResults saved to: {output_path}")
print("Coefficient analysis completed.")
