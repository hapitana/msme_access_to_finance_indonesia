
from pathlib import Path

import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# 1. Set project paths
project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "processed"
    / "model_dataset.parquet"
)
output_dir = project_root / "data" / "processed"

# 2. Load the prepared dataset
df = pl.read_parquet(data_path)

print("=" * 60)
print("LOGISTIC REGRESSION: FINANCIAL OBSTACLE PREDICTION")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# 3. Define target and predictors
target = "major_obstacle"

categorical_features = [
    "firm_size_code",
    "credit_status",
    "industry_code",
    "region_code",
]

# Convert Polars columns to NumPy arrays.
# No pandas is used.
X = df.select(categorical_features).to_numpy()
y = df.get_column(target).to_numpy()

# 4. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(y_train)}")
print(f"Testing observations: {len(y_test)}")

# 5. Encode categorical predictors
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            list(range(len(categorical_features))),
        )
    ]
)

# 6. Build the model
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

# 7. Train
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluate
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No major obstacle", "Major obstacle"],
        zero_division=0,
    )
)

# 10. Save predictions for further analysis
predictions = pl.DataFrame(
    {
        "actual_major_obstacle": y_test,
        "predicted_major_obstacle": y_pred,
        "predicted_probability": y_prob,
    }
)

prediction_path = output_dir / "logistic_regression_predictions.csv"
predictions.write_csv(prediction_path)

print(f"\nPredictions saved to: {prediction_path}")
print("Model training completed.")
