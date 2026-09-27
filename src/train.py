"""Train and evaluate the student final-grade regression pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student-mat.csv"
MODEL_PATH = ROOT / "model" / "student_performance_model.joblib"
RESULTS_PATH = ROOT / "results" / "evaluation_results.txt"
FEATURES_PATH = ROOT / "model" / "feature_schema.json"
EXCLUDED_FEATURES = {"G1", "G2", "G3"}


def load_and_validate_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load the semicolon-delimited UCI file and perform basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, sep=";")
    print(f"Rows: {len(data)} | Columns: {len(data.columns)}")
    print("Column names:", data.columns.tolist())
    print("Data types:\n", data.dtypes)
    print("First five rows:\n", data.head())
    print("Missing values:\n", data.isna().sum())
    print("Duplicate rows:", int(data.duplicated().sum()))
    print("Statistical summary:\n", data.describe(include="all").transpose())

    if "G3" not in data.columns:
        raise ValueError("Required target column G3 is missing")
    if data["G3"].isna().any():
        raise ValueError("Target G3 contains missing values")
    if not pd.api.types.is_numeric_dtype(data["G3"]):
        raise TypeError("Target G3 must be numeric")
    if not data["G3"].between(0, 20).all():
        raise ValueError("G3 contains grades outside the documented 0–20 range")

    duplicates = int(data.duplicated().sum())
    if duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates} duplicate rows.")
    else:
        print("No duplicate rows to remove.")
    missing = int(data.isna().sum().sum())
    print("Missing-value check:", f"{missing} missing value(s); pipeline imputation will handle feature gaps." if missing else "no missing values.")

    # UCI documents these ranges for the math dataset. Reject impossible values;
    # do not trim statistically unusual but valid observations.
    expected_ranges = {
        "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
        "studytime": (1, 4), "failures": (0, 4), "famrel": (1, 5), "freetime": (1, 5),
        "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
        "absences": (0, 93),
    }
    for column, (minimum, maximum) in expected_ranges.items():
        if column in data and not data[column].between(minimum, maximum).all():
            raise ValueError(f"{column} contains values outside {minimum}–{maximum}")

    features = [column for column in data.columns if column not in EXCLUDED_FEATURES]
    if EXCLUDED_FEATURES.intersection(features):
        raise AssertionError("Grade leakage detected in features")
    if len(features) != len(data.columns) - 3:
        raise AssertionError("Unexpected feature schema")
    return data, features


def build_pipeline(numerical_features: list[str], categorical_features: list[str]) -> Pipeline:
    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def main() -> None:
    data, feature_names = load_and_validate_data()
    X = data[feature_names]
    y = data["G3"]
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    print("G1/G2 excluded:", "G1" not in X and "G2" not in X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipeline = build_pipeline(numerical_features, categorical_features)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_names, indent=2) + "\n", encoding="utf-8")

    comparison = pd.DataFrame({
        "Actual G3": y_test.to_numpy(),
        "Predicted G3": predictions,
        "Absolute Error": np.abs(y_test.to_numpy() - predictions),
    }, index=y_test.index).head(10)
    results = (
        "UCI Student Performance (student-mat.csv)\n"
        f"Rows used: {len(data)}\nTrain rows: {len(X_train)}\nTest rows: {len(X_test)}\n"
        f"Features: {len(feature_names)}\nG1/G2/G3 used as inputs: no\n"
        f"MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nR2: {r2:.6f}\n"
        f"Predictions within 0–20 before clipping: {bool(np.logical_and(predictions >= 0, predictions <= 20).all())}\n\n"
        "Test-set examples (not clipped):\n" + comparison.to_string(float_format=lambda value: f"{value:.3f}") + "\n"
    )
    RESULTS_PATH.write_text(results, encoding="utf-8")
    print(results)
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
