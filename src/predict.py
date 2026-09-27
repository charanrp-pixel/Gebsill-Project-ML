"""Load the trained pipeline and predict a student's final grade."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "student_performance_model.joblib"
FEATURES_PATH = ROOT / "model" / "feature_schema.json"


def predict_student(student_data: dict[str, Any]) -> dict[str, float]:
    """Return a final-grade prediction from exactly the trained feature fields."""
    if not isinstance(student_data, dict):
        raise TypeError("student_data must be a dictionary")
    if not FEATURES_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model and feature schema are required; run src/train.py first")

    feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    received = set(student_data)
    expected = set(feature_names)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(f"Input fields must match the model schema exactly. Missing: {missing}; extra: {extra}")

    student_frame = pd.DataFrame([student_data], columns=feature_names)
    pipeline = joblib.load(MODEL_PATH)
    prediction = float(pipeline.predict(student_frame)[0])
    # Grades have a documented 0–20 scale. Clip only at the serving boundary.
    return {"predicted_grade": round(float(min(20.0, max(0.0, prediction))), 2)}
