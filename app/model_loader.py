"""ML model loader with lazy-loading singleton pattern."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = ROOT / "model" / "student_performance_model.joblib"
FEATURES_PATH = ROOT / "model" / "feature_schema.json"

_pipeline = None
_feature_names: list[str] | None = None


class ModelNotFoundError(RuntimeError):
    """Raised when the serialised model artefact is missing."""


def _load_model():
    global _pipeline, _feature_names
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python src/train.py' first."
        )
    if not FEATURES_PATH.exists():
        raise ModelNotFoundError(
            f"Feature schema not found at {FEATURES_PATH}. "
            "Run 'python src/train.py' first."
        )
    _pipeline = joblib.load(MODEL_PATH)
    _feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


def get_feature_names() -> list[str]:
    """Return the feature names the model was trained on."""
    if _feature_names is None:
        _load_model()
    return _feature_names  # type: ignore[return-value]


def run_prediction(student_df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the trained pipeline on a single-row DataFrame and return
    prediction metadata as a dict.
    """
    if _pipeline is None:
        _load_model()

    raw_pred = float(_pipeline.predict(student_df)[0])
    clipped  = round(float(min(20.0, max(0.0, raw_pred))), 2)

    # --- Performance band ---
    if clipped >= 15:
        band, band_color = "Excellent", "success"
    elif clipped >= 10:
        band, band_color = "Good", "info"
    elif clipped >= 7:
        band, band_color = "Average", "warning"
    else:
        band, band_color = "Needs Improvement", "danger"

    return {
        "predicted_grade":      clipped,
        "grade_out_of":         20,
        "percentage":           round(clipped / 20 * 100, 1),
        "performance_band":     band,
        "performance_color":    band_color,
    }
