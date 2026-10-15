"""Data validation for incoming student prediction requests."""

from __future__ import annotations

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Allowed categorical values (derived from the UCI student-mat dataset)
# -------------------------------------------------------------------
ALLOWED_VALUES: dict[str, list] = {
    "school":    ["GP", "MS"],
    "sex":       ["F", "M"],
    "address":   ["U", "R"],
    "famsize":   ["LE3", "GT3"],
    "Pstatus":   ["T", "A"],
    "Mjob":      ["teacher", "health", "services", "at_home", "other"],
    "Fjob":      ["teacher", "health", "services", "at_home", "other"],
    "reason":    ["home", "reputation", "course", "other"],
    "guardian":  ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup":    ["yes", "no"],
    "paid":      ["yes", "no"],
    "activities":["yes", "no"],
    "nursery":   ["yes", "no"],
    "higher":    ["yes", "no"],
    "internet":  ["yes", "no"],
    "romantic":  ["yes", "no"],
}

# Inclusive integer ranges
INT_RANGES: dict[str, tuple[int, int]] = {
    "age":        (15, 22),
    "Medu":       (0, 4),
    "Fedu":       (0, 4),
    "traveltime": (1, 4),
    "studytime":  (1, 4),
    "failures":   (0, 4),
    "famrel":     (1, 5),
    "freetime":   (1, 5),
    "goout":      (1, 5),
    "Dalc":       (1, 5),
    "Walc":       (1, 5),
    "health":     (1, 5),
    "absences":   (0, 93),
}

ALL_FEATURES: list[str] = (
    list(ALLOWED_VALUES.keys()) + list(INT_RANGES.keys())
)

# Canonical order matching the trained model's feature_schema.json
FEATURE_ORDER: list[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]


class ValidationError(ValueError):
    """Raised when incoming request data fails validation."""


def validate_and_prepare(raw: dict) -> pd.DataFrame:
    """
    Validate *raw* student dict and return a single-row DataFrame
    ready for the scikit-learn pipeline.

    Raises ValidationError with a descriptive message on any problem.
    """
    if not isinstance(raw, dict):
        raise ValidationError("Request body must be a JSON object.")

    errors: list[str] = []

    # --- Check for unknown / missing keys ---
    received = set(raw.keys())
    expected = set(FEATURE_ORDER)
    missing = sorted(expected - received)
    extra   = sorted(received - expected)
    if missing:
        errors.append(f"Missing fields: {missing}")
    if extra:
        errors.append(f"Unknown fields: {extra}")
    if errors:
        raise ValidationError("; ".join(errors))

    # --- Validate each field ---
    clean: dict = {}

    for field, allowed in ALLOWED_VALUES.items():
        val = raw[field]
        if not isinstance(val, str):
            errors.append(f"'{field}' must be a string (got {type(val).__name__}).")
        elif val not in allowed:
            errors.append(f"'{field}' must be one of {allowed} (got '{val}').")
        else:
            clean[field] = val

    for field, (lo, hi) in INT_RANGES.items():
        val = raw[field]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            errors.append(f"'{field}' must be a number (got {type(val).__name__}).")
        else:
            int_val = int(val)
            if not (lo <= int_val <= hi):
                errors.append(f"'{field}' must be between {lo} and {hi} (got {int_val}).")
            else:
                clean[field] = int_val

    if errors:
        raise ValidationError("; ".join(errors))

    # --- Build ordered DataFrame ---
    df = pd.DataFrame([clean], columns=FEATURE_ORDER)

    # Ensure numeric columns have correct dtypes
    numeric_cols = list(INT_RANGES.keys())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    return df
