"""Flask route blueprints for the Student Performance Prediction System."""

from __future__ import annotations

import json
import time
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request

from app.model_loader import ModelNotFoundError, get_feature_names, run_prediction
from app.validator import ALLOWED_VALUES, FEATURE_ORDER, INT_RANGES, ValidationError, validate_and_prepare

main_bp = Blueprint("main", __name__)
api_bp  = Blueprint("api",  __name__)


# ─────────────────────────────────────────────
#  Frontend route
# ─────────────────────────────────────────────
@main_bp.route("/", methods=["GET"])
def index():
    """Serve the student input form."""
    schema = {
        "allowed_values": ALLOWED_VALUES,
        "int_ranges":     {k: {"min": v[0], "max": v[1]} for k, v in INT_RANGES.items()},
        "feature_order":  FEATURE_ORDER,
    }
    return render_template("index.html", schema_json=json.dumps(schema))


# ─────────────────────────────────────────────
#  API: Health check
# ─────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    """Return service health status."""
    try:
        features = get_feature_names()
        model_ok = True
        model_msg = f"Model loaded — {len(features)} features."
    except ModelNotFoundError as exc:
        model_ok = False
        model_msg = str(exc)

    status_code = 200 if model_ok else 503
    return jsonify({
        "status":    "ok" if model_ok else "degraded",
        "model":     "ready" if model_ok else "not_loaded",
        "message":   model_msg,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), status_code


# ─────────────────────────────────────────────
#  API: Feature schema
# ─────────────────────────────────────────────
@api_bp.route("/schema", methods=["GET"])
def schema():
    """Return the expected feature schema for client reference."""
    return jsonify({
        "status":          "ok",
        "feature_order":   FEATURE_ORDER,
        "categorical":     {k: v for k, v in ALLOWED_VALUES.items()},
        "numeric_ranges":  {k: {"min": v[0], "max": v[1]} for k, v in INT_RANGES.items()},
        "total_features":  len(FEATURE_ORDER),
    })


# ─────────────────────────────────────────────
#  API: Predict
# ─────────────────────────────────────────────
@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with 30 student features and returns the
    predicted final grade (G3, 0–20 scale).

    Returns 200 on success, 400 on validation error, 503 if model missing.
    """
    # --- Parse body ---
    if not request.is_json:
        return jsonify({
            "status":  "error",
            "code":    "INVALID_CONTENT_TYPE",
            "message": "Content-Type must be application/json.",
        }), 415

    raw = request.get_json(silent=True)
    if raw is None:
        return jsonify({
            "status":  "error",
            "code":    "INVALID_JSON",
            "message": "Request body is not valid JSON.",
        }), 400

    # --- Validate ---
    try:
        student_df = validate_and_prepare(raw)
    except ValidationError as exc:
        return jsonify({
            "status":  "error",
            "code":    "VALIDATION_ERROR",
            "message": str(exc),
        }), 400

    # --- Predict ---
    t0 = time.perf_counter()
    try:
        result = run_prediction(student_df)
    except ModelNotFoundError as exc:
        return jsonify({
            "status":  "error",
            "code":    "MODEL_NOT_FOUND",
            "message": str(exc),
        }), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "status":  "error",
            "code":    "PREDICTION_FAILED",
            "message": f"Unexpected model error: {exc}",
        }), 500

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "status":     "ok",
        "input":      raw,
        "prediction": result,
        "latency_ms": latency_ms,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    }), 200


# ─────────────────────────────────────────────
#  API: Batch predict
# ─────────────────────────────────────────────
@api_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Accepts a JSON array of student records and returns predictions for each.
    Maximum 50 records per request.
    """
    if not request.is_json:
        return jsonify({
            "status":  "error",
            "code":    "INVALID_CONTENT_TYPE",
            "message": "Content-Type must be application/json.",
        }), 415

    raw_list = request.get_json(silent=True)
    if not isinstance(raw_list, list):
        return jsonify({
            "status":  "error",
            "code":    "INVALID_INPUT",
            "message": "Request body must be a JSON array of student objects.",
        }), 400

    if len(raw_list) == 0:
        return jsonify({
            "status":  "error",
            "code":    "EMPTY_BATCH",
            "message": "Batch must contain at least one student record.",
        }), 400

    if len(raw_list) > 50:
        return jsonify({
            "status":  "error",
            "code":    "BATCH_TOO_LARGE",
            "message": "Maximum 50 records per batch request.",
        }), 400

    results = []
    for idx, raw in enumerate(raw_list):
        try:
            student_df = validate_and_prepare(raw)
            result = run_prediction(student_df)
            results.append({"index": idx, "status": "ok", "prediction": result})
        except ValidationError as exc:
            results.append({"index": idx, "status": "error", "code": "VALIDATION_ERROR", "message": str(exc)})
        except ModelNotFoundError as exc:
            return jsonify({"status": "error", "code": "MODEL_NOT_FOUND", "message": str(exc)}), 503
        except Exception as exc:  # noqa: BLE001
            results.append({"index": idx, "status": "error", "code": "PREDICTION_FAILED", "message": str(exc)})

    return jsonify({
        "status":    "ok",
        "total":     len(raw_list),
        "results":   results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), 200


# ─────────────────────────────────────────────
#  Global error handlers
# ─────────────────────────────────────────────
@main_bp.app_errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "code": "NOT_FOUND", "message": "Endpoint not found."}), 404


@main_bp.app_errorhandler(405)
def method_not_allowed(e):
    return jsonify({"status": "error", "code": "METHOD_NOT_ALLOWED", "message": "HTTP method not allowed."}), 405
