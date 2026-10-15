"""Unit and integration tests for Student Performance Prediction Flask API."""

import json
from pathlib import Path
import unittest

from app import create_app

ROOT = Path(__file__).resolve().parents[1]


class StudentPerformanceAPITestCase(unittest.TestCase):
    """Test suite covering all endpoints, validations, and model predictions."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        with open(ROOT / "sample_input.json", "r", encoding="utf-8") as f:
            cls.sample_payload = json.load(f)

    def test_frontend_index(self):
        """Test GET / renders HTML frontend successfully."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student Performance Prediction", response.data)
        self.assertIn(b"predict-form", response.data)

    def test_health_check(self):
        """Test GET /api/health returns model status and 200 OK."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "ready")
        self.assertIn("Model loaded", data["message"])
        self.assertIn("timestamp", data)

    def test_schema_endpoint(self):
        """Test GET /api/schema returns feature definitions."""
        response = self.client.get("/api/schema")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["total_features"], 30)
        self.assertIn("school", data["categorical"])
        self.assertIn("age", data["numeric_ranges"])

    def test_predict_success(self):
        """Test POST /api/predict returns accurate prediction with valid payload."""
        response = self.client.post(
            "/api/predict",
            data=json.dumps(self.sample_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("prediction", data)
        pred = data["prediction"]
        self.assertIn("predicted_grade", pred)
        self.assertEqual(pred["grade_out_of"], 20)
        self.assertAlmostEqual(pred["predicted_grade"], 9.49, places=1)
        self.assertIn("percentage", pred)
        self.assertIn("performance_band", pred)
        self.assertIn("latency_ms", data)

    def test_predict_missing_field(self):
        """Test POST /api/predict fails validation when fields are missing."""
        incomplete_payload = dict(self.sample_payload)
        del incomplete_payload["absences"]

        response = self.client.post(
            "/api/predict",
            data=json.dumps(incomplete_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["code"], "VALIDATION_ERROR")
        self.assertIn("Missing fields", data["message"])

    def test_predict_extra_field(self):
        """Test POST /api/predict fails validation on unknown/extra fields."""
        extra_payload = dict(self.sample_payload)
        extra_payload["extra_col"] = "unexpected"

        response = self.client.post(
            "/api/predict",
            data=json.dumps(extra_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["code"], "VALIDATION_ERROR")
        self.assertIn("Unknown fields", data["message"])

    def test_predict_invalid_range(self):
        """Test POST /api/predict rejects out-of-range numeric values."""
        invalid_payload = dict(self.sample_payload)
        invalid_payload["age"] = 99  # Allowed is 15-22

        response = self.client.post(
            "/api/predict",
            data=json.dumps(invalid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["code"], "VALIDATION_ERROR")
        self.assertIn("age", data["message"])

    def test_predict_invalid_category(self):
        """Test POST /api/predict rejects invalid categorical values."""
        invalid_payload = dict(self.sample_payload)
        invalid_payload["sex"] = "Unknown"  # Allowed: ['F', 'M']

        response = self.client.post(
            "/api/predict",
            data=json.dumps(invalid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["code"], "VALIDATION_ERROR")
        self.assertIn("sex", data["message"])

    def test_predict_invalid_content_type(self):
        """Test POST /api/predict returns 415 when Content-Type is not application/json."""
        response = self.client.post(
            "/api/predict",
            data="plain text",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 415)
        data = response.get_json()
        self.assertEqual(data["code"], "INVALID_CONTENT_TYPE")

    def test_batch_predict(self):
        """Test POST /api/predict/batch processes a list of student records."""
        batch_payload = [self.sample_payload, dict(self.sample_payload)]
        response = self.client.post(
            "/api/predict/batch",
            data=json.dumps(batch_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["status"], "ok")

    def test_404_handler(self):
        """Test 404 returns structured JSON."""
        response = self.client.get("/non_existent_route")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data["code"], "NOT_FOUND")

    def test_405_handler(self):
        """Test 405 method not allowed returns structured JSON."""
        response = self.client.post("/api/health")
        self.assertEqual(response.status_code, 405)
        data = response.get_json()
        self.assertEqual(data["code"], "METHOD_NOT_ALLOWED")


if __name__ == "__main__":
    unittest.main()
