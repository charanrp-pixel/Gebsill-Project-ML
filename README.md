# 🎓 Student Performance Prediction System

A **Flask REST API** with a premium web UI that predicts a student's final math grade (G3, 0–20 scale) using a trained **Random Forest** model built on the UCI Student Performance dataset.

---

## 📁 Project Structure

```
Gebsill-Project-ML-main/
├── app/                        # Flask application package
│   ├── __init__.py             # App factory
│   ├── routes.py               # All API & UI route handlers
│   ├── validator.py            # Input validation (Pandas/NumPy)
│   └── model_loader.py         # Lazy-loading ML model singleton
├── templates/
│   └── index.html              # Premium dark UI (glassmorphism)
├── static/
│   └── style.css               # Full stylesheet
├── model/
│   ├── student_performance_model.joblib   # Trained RandomForest pipeline
│   └── feature_schema.json               # 30 feature names
├── data/
│   └── student-mat.csv         # UCI Student Performance dataset
├── src/
│   ├── train.py                # Model training script
│   └── predict.py              # Standalone CLI predictor
├── results/                    # Evaluation output
├── run.py                      # Flask entry point
├── requirements.txt
└── sample_input.json           # Example API payload
```

---

## 🚀 Setup & Running

### 1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. (Optional) Re-train the model

The model file is already included. To retrain:

```bash
python3 src/train.py
```

### 3. Start the Flask server

```bash
python3 run.py
```

The server starts at: **http://127.0.0.1:5000**

---

## 🌐 Web UI

Visit **http://127.0.0.1:5000** in your browser.

- Fill in 30 student features (or click **Fill Sample** for demo data)
- Click **Predict Grade**
- See the animated result with grade, percentage, and performance band

---

## 📡 REST API Reference

### `GET /api/health`

Check if the service and model are ready.

```json
{
  "status": "ok",
  "model": "ready",
  "message": "Model loaded — 30 features.",
  "timestamp": "2026-09-28T16:59:05Z"
}
```

---

### `GET /api/schema`

Returns the full feature schema (all valid values and numeric ranges).

---

### `POST /api/predict`

**Content-Type:** `application/json`

**Request body:** A JSON object with all 30 features.

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

**Success Response (200):**

```json
{
  "status": "ok",
  "input": { "school": "GP", "sex": "F", "age": 18, "...": "..." },
  "prediction": {
    "predicted_grade": 9.49,
    "grade_out_of": 20,
    "percentage": 47.5,
    "performance_band": "Average",
    "performance_color": "warning"
  },
  "latency_ms": 33.02,
  "timestamp": "2026-09-28T16:59:14Z"
}
```

**Error Response (400):**

```json
{
  "status": "error",
  "code": "VALIDATION_ERROR",
  "message": "Missing fields: ['absences']; 'age' must be between 15 and 22 (got 99)."
}
```

---

### `POST /api/predict/batch`

Send up to **50** student records in one request.

```bash
curl -X POST http://127.0.0.1:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{ ...student1... }, { ...student2... }]'
```

**Response:**

```json
{
  "status": "ok",
  "total": 2,
  "results": [
    { "index": 0, "status": "ok", "prediction": { "predicted_grade": 9.49, "..." : "..." } },
    { "index": 1, "status": "error", "code": "VALIDATION_ERROR", "message": "..." }
  ]
}
```

---

## 🧾 Feature Reference

| Feature | Type | Allowed Values / Range |
|---------|------|------------------------|
| `school` | string | `GP`, `MS` |
| `sex` | string | `F`, `M` |
| `age` | int | 15–22 |
| `address` | string | `U`, `R` |
| `famsize` | string | `LE3`, `GT3` |
| `Pstatus` | string | `T`, `A` |
| `Medu` | int | 0–4 |
| `Fedu` | int | 0–4 |
| `Mjob` | string | `teacher`, `health`, `services`, `at_home`, `other` |
| `Fjob` | string | `teacher`, `health`, `services`, `at_home`, `other` |
| `reason` | string | `home`, `reputation`, `course`, `other` |
| `guardian` | string | `mother`, `father`, `other` |
| `traveltime` | int | 1–4 |
| `studytime` | int | 1–4 |
| `failures` | int | 0–4 |
| `schoolsup` | string | `yes`, `no` |
| `famsup` | string | `yes`, `no` |
| `paid` | string | `yes`, `no` |
| `activities` | string | `yes`, `no` |
| `nursery` | string | `yes`, `no` |
| `higher` | string | `yes`, `no` |
| `internet` | string | `yes`, `no` |
| `romantic` | string | `yes`, `no` |
| `famrel` | int | 1–5 |
| `freetime` | int | 1–5 |
| `goout` | int | 1–5 |
| `Dalc` | int | 1–5 |
| `Walc` | int | 1–5 |
| `health` | int | 1–5 |
| `absences` | int | 0–93 |

---

## 🏷️ Performance Bands

| Grade | Band | Description |
|-------|------|-------------|
| 15–20 | 🌟 Excellent | Outstanding |
| 10–14 | 👍 Good | Above average |
| 7–9 | 📚 Average | Satisfactory |
| 0–6 | 💪 Needs Improvement | Requires support |

---

## ⚙️ Error Codes

| HTTP | Code | Description |
|------|------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid or missing fields |
| 400 | `INVALID_JSON` | Malformed JSON body |
| 415 | `INVALID_CONTENT_TYPE` | Must be `application/json` |
| 503 | `MODEL_NOT_FOUND` | Run `python3 src/train.py` first |
| 500 | `PREDICTION_FAILED` | Unexpected model error |

---

## 🛠️ Tech Stack

- **Python** – Core language
- **Flask 3** – REST API + HTML frontend
- **Pandas / NumPy** – Data preparation and validation
- **scikit-learn** – Random Forest pipeline
- **joblib** – Model serialization
- **Vanilla CSS** – Premium glassmorphism UI (no frameworks)
