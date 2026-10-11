"""
Employee Attrition Prediction API (Flask)

Endpoints
  GET  /            small web form for trying the model in a browser
  GET  /health      is the service up and is the model loaded?
  GET  /model-info  the model card (metrics, features, allowed values)
  POST /predict     JSON in, prediction + probability + main factors out

Run locally:   python app.py
Production:    gunicorn app:app
"""

import os
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, request, jsonify, render_template

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "attrition_model.joblib"
CARD_PATH = BASE / "model" / "model_card.json"

NUMERIC = ["Age", "Experience_Years", "Monthly_Income",
           "Satisfaction_Score", "Weekly_Work_Hours"]
CATEGORICAL = ["Education", "Department", "City", "Job_Level"]
FIELDS = NUMERIC + CATEGORICAL

# Hard limits: values outside these are rejected as nonsense (HTTP 400).
HARD_LIMITS = {"Age": (16, 80), "Experience_Years": (0, 60),
               "Monthly_Income": (0, 1_000_000),
               "Satisfaction_Score": (1, 5), "Weekly_Work_Hours": (0, 100)}

NOTE = "Educational demonstration on synthetic data. Not for employment decisions."


def risk_band(p):
    if p < 0.40:
        return "low"
    if p < 0.60:
        return "medium"
    return "high"


def validate(data, card):
    """Return (clean_row, errors, warnings). Nothing is guessed or silently fixed."""
    errors, warnings = {}, []
    if not isinstance(data, dict):
        return None, {"body": "Send a JSON object."}, warnings

    missing = [f for f in FIELDS if f not in data or data[f] is None]
    for f in missing:
        errors[f] = "required"

    row = {}
    for f in NUMERIC:
        if f in missing:
            continue
        v = data[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors[f] = "must be a number"
            continue
        lo, hi = HARD_LIMITS[f]
        if not (lo <= v <= hi):
            errors[f] = f"must be between {lo} and {hi}"
            continue
        row[f] = float(v)
        seen_lo, seen_hi = card["numeric_ranges_seen"][f]
        if not (seen_lo <= v <= seen_hi):
            warnings.append(f"{f}={v} is outside the range seen in training "
                            f"({seen_lo:g} to {seen_hi:g}); the prediction is less reliable.")

    for f in CATEGORICAL:
        if f in missing:
            continue
        allowed = card["allowed_categories"][f]
        if data[f] not in allowed:
            errors[f] = f"must be one of {allowed}"
            continue
        row[f] = data[f]

    return row, errors, warnings


def explain(model, frame, top_n=3):
    """Per-feature contribution to the log-odds of leaving (coefficient x value),
    with one-hot columns added back together under their original feature."""
    prep, clf = model.named_steps["prep"], model.named_steps["model"]
    z = prep.transform(frame)[0]
    names = prep.get_feature_names_out()
    contrib = {}
    for name, coef, val in zip(names, clf.coef_[0], z):
        if name.startswith("num__"):
            col = name[len("num__"):]
        else:
            rest = name[len("cat__"):]
            col = next(c for c in CATEGORICAL if rest.startswith(c + "_"))
        contrib[col] = contrib.get(col, 0.0) + float(coef * val)
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return [{"feature": k,
             "effect": "raises risk" if v > 0 else "lowers risk",
             "contribution": round(v, 3)} for k, v in ranked]


def create_app(model_path=MODEL_PATH, card_path=CARD_PATH):
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = joblib.load(model_path)
    with open(card_path) as f:
        card = json.load(f)
    if card.get("sklearn_version") != sklearn.__version__:
        app.logger.warning("Model was trained with scikit-learn %s but %s is installed.",
                           card.get("sklearn_version"), sklearn.__version__)
    app.logger.info("Model loaded from %s", model_path)

    @app.get("/")
    def home():
        return render_template("index.html", card=card, fields=FIELDS,
                               categories=card["allowed_categories"],
                               ranges=card["numeric_ranges_seen"])

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "model_loaded": model is not None,
                        "sklearn_version": sklearn.__version__})

    @app.get("/model-info")
    def model_info():
        return jsonify(card)

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        row, errors, warnings = validate(data, card)
        if errors:
            app.logger.info("Rejected request: %s", errors)
            return jsonify({"error": "Invalid input.", "details": errors}), 400
        frame = pd.DataFrame([row], columns=FIELDS)
        p = float(model.predict_proba(frame)[0, 1])
        body = {"prediction": "Yes" if p >= 0.5 else "No",
                "probability_attrition": round(p, 4),
                "risk_band": risk_band(p),
                "top_factors": explain(model, frame),
                "warnings": warnings,
                "note": NOTE}
        app.logger.info("Prediction p=%.3f band=%s", p, body["risk_band"])
        return jsonify(body)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def wrong_method(_):
        return jsonify({"error": "Method not allowed for this endpoint."}), 405

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
