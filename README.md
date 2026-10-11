# Employee Attrition Prediction API (Week 4 capstone)

An end-to-end machine learning project: data preprocessing, model comparison and tuning, evaluation, model serialization with joblib, and deployment as a Flask prediction API with a small web form.

> **The data is synthetic** (650 generated rows, no real employees) and the model is for demonstration only. It must not be used for hiring, firing, promotion or any other employment decision.

## Results
| | |
|---|---|
| Best model | Logistic Regression (highest cross-validated ROC-AUC of six models, 0.71) |
| Test accuracy | 0.677 (baseline: 0.515) |
| Test ROC-AUC | 0.702, 95% bootstrap interval 0.61 to 0.80 |
| Precision / recall / F1 | 0.676 / 0.716 / 0.696 (F1 of "predict everyone leaves" is 0.680, so F1 alone is not very informative) |

Strongest signal: satisfaction score. Attrition is 80% for scores 1-2 and 21% for scores 4-5.

## Project structure
```
employee-attrition-api/
  app.py                  Flask API (/, /health, /model-info, /predict)
  train_model.py          full training + evaluation pipeline, saves the model
  client_demo.py          example client using requests
  week4_capstone.ipynb    notebook version with outputs, ends with an API call
  templates/index.html    web form
  model/                  attrition_model.joblib + model_card.json
  data/                   employee_attrition.csv
  results/                metric tables (CSV)
  figures/                plots used in the report
  tests/test_api.py       12 API tests
  Dockerfile, Procfile    deployment files
  Week4_Report.docx       written report
  (Week4 slides: download from the presentation link and add here if your reviewer wants them in the repo)
```

## Setup
```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Train (optional, a trained model is included)
```bash
python train_model.py
```
Re-creates `model/`, `results/` and `figures/`.

## 2. Run the API
```bash
python app.py
```
Open http://127.0.0.1:5000 for the web form, or call the API directly.

```bash
curl http://127.0.0.1:5000/health

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "Age": 34, "Experience_Years": 9, "Monthly_Income": 22000,
  "Satisfaction_Score": 1.8, "Weekly_Work_Hours": 52,
  "Education": "Bachelor", "Department": "Sales", "City": "Patna", "Job_Level": "Mid"}'
```
PowerShell: `Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body (Get-Content body.json -Raw)`

Example response:
```json
{
  "prediction": "Yes",
  "probability_attrition": 0.705,
  "risk_band": "high",
  "top_factors": [
    {"feature": "Satisfaction_Score", "effect": "raises risk", "contribution": 0.544},
    {"feature": "Weekly_Work_Hours", "effect": "raises risk", "contribution": 0.251},
    {"feature": "Department", "effect": "raises risk", "contribution": 0.062}
  ],
  "warnings": [],
  "note": "Educational demonstration on synthetic data. Not for employment decisions."
}
```
Or run `python client_demo.py` while the server is running.

## API
| endpoint | method | purpose |
|---|---|---|
| `/` | GET | web form |
| `/health` | GET | service and model status |
| `/model-info` | GET | model card: metrics, features, allowed values |
| `/predict` | POST | JSON with all 9 fields, returns prediction, probability, risk band, top factors |

Invalid input (missing field, wrong type, value outside hard limits, unknown category, not JSON) returns **HTTP 400** with a message per field. Values that are valid but outside the range seen in training return a prediction plus a warning.

Risk bands: low below 0.40, medium 0.40 to 0.60, high 0.60 and above.

## Tests
```bash
python -m unittest -v
```

## Deploy
**Gunicorn (production server):** `gunicorn app:app`

**Docker:**
```bash
docker build -t attrition-api .
docker run -p 8000:8000 attrition-api
```
then open http://localhost:8000.

**Render / similar platforms:** create a Web Service from the GitHub repo, build command `pip install -r requirements.txt`, start command `gunicorn app:app`.

## Notes and limitations
- Verified locally (Flask server, curl, `client_demo.py`, 12 passing tests). The Docker image and a public hosting service were not built or tested.
- Train and serve with the same scikit-learn version; the app logs a warning if they differ.
- Joblib files can run code when loaded, so only load model files you trust.
- A real system would need real data, privacy and security review, fairness checks (features such as age and city should be reviewed before any real use), monitoring for drift, and human oversight.
