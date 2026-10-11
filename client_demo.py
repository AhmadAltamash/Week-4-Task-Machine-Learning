"""Tiny client showing how another program would call the API.
Start the server first (python app.py), then run:  python client_demo.py"""
import sys
import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"

employee = {"Age": 34, "Experience_Years": 9, "Monthly_Income": 22000,
            "Satisfaction_Score": 1.8, "Weekly_Work_Hours": 52,
            "Education": "Bachelor", "Department": "Sales",
            "City": "Patna", "Job_Level": "Mid"}

print("health:", requests.get(f"{URL}/health", timeout=10).json())
r = requests.post(f"{URL}/predict", json=employee, timeout=10)
print("status:", r.status_code)
result = r.json()
print(f"prediction: {result['prediction']}  "
      f"(probability {result['probability_attrition']:.1%}, risk {result['risk_band']})")
for f in result["top_factors"]:
    print(f"  - {f['feature']}: {f['effect']}")
