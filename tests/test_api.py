"""API tests. Run with:  python -m unittest -v   (or: pytest)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import create_app  # noqa: E402

GOOD = {"Age": 34, "Experience_Years": 9, "Monthly_Income": 22000,
        "Satisfaction_Score": 2.0, "Weekly_Work_Hours": 52,
        "Education": "Bachelor", "Department": "Sales", "City": "Patna", "Job_Level": "Mid"}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "healthy")
        self.assertTrue(r.get_json()["model_loaded"])

    def test_home_page_loads(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Employee Attrition Predictor", r.data)

    def test_model_info(self):
        r = self.client.get("/model-info")
        self.assertEqual(r.status_code, 200)
        self.assertIn("test_metrics", r.get_json())

    def test_valid_prediction(self):
        r = self.client.post("/predict", json=GOOD)
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertIn(j["prediction"], ("Yes", "No"))
        self.assertTrue(0 <= j["probability_attrition"] <= 1)
        self.assertIn(j["risk_band"], ("low", "medium", "high"))
        self.assertEqual(len(j["top_factors"]), 3)

    def test_low_satisfaction_raises_risk(self):
        low = self.client.post("/predict", json={**GOOD, "Satisfaction_Score": 1.0}).get_json()
        high = self.client.post("/predict", json={**GOOD, "Satisfaction_Score": 5.0}).get_json()
        self.assertGreater(low["probability_attrition"], high["probability_attrition"])

    def test_missing_field(self):
        bad = {k: v for k, v in GOOD.items() if k != "Age"}
        r = self.client.post("/predict", json=bad)
        self.assertEqual(r.status_code, 400)
        self.assertIn("Age", r.get_json()["details"])

    def test_wrong_type(self):
        r = self.client.post("/predict", json={**GOOD, "Age": "thirty"})
        self.assertEqual(r.status_code, 400)

    def test_out_of_limits(self):
        r = self.client.post("/predict", json={**GOOD, "Satisfaction_Score": 9})
        self.assertEqual(r.status_code, 400)

    def test_unknown_category(self):
        r = self.client.post("/predict", json={**GOOD, "City": "Atlantis"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("City", r.get_json()["details"])

    def test_not_json(self):
        r = self.client.post("/predict", data="hello", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_outside_training_range_gives_warning(self):
        r = self.client.post("/predict", json={**GOOD, "Monthly_Income": 90000})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["warnings"])

    def test_unknown_route_and_method(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/predict").status_code, 405)


if __name__ == "__main__":
    unittest.main()
