from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


_FAKE_CONSO = pd.Series(
    [40000.0] * 200,
    index=pd.date_range("2012-01-01", periods=200, freq="30min", tz="UTC"),
)
_FAKE_TEMP = pd.Series(
    [10.0] * 200,
    index=pd.date_range("2012-01-01", periods=200, freq="30min", tz="UTC"),
)
_FAKE_ARTIFACTS = {
    "model":          MagicMock(**{"predict.return_value": [40000.0] * 2000}),
    "model_name":     "model_xgboost_20260606_1.pkl",
    "p10_model":      MagicMock(**{"predict.return_value": [39000.0] * 2000}),
    "p90_model":      MagicMock(**{"predict.return_value": [41000.0] * 2000}),
    "imputer":        MagicMock(**{"transform.side_effect": lambda x: x}),
    "feature_cols":   ["hour_sin", "hour_cos"],
    "cqr_correction": 500.0,
}
_FAKE_WEATHER = pd.DataFrame(
    {"temperature_2m": 15.0, "apparent_temperature": 14.0,
     "precipitation": 0.0, "cloud_cover": 30.0},
    index=pd.date_range("2012-01-01", periods=200000, freq="30min", tz="UTC"),
)


@pytest.fixture(scope="module")
def client():
    with patch("main._get_artifacts",   return_value=_FAKE_ARTIFACTS), \
         patch("main._get_raw_history", return_value=(_FAKE_CONSO, _FAKE_TEMP)), \
         patch("main._get_raw_weather", return_value=_FAKE_WEATHER), \
         patch("main._get_weather",     return_value=_FAKE_WEATHER), \
         patch("main._get_combined_weather", return_value=_FAKE_WEATHER):
        import main as api_main
        with TestClient(api_main.app, raise_server_exceptions=False) as c:
            yield c


class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_shape(self, client):
        r = client.get("/health")
        assert "status" in r.json()
        assert "model_loaded" in r.json()


class TestPredictRangeValidation:
    def test_end_before_start(self, client):
        r = client.get("/predict/range?start=2026-06-10&end=2026-06-05")
        assert r.status_code == 422
        assert "postérieur" in r.json()["detail"]

    def test_end_equals_start(self, client):
        r = client.get("/predict/range?start=2026-06-05&end=2026-06-05")
        assert r.status_code == 422

    def test_gap_exceeds_14_days(self, client):
        r = client.get("/predict/range?start=2026-06-01&end=2026-06-20")
        assert r.status_code == 422
        assert "14 jours" in r.json()["detail"]

    def test_end_beyond_j14(self, client):
        far_future = (datetime.now(timezone.utc) + timedelta(days=20)).strftime("%Y-%m-%d")
        start      = (datetime.now(timezone.utc) + timedelta(days=8)).strftime("%Y-%m-%d")
        r = client.get(f"/predict/range?start={start}&end={far_future}")
        assert r.status_code == 422
        assert "J+14" in r.json()["detail"]

    def test_invalid_date_format(self, client):
        r = client.get("/predict/range?start=not-a-date&end=2026-06-10")
        assert r.status_code == 422
        assert "invalide" in r.json()["detail"]

    def test_before_dataset_start(self, client):
        r = client.get("/predict/range?start=2010-01-01&end=2010-01-03")
        assert r.status_code == 422
        assert "dataset" in r.json()["detail"]

    def test_gap_exactly_14_days_not_rejected(self, client):
        r = client.get("/predict/range?start=2026-06-01&end=2026-06-15")
        assert r.json().get("detail") != "Écart maximum : 14 jours."

    def test_gap_14_days_plus_one_rejected(self, client):
        r = client.get("/predict/range?start=2026-06-01&end=2026-06-16")
        assert r.status_code == 422
        assert "14 jours" in r.json()["detail"]
