"""
Tests d'intégration end-to-end : vrais modèles (outputs/) + vraies données (data/raw_data.csv).
Aucun mock. Skippés automatiquement en CI (fichiers absents).
"""
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api"))

_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
_RAW_CSV     = Path(__file__).parent.parent / "data" / "raw_data.csv"


def _pick_test_date() -> str:
    """Date 14 jours avant la fin du dataset — données complètes garanties."""
    df = pd.read_csv(_RAW_CSV, sep=";", encoding="utf-8-sig",
                     usecols=["Date et Heure"], low_memory=False)
    last = pd.to_datetime(df["Date et Heure"], utc=True).max()
    return (last - pd.Timedelta(days=14)).normalize().strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def client_real():
    if not _OUTPUTS_DIR.exists():
        pytest.skip("outputs/ absent")
    if not _RAW_CSV.exists():
        pytest.skip("data/raw_data.csv absent")

    import main as api_main

    # Vider les caches module — évite d'hériter d'un état laissé par les fixtures mock
    api_main._artifacts         = None
    api_main._raw_cache         = None
    api_main._raw_weather_cache = None
    api_main._weather_cache     = None
    api_main._features_cache    = None
    # Le lifespan de test_api.py a shutdown l'executor — on le recrée
    api_main._predict_executor  = ThreadPoolExecutor(max_workers=3)

    # weather_forecast.csv est un fichier opérationnel écrit par l'ETL — absent hors prod.
    # Seul mock nécessaire : météo future pour /predict (les autres endpoints lisent raw_data.csv)
    now = pd.Timestamp.now(tz="UTC").floor("30min")
    _fake_forecast = pd.DataFrame(
        {"temperature_2m": 15.0, "apparent_temperature": 14.0,
         "precipitation": 0.0, "cloud_cover": 30.0},
        index=pd.date_range(start=now, periods=200, freq="30min", tz="UTC"),
    )

    with patch("main._get_weather", return_value=_fake_forecast):
        with TestClient(api_main.app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture(scope="module")
def test_date():
    if not _RAW_CSV.exists():
        pytest.skip("data/raw_data.csv absent")
    return _pick_test_date()


@pytest.fixture(scope="module")
def predictions(client_real, test_date):
    end = (pd.Timestamp(test_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    r = client_real.get(f"/predict/range?start={test_date}&end={end}")
    assert r.status_code == 200, f"API erreur : {r.text}"
    return r.json()


@pytest.fixture(scope="module")
def actuals(test_date):
    df = pd.read_csv(_RAW_CSV, sep=";", encoding="utf-8-sig",
                     usecols=["Date et Heure", "Consommation (MW)"], low_memory=False)
    df["ts"] = pd.to_datetime(df["Date et Heure"], utc=True)
    df["mw"] = pd.to_numeric(df["Consommation (MW)"], errors="coerce")
    start = pd.Timestamp(test_date, tz="Europe/Paris").tz_convert("UTC")
    end   = start + pd.Timedelta(days=1)
    return df[(df["ts"] >= start) & (df["ts"] < end)].dropna(subset=["mw"]).set_index("ts")["mw"]


@pytest.fixture(scope="module")
def predict_response(client_real):
    r = client_real.get("/predict")
    assert r.status_code == 200
    return r.json()


class TestPredict:
    """Route /predict — 48 slots suivants calculés sur données et modèles réels."""

    def test_retourne_48_slots(self, predict_response):
        assert len(predict_response) == 48

    def test_structure_slot(self, predict_response):
        slot = predict_response[0]
        assert {"datetime", "prediction_mw", "prediction_p10", "prediction_p90"} <= slot.keys()

    def test_valeurs_realistes(self, predict_response):
        for slot in predict_response:
            assert 20_000 <= slot["prediction_mw"] <= 120_000

    def test_arrondi_25mw(self, predict_response):
        for slot in predict_response:
            assert slot["prediction_mw"] % 25 == 0


class TestModelInfo:
    """Route /model/info — métadonnées du modèle actif."""

    def test_structure(self, client_real):
        r = client_real.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert {"model_name", "models_available", "feature_cols"} <= body.keys()

    def test_nom_modele_reel(self, client_real):
        body = client_real.get("/model/info").json()
        assert "xgboost" in body["model_name"].lower()

    def test_26_features(self, client_real):
        body = client_real.get("/model/info").json()
        assert len(body["feature_cols"]) == 26


class TestPredictDonneesReelles:
    def test_retourne_48_slots(self, predictions):
        assert isinstance(predictions, list)
        assert len(predictions) == 48

    def test_structure_slot(self, predictions):
        slot = predictions[0]
        assert {"datetime", "prediction_mw", "prediction_p10", "prediction_p90"} <= slot.keys()

    def test_valeurs_realistes(self, predictions):
        for slot in predictions:
            mw = slot["prediction_mw"]
            assert 20_000 <= mw <= 120_000, f"Valeur hors plage : {mw} MW"

    def test_arrondi_25mw(self, predictions):
        for slot in predictions:
            assert slot["prediction_mw"] % 25 == 0

    def test_intervalles_ordonnes(self, predictions):
        # p10 ≤ p90 est garanti — p10 ≤ central ≤ p90 ne l'est pas
        # (3 modèles XGBoost indépendants, pas de contrainte d'ordre entre eux)
        for slot in predictions:
            assert slot["prediction_p10"] <= slot["prediction_p90"]

    def test_rmse_vs_reels(self, predictions, actuals, test_date):
        pred = pd.Series(
            {pd.Timestamp(s["datetime"]): s["prediction_mw"] for s in predictions}
        )
        merged = pred.reindex(actuals.index).dropna()
        if len(merged) < 10:
            pytest.skip(f"Moins de 10 slots communs pour {test_date}")
        residuals = merged - actuals.reindex(merged.index)
        rmse = float(np.sqrt((residuals ** 2).mean()))
        bias = float(residuals.mean())
        print(f"\nDate test : {test_date} | {len(merged)} slots")
        print(f"RMSE = {rmse:.0f} MW | Biais = {bias:+.0f} MW")
        # Seuil volontairement large : détecte un échec catastrophique, pas une anomalie saisonnière
        assert rmse < 5_000, f"RMSE anormalement élevé : {rmse:.0f} MW"
        assert abs(bias) < 3_000, f"Biais anormalement élevé : {bias:+.0f} MW"

    def test_health_modele_reel(self, client_real):
        r = client_real.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "xgboost" in r.json()["model_loaded"].lower()
