import logging
import os
import sys
import traceback
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from utils import load_latest_model
from transform import prepare_features, RAW_DIR, _DATETIME_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "../outputs"))

_WEATHER_COLS   = ["temperature_2m", "apparent_temperature", "precipitation", "cloud_cover"]
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_TTL    = timedelta(hours=1)

_conso_cache: pd.Series | None = None
_conso_mtime: float = 0.0
_conso_lock = threading.Lock()

_artifacts: dict | None = None
_artifacts_lock = threading.Lock()

_weather_cache: pd.DataFrame | None = None
_weather_fetched_at: datetime | None = None
_weather_lock = threading.Lock()


def _get_conso() -> pd.Series:
    global _conso_cache, _conso_mtime
    raw_path = RAW_DIR / "raw_data.csv"
    mtime = raw_path.stat().st_mtime
    if _conso_cache is not None and mtime == _conso_mtime:
        return _conso_cache
    with _conso_lock:
        if _conso_cache is None or mtime != _conso_mtime:
            _raw = pd.read_csv(
                raw_path, sep=";", encoding="utf-8-sig",
                usecols=["Date et Heure", "Consommation (MW)"], low_memory=False,
            )
            _raw[_DATETIME_COL] = pd.to_datetime(_raw["Date et Heure"], utc=True)
            conso = pd.to_numeric(
                _raw.set_index(_DATETIME_COL)["Consommation (MW)"].replace("ND", np.nan),
                errors="coerce",
            )
            _conso_cache = conso[~conso.index.duplicated(keep="first")]
            _conso_mtime = mtime
            log.info(f"conso rechargé depuis raw_data.csv ({len(_conso_cache)} points)")
    return _conso_cache


def _get_artifacts() -> dict:
    global _artifacts
    _, latest_name = load_latest_model(str(MODELS_DIR))
    if _artifacts is not None and _artifacts["model_name"] == latest_name:
        return _artifacts
    with _artifacts_lock:
        if _artifacts is None or _artifacts["model_name"] != latest_name:
            model, model_name = load_latest_model(str(MODELS_DIR))
            _artifacts = {
                "model":        model,
                "model_name":   model_name,
                "imputer":      joblib.load(MODELS_DIR / "imputer.pkl"),
                "feature_cols": joblib.load(MODELS_DIR / "feature_cols.pkl"),
            }
            log.info(f"Artefacts rechargés : {model_name}")
    return _artifacts


def _get_weather() -> pd.DataFrame:
    global _weather_cache, _weather_fetched_at
    now = datetime.now(timezone.utc)
    if _weather_cache is not None and _weather_fetched_at is not None:
        if now - _weather_fetched_at < _WEATHER_TTL:
            return _weather_cache
    with _weather_lock:
        if _weather_cache is None or _weather_fetched_at is None or now - _weather_fetched_at >= _WEATHER_TTL:
            params = {
                "latitude":   46,
                "longitude":  2,
                "timezone":   "Europe/Paris",
                "hourly":     ",".join(_WEATHER_COLS),
                "past_days":  1,
                "forecast_days": 2,
            }
            resp = requests.get(_OPEN_METEO_URL, params=params, timeout=30)
            resp.raise_for_status()
            hourly = resp.json()["hourly"]

            df = pd.DataFrame(hourly)
            df["time"] = (
                pd.to_datetime(df["time"])
                .dt.tz_localize("Europe/Paris", ambiguous="infer")
                .dt.tz_convert("UTC")
            )
            df = df.set_index("time")
            df = df.resample("15min").interpolate("linear")
            _weather_cache = df
            _weather_fetched_at = now
            log.info(f"Météo rechargée : {len(df)} slots 15-min")
    return _weather_cache


# Chargement initial — plante au démarrage si les artefacts sont absents
_get_artifacts()
_get_conso()


@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    log.error(f"Erreur non gérée : {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health():
    a = _get_artifacts()
    return {"status": "ok", "model_loaded": a["model_name"]}


@app.get("/predict")
def predict():
    try:
        now = datetime.now(timezone.utc)
        remainder = now.minute % 30
        delta = timedelta(minutes=(30 - remainder) % 30)
        start = (now + delta).replace(second=0, microsecond=0)
        slots = pd.date_range(start, periods=48, freq="30min")

        conso   = _get_conso()
        weather = _get_weather()
        a       = _get_artifacts()

        df_slots = pd.DataFrame(
            {"Date et Heure": slots, **{col: weather[col].reindex(slots).values for col in _WEATHER_COLS}}
        )
        df = prepare_features(df_slots, conso_series=conso)

        X = df.reindex(columns=a["feature_cols"]).values.astype(float)
        X = a["imputer"].transform(X)
        preds = a["model"].predict(X)

        log.info(f"Prédiction de {len(slots)} slots depuis {start.isoformat()}")
        return [
            {"datetime": s.isoformat(), "prediction_mw": round(float(p), 1)}
            for s, p in zip(slots, preds)
        ]

    except Exception as e:
        log.error(f"Erreur predict : {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    a = _get_artifacts()
    return {
        "model_name":       a["model_name"],
        "models_available": os.listdir(str(MODELS_DIR)),
        "feature_cols":     a["feature_cols"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
