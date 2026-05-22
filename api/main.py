import logging
import os
import traceback
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from utils import load_latest_model
from transform import prepare_features, RAW_DIR, _DATETIME_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "../outputs"))

_conso_cache: pd.Series | None = None
_conso_mtime: float = 0.0
_conso_lock = threading.Lock()

_artifacts: dict | None = None
_artifacts_mtime: float = 0.0
_artifacts_lock = threading.Lock()


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
    global _artifacts, _artifacts_mtime
    mtime = MODELS_DIR.stat().st_mtime
    if _artifacts is not None and mtime == _artifacts_mtime:
        return _artifacts
    with _artifacts_lock:
        if _artifacts is None or mtime != _artifacts_mtime:
            model, model_name = load_latest_model(str(MODELS_DIR))
            _artifacts = {
                "model":        model,
                "model_name":   model_name,
                "imputer":      joblib.load(MODELS_DIR / "imputer.pkl"),
                "feature_cols": joblib.load(MODELS_DIR / "feature_cols.pkl"),
            }
            _artifacts_mtime = mtime
            log.info(f"Artefacts rechargés : {model_name}")
    return _artifacts


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

        conso = _get_conso()
        a = _get_artifacts()

        df_slots = pd.DataFrame({_DATETIME_COL: slots})
        df_slots = prepare_features(df_slots)
        df_slots["conso_h24"]  = conso.reindex(slots - pd.Timedelta(hours=24)).values
        df_slots["conso_h168"] = conso.reindex(slots - pd.Timedelta(hours=168)).values

        X = df_slots.reindex(columns=a["feature_cols"]).values.astype(float)
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
