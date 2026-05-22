import logging
import os
import traceback
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

try:
    model, model_name = load_latest_model(str(MODELS_DIR))
    imputer      = joblib.load(MODELS_DIR / "imputer.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    log.info(f"Modèle chargé : {model_name} ({len(feature_cols)} features)")
except Exception as e:
    log.error(f"Erreur au chargement des artefacts : {e}")
    raise


@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    log.error(f"Erreur non gérée : {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_name}


@app.get("/predict")
def predict():
    try:
        # Prochain slot de 30 min à partir de maintenant
        now = datetime.now(timezone.utc)
        remainder = now.minute % 30
        delta = timedelta(minutes=(30 - remainder) % 30)
        start = (now + delta).replace(second=0, microsecond=0)
        slots = pd.date_range(start, periods=48, freq="30min")

        # Lecture de raw_data pour les lags (conso_h24 et conso_h168)
        raw = pd.read_csv(RAW_DIR / "raw_data.csv", sep=";", encoding="utf-8-sig", low_memory=False)
        raw[_DATETIME_COL] = pd.to_datetime(raw["Date et Heure"], utc=True)
        conso = pd.to_numeric(
            raw.set_index(_DATETIME_COL)["Consommation (MW)"].replace("ND", np.nan),
            errors="coerce",
        )
        conso = conso[~conso.index.duplicated(keep="first")]

        # Features temporelles pour les 48 slots
        df_slots = pd.DataFrame({_DATETIME_COL: slots})
        df_slots = prepare_features(df_slots)

        df_slots["conso_h24"]  = conso.reindex(slots - pd.Timedelta(hours=24)).values
        df_slots["conso_h168"] = conso.reindex(slots - pd.Timedelta(hours=168)).values

        X = df_slots.reindex(columns=feature_cols).values.astype(float)
        X = imputer.transform(X)
        preds = model.predict(X)

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
    return {
        "model_name": model_name,
        "models_available": os.listdir(str(MODELS_DIR)),
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)