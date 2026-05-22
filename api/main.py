import logging
import os
import traceback
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils import load_latest_model
from transform import prepare_features

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


class PredictRequest(BaseModel):
    data: dict[str, Any]


@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    log.error(f"Erreur non gérée : {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_name}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame([request.data])
        log.info(f"Requête reçue : {list(request.data.keys())}")

        df = prepare_features(df)
        log.info(f"Features après transformation : {list(df.columns)}")

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            log.warning(f"{len(missing)} features manquantes (seront imputées) : {missing[:5]}...")

        X = df.reindex(columns=feature_cols).values.astype(float)
        X = imputer.transform(X)
        prediction = model.predict(X)

        result = float(prediction[0])
        log.info(f"Prédiction : {result:.1f} MW")
        return {"prediction": result, "model_used": model_name}

    except HTTPException:
        raise
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
