import logging
import os
import sys
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils import load_latest_model, load_quantile_models, list_models
from transform import prepare_features, add_lags, _WEATHER_COLS

try:
    from google.cloud.logging.handlers import StructuredLogHandler
    logging.basicConfig(handlers=[StructuredLogHandler()], level=logging.INFO)
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

_ROOT      = Path(__file__).parent.parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_ROOT / "outputs")))
RAW_DIR    = Path(os.getenv("RAW_DIR",    str(_ROOT / "data")))

_artifacts: dict | None = None
_artifacts_lock = threading.Lock()

_weather_cache: pd.DataFrame | None = None
_weather_mtime: float | None = None
_weather_lock = threading.Lock()

_raw_cache: tuple[pd.Series, pd.Series] | None = None
_raw_mtime: float | None = None
_raw_lock = threading.Lock()

_raw_weather_cache: pd.DataFrame | None = None
_raw_weather_mtime: float | None = None
_raw_weather_lock = threading.Lock()

# Cache de la matrice de features : invalide quand l'ETL tourne (mtime) ou quand le slot de 30min avance
_features_cache: dict | None = None
_features_lock = threading.Lock()

_predict_executor = ThreadPoolExecutor(max_workers=3)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _get_artifacts()
        _get_raw_history()
        _get_raw_weather()
        _get_weather()
        log.info("Pre-warm terminé")
    except Exception as e:
        log.warning(f"Pre-warm partiel : {e}")
    yield
    _predict_executor.shutdown(wait=False)


class SlotPrediction(BaseModel):
    datetime: str
    prediction_mw: int
    prediction_p10: int
    prediction_p90: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: str

class ModelInfoResponse(BaseModel):
    model_name: str
    models_available: list[str]
    feature_cols: list[str]


app = FastAPI(
    title="API Prévision Consommation Électrique",
    description=(
        "Prédictions de consommation électrique nationale (France) "
        "sur les 24h suivantes, avec intervalles de confiance à 80% (CQR)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    redoc_url=None,
)


def _get_artifacts() -> dict:
    global _artifacts
    # listdir uniquement pour comparer le nom — pas de désérialisation pickle
    models = list_models(str(MODELS_DIR))
    if not models:
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier.")
    latest_name = models[0]
    if _artifacts is not None and _artifacts["model_name"] == latest_name:
        return _artifacts
    with _artifacts_lock:
        if _artifacts is None or _artifacts["model_name"] != latest_name:
            model, model_name = load_latest_model(str(MODELS_DIR))
            p10_model, p90_model = load_quantile_models(str(MODELS_DIR))
            _artifacts = {
                "model":          model,
                "model_name":     model_name,
                "p10_model":      p10_model,
                "p90_model":      p90_model,
                "imputer":        joblib.load(MODELS_DIR / "imputer.pkl"),
                "feature_cols":   joblib.load(MODELS_DIR / "feature_cols.pkl"),
                "cqr_correction": joblib.load(MODELS_DIR / "cqr_correction.pkl"),
            }
            log.info(f"Artefacts rechargés : {model_name}")
    return _artifacts


def _get_raw_history() -> tuple[pd.Series, pd.Series]:
    """Retourne (conso_hist, temp_hist) indexées par datetimes UTC pour les lags.
    Cache invalidé par mtime — rechargé exactement quand l'ETL modifie le fichier."""
    global _raw_cache, _raw_mtime
    raw_path = RAW_DIR / "raw_data.csv"
    current_mtime = raw_path.stat().st_mtime
    if _raw_cache is not None and _raw_mtime == current_mtime:
        return _raw_cache
    with _raw_lock:
        current_mtime = raw_path.stat().st_mtime
        if _raw_cache is None or _raw_mtime != current_mtime:
            df = pd.read_csv(
                raw_path,
                sep=";", encoding="utf-8-sig",
                usecols=["Date et Heure", "Consommation (MW)", "temperature_2m"],
                low_memory=False,
            )
            dt    = pd.to_datetime(df["Date et Heure"], utc=True)
            conso = pd.Series(pd.to_numeric(df["Consommation (MW)"], errors="coerce").values, index=dt)
            temp  = pd.Series(pd.to_numeric(df["temperature_2m"],    errors="coerce").values, index=dt)
            _raw_cache = (conso, temp)
            _raw_mtime = current_mtime
            log.info(f"raw_data.csv rechargé : {len(dt)} slots")
    return _raw_cache


def _get_weather() -> pd.DataFrame:
    """Lit weather_forecast.csv écrit par l'ETL. Cache invalidé par mtime."""
    global _weather_cache, _weather_mtime
    weather_path = RAW_DIR / "weather_forecast.csv"
    current_mtime = weather_path.stat().st_mtime
    if _weather_cache is not None and _weather_mtime == current_mtime:
        return _weather_cache
    with _weather_lock:
        current_mtime = weather_path.stat().st_mtime
        if _weather_cache is None or _weather_mtime != current_mtime:
            df = pd.read_csv(weather_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            _weather_cache = df
            _weather_mtime = current_mtime
            log.info(f"weather_forecast.csv rechargé : {len(df)} slots")
    return _weather_cache


def _get_raw_weather() -> pd.DataFrame:
    """Colonnes météo de raw_data.csv, indexées en UTC. Cache invalidé par mtime."""
    global _raw_weather_cache, _raw_weather_mtime
    raw_path = RAW_DIR / "raw_data.csv"
    current_mtime = raw_path.stat().st_mtime
    if _raw_weather_cache is not None and _raw_weather_mtime == current_mtime:
        return _raw_weather_cache
    with _raw_weather_lock:
        current_mtime = raw_path.stat().st_mtime
        if _raw_weather_cache is None or _raw_weather_mtime != current_mtime:
            df = pd.read_csv(
                raw_path, sep=";", encoding="utf-8-sig",
                usecols=["Date et Heure"] + list(_WEATHER_COLS), low_memory=False,
            )
            df.index = pd.to_datetime(df["Date et Heure"], utc=True)
            df = df[list(_WEATHER_COLS)]
            df = df[~df.index.duplicated(keep="first")]
            _raw_weather_cache = df
            _raw_weather_mtime = current_mtime
            log.info(f"raw_weather rechargé : {len(df)} slots")
    return _raw_weather_cache


def _get_combined_weather(slots: pd.DatetimeIndex) -> pd.DataFrame:
    """Météo pour des slots arbitraires : raw_data (passé) + forecast (futur proche) + NaN au-delà."""
    raw_weather  = _get_raw_weather()
    try:
        forecast = _get_weather()
    except Exception:
        forecast = pd.DataFrame(index=pd.DatetimeIndex([]), columns=list(_WEATHER_COLS))

    combined = raw_weather.reindex(slots).copy()
    for col in _WEATHER_COLS:
        if col in forecast.columns:
            combined[col] = combined[col].fillna(forecast[col].reindex(slots))
    return combined


def _get_features(start: datetime) -> tuple:
    """Retourne (X_imputed, slots) cachés par (weather_mtime, raw_mtime, start).
    Invalide automatiquement à chaque run ETL (mtime change) ou nouveau slot de 30min."""
    global _features_cache
    key = (_weather_mtime, _raw_mtime, start.isoformat())
    with _features_lock:
        if _features_cache is not None and _features_cache["key"] == key:
            return _features_cache["X"], _features_cache["slots"]

        weather               = _get_weather()
        a                     = _get_artifacts()
        conso_hist, temp_hist = _get_raw_history()

        slots = pd.date_range(start, periods=48, freq="30min")
        df_slots = pd.DataFrame(
            {"Date et Heure": slots, **{col: weather[col].reindex(slots).values for col in _WEATHER_COLS}}
        )
        df = prepare_features(df_slots)
        df = add_lags(df, conso_hist=conso_hist, temp_hist=temp_hist)
        X = df.reindex(columns=a["feature_cols"]).values.astype(float)
        X = a["imputer"].transform(X)

        _features_cache = {"key": key, "X": X, "slots": slots}
        log.info(f"Features recalculées pour slot {start.isoformat()}")
        return X, slots


@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    log.error(f"Erreur non gérée : {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérifie que l'API est opérationnelle et qu'un modèle est chargé."""
    a = _get_artifacts()
    return {"status": "ok", "model_loaded": a["model_name"]}


@app.get("/predict", response_model=list[SlotPrediction], tags=["Prévisions"])
def predict():
    """
    Retourne les prévisions de consommation pour les 48 prochains slots de 30 minutes
    (24h), à partir du prochain slot entier après l'heure courante UTC.

    - **prediction_mw** : prévision centrale (arrondie à 25 MW)
    - **prediction_p10 / prediction_p90** : borne basse/haute de l'intervalle de confiance à 80% (CQR)
    """
    try:
        now = datetime.now(timezone.utc)
        start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        a        = _get_artifacts()
        X, slots = _get_features(start)
        q        = a["cqr_correction"]

        f_pred = _predict_executor.submit(a["model"].predict,     X)
        f_p10  = _predict_executor.submit(a["p10_model"].predict, X)
        f_p90  = _predict_executor.submit(a["p90_model"].predict, X)
        preds     = f_pred.result()
        preds_p10 = f_p10.result() - q
        preds_p90 = f_p90.result() + q

        log.info(f"Prédiction de {len(slots)} slots depuis {start.isoformat()}")
        return [
            {
                "datetime":       s.isoformat(),
                "prediction_mw":  round(float(p)   / 25) * 25,
                "prediction_p10": round(float(p10) / 25) * 25,
                "prediction_p90": round(float(p90) / 25) * 25,
            }
            for s, p, p10, p90 in zip(slots, preds, preds_p10, preds_p90)
        ]

    except Exception as e:
        log.error(f"Erreur predict : {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/range", response_model=list[SlotPrediction], tags=["Prévisions"])
def predict_range(start: str, end: str):
    """
    Prédit tous les slots de 30 min entre **start** et **end** (max 14 jours d'écart).

    - **start / end** : ISO 8601 — timezone Europe/Paris si absente (ex: `2026-06-05` ou `2026-06-05T00:00:00+02:00`)
    - Jusqu'à **J+14** : la météo Open-Meteo couvre 14 jours, les lags de conso manquants sont imputés automatiquement.
    """
    def _parse(s: str) -> pd.Timestamp:
        try:
            ts = pd.Timestamp(s)
            if ts.tzinfo is None:
                ts = ts.tz_localize("Europe/Paris")
            return ts.tz_convert("UTC")
        except Exception:
            raise HTTPException(status_code=422, detail=f"Date invalide : '{s}'. Utiliser ISO 8601.")

    start_dt = _parse(start)
    end_dt   = _parse(end)

    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="end doit être postérieur à start.")
    if (end_dt - start_dt) > pd.Timedelta(days=14):
        raise HTTPException(status_code=422, detail="Écart maximum : 14 jours.")
    now = pd.Timestamp.now(tz="UTC")
    if end_dt > now + pd.Timedelta(days=14):
        raise HTTPException(status_code=422, detail="end ne peut pas dépasser J+14 (horizon météo Open-Meteo).")

    try:
        conso_hist, temp_hist = _get_raw_history()
        dataset_start = conso_hist.index.min()
        if start_dt < dataset_start:
            raise HTTPException(
                status_code=422,
                detail=f"start ne peut pas être avant le début du dataset ({dataset_start.date()})."
            )

        slots = pd.date_range(start_dt, end_dt, freq="30min", inclusive="left")
        if len(slots) == 0:
            return []

        a = _get_artifacts()
        weather               = _get_combined_weather(slots)
        q                     = a["cqr_correction"]

        df_slots = pd.DataFrame(
            {"Date et Heure": slots, **{col: weather[col].values for col in _WEATHER_COLS}}
        )
        df = prepare_features(df_slots)
        df = add_lags(df, conso_hist=conso_hist, temp_hist=temp_hist)
        X  = df.reindex(columns=a["feature_cols"]).values.astype(float)
        X  = a["imputer"].transform(X)

        f_pred = _predict_executor.submit(a["model"].predict,     X)
        f_p10  = _predict_executor.submit(a["p10_model"].predict, X)
        f_p90  = _predict_executor.submit(a["p90_model"].predict, X)
        preds     = f_pred.result()
        preds_p10 = f_p10.result() - q
        preds_p90 = f_p90.result() + q

        log.info(f"predict_range : {len(slots)} slots ({start_dt.date()} → {end_dt.date()})")
        return [
            {
                "datetime":       s.isoformat(),
                "prediction_mw":  round(float(p)   / 25) * 25,
                "prediction_p10": round(float(p10) / 25) * 25,
                "prediction_p90": round(float(p90) / 25) * 25,
            }
            for s, p, p10, p90 in zip(slots, preds, preds_p10, preds_p90)
        ]
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Erreur predict_range : {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Retourne le nom du modèle actif, les modèles disponibles et les features utilisées."""
    a = _get_artifacts()
    return {
        "model_name":       a["model_name"],
        "models_available": list_models(str(MODELS_DIR)),
        "feature_cols":     a["feature_cols"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
