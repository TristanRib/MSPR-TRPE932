import logging
import os
import re
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT          = Path(__file__).parent.parent
RAW_DIR       = Path(os.getenv("RAW_DIR",       str(ROOT / "data")))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT / "data")))
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    str(ROOT / "outputs")))

_DATETIME_COL = "date_heure"
_CONSO_COL    = "Consommation"
_WEATHER_COLS = ["temperature_2m", "apparent_temperature", "precipitation", "cloud_cover"]


def add_lags(
    df: pd.DataFrame,
    conso_hist: pd.Series | None = None,
    temp_hist: pd.Series | None = None,
) -> pd.DataFrame:
    """Ajoute conso_h24, conso_h48, conso_h168, temp_h24, temp_h48.
    En mode entraînement (historiques None) : lookup dans df lui-même (index UTC).
    En mode API (historiques fournis) : lookup dans les séries historiques indexées UTC."""
    df = df.copy()
    conso = conso_hist if conso_hist is not None else df[_CONSO_COL]
    temp  = temp_hist  if temp_hist  is not None else df["temperature_2m"]
    df["conso_h24"]  = conso.reindex(df.index - pd.Timedelta(hours=24)).values
    df["conso_h48"]  = conso.reindex(df.index - pd.Timedelta(hours=48)).values
    df["conso_h168"] = conso.reindex(df.index - pd.Timedelta(hours=168)).values
    df["temp_h24"]   = temp.reindex(df.index - pd.Timedelta(hours=24)).values
    df["temp_h48"]   = temp.reindex(df.index - pd.Timedelta(hours=48)).values
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [re.sub(r"\s*\(.*?\)\s*$", "", col).strip() for col in df.columns]
    df = df.rename(columns={"Date et Heure": _DATETIME_COL})

    dt     = pd.to_datetime(df[_DATETIME_COL], utc=True)
    dt_par = dt.dt.tz_convert("Europe/Paris")
    df = df.drop(columns=[_DATETIME_COL])

    hour_frac = dt_par.dt.hour + dt_par.dt.minute / 60
    df["hour_sin"]  = np.sin(2 * np.pi * hour_frac            / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour_frac            / 24)
    df["day_sin"]   = np.sin(2 * np.pi * dt_par.dt.dayofweek  / 7)
    df["day_cos"]   = np.cos(2 * np.pi * dt_par.dt.dayofweek  / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt_par.dt.month      / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt_par.dt.month      / 12)

    df.index = dt

    fr_holidays = holidays.France()
    df["is_holiday"] = dt_par.dt.date.map(lambda d: int(d in fr_holidays)).values

    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].replace("ND", np.nan), errors="coerce")

    df["heating_degrees"] = (17 - df["temperature_2m"]).clip(lower=0)
    df["cooling_degrees"] = (df["temperature_2m"] - 21).clip(lower=0)

    return df


def main():
    df = pd.read_csv(
        RAW_DIR / "raw_data.csv",
        sep=";",
        encoding="utf-8-sig",
        index_col=False,
        low_memory=False,
    )
    df = df.sort_values(by=["Date et Heure"], ascending=True).reset_index(drop=True)
    df = prepare_features(df)
    df = df[~df.index.duplicated(keep="first")]
    df = add_lags(df)
    df = df.dropna(subset=[_CONSO_COL, "conso_h24", "conso_h48", "conso_h168", "temp_h24", "temp_h48"])

    feature_cols = [
        "hour_sin", "hour_cos",
        "day_sin",  "day_cos",
        "month_sin", "month_cos",
        "temperature_2m", "apparent_temperature",
        "precipitation", "cloud_cover",
        "heating_degrees", "cooling_degrees",
        "is_holiday", "conso_h24", "conso_h48", "conso_h168", "temp_h24", "temp_h48",
    ]

    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    iso_data = df[feature_cols + [_CONSO_COL]].values
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(iso_data)
    train_scores = iso.decision_function(iso_data)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(imputer,      MODELS_DIR / "imputer.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    joblib.dump({
        "model":     iso,
        "score_min": float(train_scores.min()),
        "score_max": float(train_scores.max()),
    }, MODELS_DIR / "isolation_forest.pkl")
    log.info(f"imputer.pkl + isolation_forest.pkl sauvegardés ({len(feature_cols)} features)")

    PROCESSED_DIR.mkdir(exist_ok=True)
    df[feature_cols + [_CONSO_COL]].to_csv(
        PROCESSED_DIR / "transformed_data.csv", index=False
    )
    log.info(f"transformed_data.csv sauvegardé ({len(df)} lignes)")


if __name__ == "__main__":
    main()
