import logging
import os
import re
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
load_dotenv()

import holidays
import joblib
import numpy as np
import pandas as pd

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
    """Ajoute lags conso/temp et moyennes glissantes.
    En mode entraînement (historiques None) : lookup dans df lui-même (index UTC).
    En mode API (historiques fournis) : lookup dans les séries historiques indexées UTC."""
    df = df.copy()
    idx   = cast(pd.DatetimeIndex, df.index)
    conso = conso_hist if conso_hist is not None else df[_CONSO_COL]
    temp  = temp_hist  if temp_hist  is not None else df["temperature_2m"]

    df["conso_h24"]    = conso.reindex(idx - pd.Timedelta(hours=24)).to_numpy()
    df["conso_h48"]    = conso.reindex(idx - pd.Timedelta(hours=48)).to_numpy()
    df["conso_h168"]   = conso.reindex(idx - pd.Timedelta(hours=168)).to_numpy()

    vals_7d = np.stack([conso.reindex(idx - pd.Timedelta(hours=h)).to_numpy()
                        for h in range(24, 169, 24)], axis=1)
    df["conso_mean_7d"] = np.nanmean(vals_7d, axis=1)

    vals_12w = np.stack([conso.reindex(idx - pd.Timedelta(hours=168 * w)).to_numpy()
                         for w in range(1, 13)], axis=1)
    df["conso_mean_12w"] = np.nanmean(vals_12w, axis=1)

    vals_52w = np.stack([conso.reindex(idx - pd.Timedelta(hours=168 * w)).to_numpy()
                         for w in range(1, 53)], axis=1)
    df["conso_mean_52w"] = np.nanmean(vals_52w, axis=1)

    df["temp_h24"]  = temp.reindex(idx - pd.Timedelta(hours=24)).to_numpy()
    df["temp_h48"]  = temp.reindex(idx - pd.Timedelta(hours=48)).to_numpy()
    df["temp_h168"] = temp.reindex(idx - pd.Timedelta(hours=168)).to_numpy()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [re.sub(r"\s*\(.*?\)\s*$", "", col).strip() for col in df.columns]
    df = df.rename(columns={"Date et Heure": _DATETIME_COL})

    dt     = pd.to_datetime(df[_DATETIME_COL], utc=True)
    dt_par = dt.dt.tz_convert("Europe/Paris")
    df = df.drop(columns=[_DATETIME_COL])

    hour_frac = dt_par.dt.hour + dt_par.dt.minute / 60
    doy       = dt_par.dt.dayofyear
    df["hour_sin"]  = np.sin(2 * np.pi * hour_frac            / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour_frac            / 24)
    df["day_sin"]   = np.sin(2 * np.pi * dt_par.dt.dayofweek  / 7)
    df["day_cos"]   = np.cos(2 * np.pi * dt_par.dt.dayofweek  / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt_par.dt.month      / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt_par.dt.month      / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * doy                  / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * doy                  / 365)

    df.index = dt

    fr_holidays = holidays.France()
    df["is_holiday"]           = [int(ts.date() in fr_holidays)                          for ts in dt_par]
    df["is_day_after_holiday"] = [int((ts - pd.Timedelta(days=1)).date() in fr_holidays) for ts in dt_par]

    df["is_energy_crisis"] = ((dt_par >= "2022-08-01") & (dt_par <= "2023-03-31")).astype(int).values

    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].replace("ND", np.nan), errors="coerce")

    df["heating_apparent"] = (17 - df["apparent_temperature"]).clip(lower=0)
    df["cooling_apparent"] = (df["apparent_temperature"] - 21).clip(lower=0)

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
    df = df.dropna(subset=[_CONSO_COL])

    feature_cols = [
        "hour_sin", "hour_cos",
        "day_sin",  "day_cos",
        "month_sin", "month_cos",
        "doy_sin", "doy_cos",
        "temperature_2m", "apparent_temperature",
        "precipitation", "cloud_cover",
        "heating_apparent", "cooling_apparent",
        "is_holiday", "is_day_after_holiday",
        "conso_h24", "conso_h48", "conso_h168",
        "conso_mean_7d", "conso_mean_12w", "conso_mean_52w",
        "temp_h24", "temp_h48", "temp_h168",
        "is_energy_crisis",
    ]

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    log.info(f"feature_cols.pkl sauvegardé ({len(feature_cols)} features)")

    PROCESSED_DIR.mkdir(exist_ok=True)
    df[feature_cols + [_CONSO_COL]].to_csv(PROCESSED_DIR / "transformed_data.csv", index=False)
    log.info(f"transformed_data.csv sauvegardé ({len(df)} lignes)")


if __name__ == "__main__":
    main()
