import os
import re
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT          = Path(__file__).parent.parent
RAW_DIR       = Path(os.getenv("RAW_DIR",       str(ROOT / "data")))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT / "data")))
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    str(ROOT / "outputs")))

_DATETIME_COL = "date_heure"
_CONSO_COL    = "Consommation"


def prepare_features(df: pd.DataFrame, conso_series: pd.Series | None = None) -> pd.DataFrame:
    df = df.copy()

    # Normalise les noms : strip unités "(MW)" et unifie "Date et Heure" → "date_heure"
    df.columns = [re.sub(r"\s*\(.*?\)\s*$", "", col).strip() for col in df.columns]
    df = df.rename(columns={"Date et Heure": _DATETIME_COL})

    # Datetime UTC
    dt = pd.to_datetime(df[_DATETIME_COL], utc=True)
    df = df.drop(columns=[_DATETIME_COL])

    # Encodages temporels cycliques
    hour_frac = dt.dt.hour + dt.dt.minute / 60
    df["hour_sin"]  = np.sin(2 * np.pi * hour_frac        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour_frac        / 24)
    df["day_sin"]   = np.sin(2 * np.pi * dt.dt.dayofweek  / 7)
    df["day_cos"]   = np.cos(2 * np.pi * dt.dt.dayofweek  / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month      / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month      / 12)

    df.index = dt

    # Jours fériés France métropolitaine
    fr_holidays = holidays.France()
    df["is_holiday"] = dt.dt.date.map(lambda d: int(d in fr_holidays)).values

    # Convertit les colonnes object en numérique
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].replace("ND", np.nan), errors="coerce")

    # Lags consommation — série externe (inférence) ou interne (entraînement)
    if conso_series is not None:
        conso = conso_series
    else:
        conso = pd.to_numeric(df[_CONSO_COL], errors="coerce")
        conso = conso[~conso.index.duplicated(keep="first")]
    df["conso_h24"]  = conso.reindex(dt - pd.Timedelta(hours=24)).values
    df["conso_h168"] = conso.reindex(dt - pd.Timedelta(hours=168)).values

    # Transformations et lags météo
    df["heating_degrees"] = (17 - df["temperature_2m"]).clip(lower=0)
    df["cooling_degrees"] = (df["temperature_2m"] - 21).clip(lower=0)
    temp = df["temperature_2m"].copy()
    temp = temp[~temp.index.duplicated(keep="first")]
    df["temp_h24"] = temp.reindex(dt - pd.Timedelta(hours=24)).values

    dropna_cols = ["conso_h24", "conso_h168"]
    if conso_series is None:
        dropna_cols.append(_CONSO_COL)
    df = df.dropna(subset=dropna_cols)

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

    feature_cols = [
        "hour_sin", "hour_cos",
        "day_sin",  "day_cos",
        "month_sin", "month_cos",
        "conso_h24", "conso_h168",
        "temperature_2m", "apparent_temperature",
        "precipitation", "cloud_cover",
        "heating_degrees", "cooling_degrees",
        "temp_h24",
        "is_holiday",
    ]

    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(imputer,      MODELS_DIR / "imputer.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print(f"imputer.pkl sauvegardé ({len(feature_cols)} features)")

    PROCESSED_DIR.mkdir(exist_ok=True)
    df[feature_cols + [_CONSO_COL]].to_csv(
        PROCESSED_DIR / "transformed_data.csv", index=False
    )
    print(f"transformed_data.csv sauvegardé ({len(df)} lignes)")


if __name__ == "__main__":
    main()
