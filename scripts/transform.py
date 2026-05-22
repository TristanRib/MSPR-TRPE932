import os
import re
from pathlib import Path

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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Normalise les noms : strip unités "(MW)" et unifie "Date et Heure" → "date_heure"
    df.columns = [re.sub(r"\s*\(.*?\)\s*$", "", col).strip() for col in df.columns]
    df = df.rename(columns={"Date et Heure": _DATETIME_COL})

    # 2. Encodages temporels cycliques
    dt = pd.to_datetime(df[_DATETIME_COL], utc=True)
    hour_frac = dt.dt.hour + dt.dt.minute / 60
    df["hour_sin"]  = np.sin(2 * np.pi * hour_frac        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour_frac        / 24)
    df["day_sin"]   = np.sin(2 * np.pi * dt.dt.dayofweek  / 7)
    df["day_cos"]   = np.cos(2 * np.pi * dt.dt.dayofweek  / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month      / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month      / 12)

    df = df.drop(columns=[_DATETIME_COL])

    # 3. Convertit les colonnes object restantes en numérique ("ND" → NaN)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].replace("ND", np.nan), errors="coerce")

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

    dt = pd.to_datetime(df["Date et Heure"], utc=True)

    df = prepare_features(df)  # "Consommation (MW)" → "Consommation", drop date_heure
    df.index = dt

    conso = pd.to_numeric(df[_CONSO_COL].replace("ND", np.nan), errors="coerce")
    conso = conso[~conso.index.duplicated(keep="first")]
    df["conso_h24"]  = conso.reindex(dt - pd.Timedelta(hours=24)).values
    df["conso_h168"] = conso.reindex(dt - pd.Timedelta(hours=168)).values

    df = df.dropna(subset=["conso_h24", "conso_h168", _CONSO_COL])

    feature_cols = [
        "hour_sin", "hour_cos",
        "day_sin",  "day_cos",
        "month_sin", "month_cos",
        "conso_h24", "conso_h168",
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