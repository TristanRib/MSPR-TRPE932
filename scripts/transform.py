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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Supprimer les unités entre parenthèses : "Consommation (MW)" → "Consommation"
    df.columns = [re.sub(r"\s*\(.*?\)\s*$", "", col).strip() for col in df.columns]

    df = df.drop(columns=[c for c in ["Périmètre", "Nature"] if c in df.columns])

    # Supporte "Date et Heure" (CSV) ou "date_heure" (API ODRE)
    if "Date et Heure" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date et Heure"], utc=True)
    elif "date_heure" in df.columns:
        df["datetime"] = pd.to_datetime(df["date_heure"])

    df["hour_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["day_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)

    drop_cols = ["datetime", "Date", "Heure", "Date et Heure", "date_heure"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].replace("ND", np.nan), errors="coerce")

    return df


def main():
    df = pd.read_csv(
        RAW_DIR / "eco2mix-national-cons-def.csv",
        sep=";",
        encoding="utf-8-sig",
        index_col=False,
        low_memory=False,
    )
    df = df.sort_values(by=["Date", "Heure"], ascending=True).reset_index(drop=True)

    df = prepare_features(df)

    imputer = SimpleImputer(strategy="median")
    feature_cols = [c for c in df.columns if c != "Consommation"]

    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    df = df.dropna(subset=["Consommation"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(imputer,      MODELS_DIR / "imputer.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print(f"imputer.pkl sauvegardé ({len(feature_cols)} features)")

    PROCESSED_DIR.mkdir(exist_ok=True)
    df.to_csv(PROCESSED_DIR / "transformed_data.csv", index=False)
    print(f"transformed_data.csv sauvegardé ({len(df)} lignes)")


if __name__ == "__main__":
    main()
