import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT     = Path(__file__).parent.parent
RAW_DIR  = Path(os.getenv("RAW_DIR", str(ROOT / "data")))
RAW_CSV  = RAW_DIR / "eco2mix-national-cons-def.csv"
CARD_OUT = RAW_DIR / "raw_data.yaml"

DESCRIPTIONS = {
    "Périmètre":                          "France",
    "Nature":                             "Données temps réel / Données consolidées / Données définitives",
    "Date":                               "Date du jour (jj/mm/aaaa)",
    "Heure":                              "Point horaire par pas de 15 minutes (hh:mm)",
    "Date et Heure":                      "Horodatage ISO 8601",
    "Consommation (MW)":                  "Consommation en MW",
    "Prévision J-1 (MW)":                "Prévision J-1 de consommation en MW",
    "Prévision J (MW)":                  "Prévision J de consommation en MW",
    "Fioul (MW)":                         "Production fioul en MW",
    "Charbon (MW)":                       "Production charbon en MW",
    "Gaz (MW)":                           "Production gaz en MW",
    "Nucléaire (MW)":                     "Production nucléaire en MW",
    "Eolien (MW)":                        "Production éolienne en MW",
    "Solaire (MW)":                       "Production solaire en MW",
    "Hydraulique (MW)":                   "Production hydraulique en MW",
    "Pompage (MW)":                       "Pompage hydraulique en MW",
    "Bioénergies (MW)":                   "Production Bioénergies en MW",
    "Ech. physiques (MW)":               "Solde imports/exports (flux physiques) en MW",
    "Taux de CO2 (g/kWh)":              "Estimation des émissions de CO2 en g/kWh",
    "Ech. comm. Angleterre (MW)":        "Solde imports/exports Angleterre en MW",
    "Ech. comm. Espagne (MW)":           "Solde imports/exports Espagne en MW",
    "Ech. comm. Italie (MW)":            "Solde imports/exports Italie en MW",
    "Ech. comm. Suisse (MW)":            "Solde imports/exports Suisse en MW",
    "Ech. comm. Allemagne-Belgique (MW)": "Solde imports/exports Allemagne-Belgique en MW",
    "Fioul - TAC (MW)":                  "Détail technologie turbine à combustion — fioul",
    "Fioul - Cogénération (MW)":         "Détail technologie cogénération — fioul",
    "Fioul - Autres (MW)":               "Détail autres technologies — fioul",
    "Gaz - TAC (MW)":                    "Détail technologie turbine à combustion — gaz",
    "Gaz - Cogénération (MW)":           "Détail technologie cogénération — gaz",
    "Gaz - CCG (MW)":                    "Détail technologie cycle combiné gaz",
    "Gaz - Autres (MW)":                 "Détail autres technologies — gaz",
    "Hydraulique - Fil de l'eau + éclusée (MW)": "Détail fil de l'eau et éclusée — hydraulique",
    "Hydraulique - Lacs (MW)":           "Détail technologie lacs — hydraulique",
    "Hydraulique - STEP turbinage (MW)": "Détail technologie STEP turbinage — hydraulique",
    "Bioénergies - Déchets (MW)":        "Détail technologie déchets — bioénergies",
    "Bioénergies - Biomasse (MW)":       "Détail technologie biomasse — bioénergies",
    "Bioénergies - Biogaz (MW)":         "Détail technologie biogaz — bioénergies",
}


def _col_stats(series: pd.Series, n_total: int) -> dict:
    count    = int(series.count())
    nunique  = int(series.nunique())
    missing  = int(series.isna().sum())

    stats = {
        "count":           count,
        "unique_count":    nunique,
        "duplicate_count": count - nunique,
        "duplicate_ratio": round((count - nunique) / count, 2) if count else 0.0,
        "missing_count":   missing,
        "missing_ratio":   round(missing / n_total, 2),
    }
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        stats.update({
            "mean": float(desc["mean"]),
            "std":  float(desc["std"]),
            "min":  float(desc["min"]),
            "25%":  float(desc["25%"]),
            "50%":  float(desc["50%"]),
            "75%":  float(desc["75%"]),
            "max":  float(desc["max"]),
        })
    return stats


def generate(csv_path: Path = RAW_CSV, out_path: Path = CARD_OUT):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
    n  = len(df)

    period = "unknown"
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if not dates.empty:
            period = f"{dates.min().year}-{dates.max().year}"

    feature_list = {}
    for col in df.columns:
        feature_list[col] = {
            "description": DESCRIPTIONS.get(col, ""),
            "type":        str(df[col].dtype),
            "stats":       _col_stats(df[col], n),
            "sample":      df[col].dropna().unique()[:5].tolist(),
        }

    card = {
        "name":    "eCO2mix_RTE_Annuel-Definitif",
        "version": "1.0",
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "domain":  "energy",
        "period":  period,
        "data": {
            "instance_count": n,
            "feature_count":  len(df.columns),
            "feature_list":   feature_list,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(card, f, allow_unicode=True, sort_keys=False)

    print(f"Datacard mise à jour : {n} lignes, période {period}")


if __name__ == "__main__":
    generate()
