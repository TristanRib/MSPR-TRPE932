import sys
from pathlib import Path

# En local : transform.py et train.py sont dans scripts/
# En Docker : ils sont copiés dans le même répertoire que ce fichier
_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent / "scripts"))

import os
from datetime import date, timedelta
from io import StringIO

import requests
import pandas as pd

from transform import main as run_transform, RAW_DIR
from train import main as run_train
from datacard import generate as update_datacard


_API_TO_FRENCH = {
    "perimetre":                    "Périmètre",
    "nature":                       "Nature",
    "date":                         "Date",
    "heure":                        "Heure",
    "date_heure":                   "Date et Heure",
    "consommation":                 "Consommation (MW)",
    "prevision_j1":                 "Prévision J-1 (MW)",
    "prevision_j":                  "Prévision J (MW)",
    "fioul":                        "Fioul (MW)",
    "charbon":                      "Charbon (MW)",
    "gaz":                          "Gaz (MW)",
    "nucleaire":                    "Nucléaire (MW)",
    "eolien":                       "Eolien (MW)",
    "solaire":                      "Solaire (MW)",
    "hydraulique":                  "Hydraulique (MW)",
    "pompage":                      "Pompage (MW)",
    "bioenergies":                  "Bioénergies (MW)",
    "ech_physiques":                "Ech. physiques (MW)",
    "taux_co2":                     "Taux de CO2 (g/kWh)",
    "ech_comm_angleterre":          "Ech. comm. Angleterre (MW)",
    "ech_comm_espagne":             "Ech. comm. Espagne (MW)",
    "ech_comm_italie":              "Ech. comm. Italie (MW)",
    "ech_comm_suisse":              "Ech. comm. Suisse (MW)",
    "ech_comm_allemagne_belgique":  "Ech. comm. Allemagne-Belgique (MW)",
    "fioul_tac":                    "Fioul - TAC (MW)",
    "fioul_cogen":                  "Fioul - Cogénération (MW)",
    "fioul_autres":                 "Fioul - Autres (MW)",
    "gaz_tac":                      "Gaz - TAC (MW)",
    "gaz_cogen":                    "Gaz - Cogénération (MW)",
    "gaz_ccg":                      "Gaz - CCG (MW)",
    "gaz_autres":                   "Gaz - Autres (MW)",
    "hydraulique_fil_eau_eclusee":  "Hydraulique - Fil de l'eau + éclusée (MW)",
    "hydraulique_lacs":             "Hydraulique - Lacs (MW)",
    "hydraulique_step_turbinage":   "Hydraulique - STEP turbinage (MW)",
    "bioenergies_dechets":          "Bioénergies - Déchets (MW)",
    "bioenergies_biomasse":         "Bioénergies - Biomasse (MW)",
    "bioenergies_biogaz":           "Bioénergies - Biogaz (MW)",
}

_TR_EXTRA_COLS = {"eolien_terrestre", "eolien_offshore", "stockage_batterie", "destockage_batterie"}


def fetch_day(target_date: date) -> pd.DataFrame:
    date_str = target_date.strftime("%Y-%m-%d")
    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/csv"
    params = {
        "where":     f"date='{date_str}'",
        "limit":     -1,
        "delimiter": ";",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.content.decode("utf-8-sig")), sep=";")
    df = df.drop(columns=[c for c in _TR_EXTRA_COLS if c in df.columns])
    df = df.rename(columns=_API_TO_FRENCH)
    print(f"{len(df)} lignes récupérées pour le {date_str}")
    return df


def append_to_raw(new_df: pd.DataFrame):
    raw_csv = RAW_DIR / "eco2mix-national-cons-def.csv"

    existing = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig", low_memory=False)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date et Heure"])
    combined = combined.sort_values(by=["Date", "Heure"]).reset_index(drop=True)

    added = len(combined) - len(existing)
    combined.to_csv(raw_csv, sep=";", encoding="utf-8-sig", index=False)
    print(f"{added} nouvelles lignes ajoutées ({len(combined)} total)")


def redeploy_api():
    api_url = os.getenv("API_URL", "")
    if not api_url:
        print("API_URL non défini, skip reload")
        return
    resp = requests.post(f"{api_url}/reload", timeout=30)
    resp.raise_for_status()
    print(f"API rechargée : {resp.json()['model_loaded']}")


def main():
    yesterday = date.today() - timedelta(days=1)
    print(f"--- Pipeline ETL du {yesterday} ---")

    new_df = fetch_day(yesterday)
    if new_df.empty:
        print("Aucune donnée disponible, abandon.")
        return

    append_to_raw(new_df)
    update_datacard()
    run_transform()
    run_train()
    redeploy_api()

    print("--- Pipeline terminé ---")


if __name__ == "__main__":
    main()
