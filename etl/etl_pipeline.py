import sys
from pathlib import Path

# En local : transform.py et train.py sont dans scripts/
# En Docker : ils sont copiés dans le même répertoire que ce fichier
_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent / "scripts"))

from datetime import date, timedelta
from io import StringIO

import requests
import pandas as pd

from transform import main as run_transform, RAW_DIR
from train import main as run_train
from datacard import generate as update_datacard


def fetch_day(target_date: date) -> pd.DataFrame:
    date_str = target_date.strftime("%Y-%m-%d")
    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv"
    params = {
        "where":     f"date='{date_str}'",
        "limit":     -1,
        "delimiter": ";",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.content.decode("utf-8-sig")), sep=";")
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

    print("--- Pipeline terminé ---")


if __name__ == "__main__":
    main()
