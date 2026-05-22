import sys
from pathlib import Path

# En local : transform.py et train.py sont dans scripts/
# En Docker : ils sont copiés dans le même répertoire que ce fichier
_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

from datetime import date, timedelta
from io import StringIO

import requests
import pandas as pd

from transform import RAW_DIR, _WEATHER_COLS
from datacard import generate as update_datacard

_OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"


_ECO2MIX_RENAME = {
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

_ECO2MIX_TR_EXTRA_COLS = {"eolien_terrestre", "eolien_offshore", "stockage_batterie", "destockage_batterie"}


def fetch_weather(target_date: date) -> pd.DataFrame:
    """Fetches hourly weather from Open-Meteo and resamples to 15-min intervals.
    Returns a DataFrame indexed by UTC-aware datetimes."""
    date_str = target_date.isoformat()
    params = {
        "latitude":   46,
        "longitude":  2,
        "timezone":   "Europe/Paris",
        "hourly":     ",".join(_WEATHER_COLS),
        "start_date": date_str,
        "end_date":   date_str,
    }
    use_archive = (date.today() - target_date).days > 7
    url  = _OPEN_METEO_ARCHIVE_URL if use_archive else _OPEN_METEO_FORECAST_URL
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(hourly)
    df["time"] = (
        pd.to_datetime(df["time"])
        .dt.tz_localize("Europe/Paris", ambiguous="infer")
        .dt.tz_convert("UTC")
    )
    df = df.set_index("time")
    df = df.resample("15min").interpolate("linear")
    print(f"Meteo : {len(df)} slots 15-min ({date_str})")
    return df


def merge_weather(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merges weather columns onto energy_df by matching UTC timestamps."""
    dt = pd.to_datetime(energy_df["Date et Heure"], utc=True)
    energy_df = energy_df.copy()
    for col in _WEATHER_COLS:
        energy_df[col] = weather_df[col].reindex(dt).values
    return energy_df


def fetch_energy(target_date: date) -> pd.DataFrame:
    date_str = target_date.strftime("%Y-%m-%d")
    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/csv"
    params = {
        "where":     f"date='{date_str}'",
        "limit":     -1,
        "delimiter": ";",
    }
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.content.decode("utf-8-sig")), sep=";")
    df = df.drop(columns=[c for c in _ECO2MIX_TR_EXTRA_COLS if c in df.columns])
    df = df.rename(columns=_ECO2MIX_RENAME)
    print(f"{len(df)} lignes récupérées pour le {date_str}")
    return df


def upsert_raw(new_df: pd.DataFrame):
    raw_csv = RAW_DIR / "raw_data.csv"

    existing = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig", low_memory=False)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date et Heure"])
    combined = combined.sort_values(by=["Date", "Heure"]).reset_index(drop=True)

    added = len(combined) - len(existing)
    combined.to_csv(raw_csv, sep=";", encoding="utf-8-sig", index=False)
    print(f"{added} nouvelles lignes ajoutées ({len(combined)} total)")


def latest_raw_date() -> date | None:
    raw_csv = RAW_DIR / "raw_data.csv"
    df = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig", usecols=["Date"], low_memory=False)
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    return dates.max().date() if not dates.empty else None


def backfill(since: date, until: date):
    missing = [since + timedelta(days=i) for i in range((until - since).days)]
    if not missing:
        return
    print(f"Backfill : {len(missing)} jours manquants ({missing[0]} -> {missing[-1]})")
    for target in missing:
        energy = fetch_energy(target)
        if energy.empty:
            print(f"  {target} : aucune donnée, ignoré")
            continue
        weather = fetch_weather(target)
        energy  = merge_weather(energy, weather)
        upsert_raw(energy)


def main():
    today = date.today()
    print(f"--- ETL données du {today} ---")

    last = latest_raw_date()
    if last is not None and last < today - timedelta(days=1):
        backfill(since=last + timedelta(days=1), until=today)

    energy = fetch_energy(today)
    if energy.empty:
        print("Aucune donnée disponible pour aujourd'hui.")
        return

    weather = fetch_weather(today)
    energy  = merge_weather(energy, weather)
    upsert_raw(energy)
    update_datacard()

    print("--- ETL données terminé ---")


if __name__ == "__main__":
    main()
