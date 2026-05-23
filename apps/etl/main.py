import sys
from pathlib import Path

# En local : transform.py et train.py sont dans scripts/
# En Docker : ils sont copiés dans le même répertoire que ce fichier
_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

import logging
from datetime import date, timedelta
from io import StringIO

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

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

def fetch_weather_forecast() -> pd.DataFrame:
    params = {
        "latitude":      46,
        "longitude":     2,
        "timezone":      "Europe/Paris",
        "hourly":        ",".join(_WEATHER_COLS),
        "forecast_days": 2,
    }
    resp = requests.get(_OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = (
        pd.to_datetime(df["time"])
        .dt.tz_localize("Europe/Paris", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.set_index("time")
    df = df.resample("30min").interpolate("linear")
    log.info(f"Météo forecast : {len(df)} slots 30-min")
    return df

def merge_weather(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes météo sur energy_df en matchant les timestamps UTC."""
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
        "timezone":  "Europe/Paris",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.content.decode("utf-8-sig")), sep=";")
    df = df.drop(columns=[c for c in _ECO2MIX_TR_EXTRA_COLS if c in df.columns])
    df = df.rename(columns=_ECO2MIX_RENAME)
    # Garder uniquement les créneaux :00 et :30 pour cohérence avec l'historique
    mask = pd.to_datetime(df["Date et Heure"], utc=True).dt.minute.isin([0, 30])
    df = df[mask].reset_index(drop=True)
    log.info(f"{len(df)} lignes récupérées pour le {date_str} (pas 30 min)")
    return df

def latest_raw_date() -> date | None:
    raw_csv = RAW_DIR / "raw_data.csv"
    df = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig", usecols=["Date"], low_memory=False)
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    return dates.max().date() if not dates.empty else None

def _fetch_energy_range(since: date, until: date) -> pd.DataFrame | None:
    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/csv"
    params = {
        "where":     f"date_heure >= '{(since - timedelta(days=1)).isoformat()}T22:00:00' AND date_heure < '{until.isoformat()}T00:00:00'",
        "limit":     -1,
        "delimiter": ";",
        "timezone":  "Europe/Paris",
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.content.decode("utf-8-sig")), sep=";")
        df = df[(df["date"] >= since.isoformat()) & (df["date"] < until.isoformat())]
        df = df.drop(columns=[c for c in _ECO2MIX_TR_EXTRA_COLS if c in df.columns])
        df = df.rename(columns=_ECO2MIX_RENAME)
        mask = pd.to_datetime(df["Date et Heure"], utc=True).dt.minute.isin([0, 30])
        df = df[mask].reset_index(drop=True)
        log.info(f"Énergie bulk : {len(df)} lignes ({since} → {until - timedelta(days=1)})")
        return df
    except Exception as e:
        log.warning(f"Énergie bulk ({since} → {until - timedelta(days=1)}) : {e}")
        return None

def _fetch_weather_range(since: date, until: date) -> pd.DataFrame | None:
    params = {
        "latitude":   46,
        "longitude":  2,
        "timezone":   "Europe/Paris",
        "hourly":     ",".join(_WEATHER_COLS),
        "start_date": since.isoformat(),
        "end_date":   (until - timedelta(days=1)).isoformat(),
    }
    try:
        resp = requests.get(_OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["hourly"])
        df["time"] = (
            pd.to_datetime(df["time"])
            .dt.tz_localize("Europe/Paris", ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
        df = df.set_index("time").resample("30min").interpolate("linear")
        log.info(f"Météo archive bulk : {len(df)} slots ({since} → {until - timedelta(days=1)})")
        return df
    except Exception as e:
        log.warning(f"Météo archive bulk ({since} → {until - timedelta(days=1)}) : {e}")
        return None


def backfill(since: date, until: date):
    days = (until - since).days
    if days <= 0:
        return
    log.info(f"Backfill : {days} jours ({since} → {until - timedelta(days=1)})")

    energy = _fetch_energy_range(since, until)
    if energy is None:
        log.warning("Backfill annulé : énergie bulk indisponible")
        return

    weather = _fetch_weather_range(since, until)
    if weather is not None:
        energy = merge_weather(energy, weather)
    else:
        log.warning("Météo indisponible, sauvegarde sans météo")

    upsert_raw(energy)


def upsert_raw(new_df: pd.DataFrame):
    raw_csv = RAW_DIR / "raw_data.csv"

    existing = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig", low_memory=False)

    # Moyenner les doublons UTC dans new_df avant de merger (ex: créneau DST dupliqué par eco2mix)
    new_df = new_df.copy()
    new_df["_dt_utc"] = pd.to_datetime(new_df["Date et Heure"], utc=True)
    if new_df["_dt_utc"].duplicated().any():
        num_cols = new_df.select_dtypes(include="number").columns.tolist()
        str_cols = [c for c in new_df.columns if c not in num_cols and c != "_dt_utc"]
        new_df = new_df.groupby("_dt_utc", sort=False).agg(
            {**{c: "first" for c in str_cols}, **{c: "mean" for c in num_cols}}
        ).reset_index().drop(columns=["_dt_utc"])
    else:
        new_df = new_df.drop(columns=["_dt_utc"])

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["_dt_utc"] = pd.to_datetime(combined["Date et Heure"], utc=True)
    combined = combined.drop_duplicates(subset=["_dt_utc"]).drop(columns=["_dt_utc"])
    combined = combined.sort_values(by=["Date", "Heure"]).reset_index(drop=True)

    added = len(combined) - len(existing)
    combined.to_csv(raw_csv, sep=";", encoding="utf-8-sig", index=False)
    log.info(f"{added} nouvelles lignes ajoutées ({len(combined)} total)")


def main():
    today = date.today()
    log.info(f"--- ETL données du {today} ---")

    last = latest_raw_date()
    if last is not None and last < today - timedelta(days=1):
        backfill(since=last + timedelta(days=1), until=today)

    energy = fetch_energy(today)
    if energy.empty:
        log.warning("Aucune donnée disponible pour aujourd'hui.")
        return

    forecast = fetch_weather_forecast()
    energy   = merge_weather(energy, forecast)
    upsert_raw(energy)
    update_datacard()

    forecast.to_csv(RAW_DIR / "weather_forecast.csv")
    log.info(f"weather_forecast.csv sauvegardé ({len(forecast)} slots)")

    log.info("--- ETL données terminé ---")


if __name__ == "__main__":
    main()
