import sys
from pathlib import Path

# En local : transform.py et train.py sont dans scripts/
# En Docker : ils sont copiés dans le même répertoire que ce fichier
_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

import logging
import os
import time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from io import StringIO

import numpy as np
import requests
import pandas as pd

try:
    from google.cloud.logging.handlers import StructuredLogHandler
    logging.basicConfig(handlers=[StructuredLogHandler()], level=logging.INFO)
except Exception:
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


def _with_retry(fn, *args, label: str, retries: int = 4, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"{label} tentative {attempt + 1}/{retries} échouée ({e}), retry dans {wait}s")
                time.sleep(wait)
    log.error(f"{label} indisponible après {retries} tentatives : {last_exc}")
    raise RuntimeError(f"{label} indisponible après {retries} tentatives") from last_exc


def fetch_weather_forecast() -> pd.DataFrame:
    params = {
        "latitude":      46,
        "longitude":     2,
        "timezone":      "Europe/Paris",
        "hourly":        ",".join(_WEATHER_COLS),
        "forecast_days": 14,
    }
    def _fetch():
        resp = requests.get(_OPEN_METEO_FORECAST_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp
    resp = _with_retry(_fetch, label="Météo forecast")
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = (
        pd.to_datetime(df["time"])
        .dt.tz_localize("Europe/Paris", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.set_index("time")
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("30min").interpolate("linear").ffill()
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
    def _fetch():
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp
    resp = _with_retry(_fetch, label=f"Énergie {date_str}")
    df = pd.read_csv(StringIO(resp.content.decode("utf-8-sig")), sep=";")
    df = df.drop(columns=[c for c in _ECO2MIX_TR_EXTRA_COLS if c in df.columns])
    df = df.rename(columns=_ECO2MIX_RENAME)
    mask = pd.to_datetime(df["Date et Heure"], utc=True).dt.minute.isin([0, 30])
    df = df[mask].reset_index(drop=True)
    log.info(f"{len(df)} lignes récupérées pour le {date_str} (pas 30 min)")
    return df


def _slot_dates(df: pd.DataFrame) -> "pd.Series[date]":
    return (pd.to_datetime(df["Date et Heure"], utc=True)
            .dt.tz_convert("Europe/Paris").dt.date)


def _find_gaps() -> list[tuple[date, date]]:
    """Plages de dates avec slots énergie ou météo incomplets, jusqu'à hier inclus."""
    raw_csv = RAW_DIR / "raw_data.csv"
    df = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig",
                     usecols=["Date et Heure", "Consommation (MW)", "temperature_2m"],
                     low_memory=False)
    yesterday = datetime.now(ZoneInfo("Europe/Paris")).date() - timedelta(days=1)
    df["_date"] = _slot_dates(df)

    min_date = df["_date"].dropna().min()
    if pd.isna(min_date):
        return []

    energy_ok  = df[df["Consommation (MW)"].notna()].groupby("_date").size()
    weather_ok = df[df["temperature_2m"].notna()].groupby("_date").size()
    total      = df.groupby("_date").size()

    gap_dates = sorted(
        d for d in (min_date + timedelta(days=i)
                    for i in range((yesterday - min_date).days + 1))
        if d not in total.index                        # date complètement absente
        or energy_ok.get(d, 0) < total[d]             # slots avec NaN énergie
        or weather_ok.get(d, 0) < total[d]            # slots avec NaN météo
    )

    if not gap_dates:
        return []
    ranges, start, prev = [], gap_dates[0], gap_dates[0]
    for d in gap_dates[1:]:
        if (d - prev).days > 1:
            ranges.append((start, prev + timedelta(days=1)))
            start = d
        prev = d
    ranges.append((start, prev + timedelta(days=1)))
    return ranges



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
        df = df.set_index("time")
        df = df[~df.index.duplicated(keep="first")]
        df = df.resample("30min").interpolate("linear").ffill()
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

    combined = pd.concat([new_df, existing], ignore_index=True)
    combined["_dt_utc"] = pd.to_datetime(combined["Date et Heure"], utc=True)
    combined = combined.drop_duplicates(subset=["_dt_utc"]).drop(columns=["_dt_utc"])
    combined = combined.sort_values(by=["Date", "Heure"]).reset_index(drop=True)

    added = len(combined) - len(existing)
    combined.to_csv(raw_csv, sep=";", encoding="utf-8-sig", index=False)
    log.info(f"{added} nouvelles lignes ajoutées ({len(combined)} total)")


# Fallback statique si l'historique BQ est insuffisant (< MIN_SLOTS_PER_MONTH slots/mois)
_FALLBACK_RMSE_REF = {
    1: 1922, 2: 1800, 3: 1500, 4: 1200, 5: 1000,
    6:  900, 7:  759, 8:  759, 9:  900, 10: 1300,
    11: 1600, 12: 1900,
}
_FALLBACK_BIAS_REF = {
    1:  462, 2:  400, 3:  200, 4:   50, 5:  -50,
    6: -150, 7: -193, 8: -193, 9: -100, 10:  50,
    11: 250, 12: 400,
}
DRIFT_RMSE_MULTIPLIER  = 1.5    # alerte si RMSE > 1.5× la référence du mois
DRIFT_BIAS_DELTA       = 400.0  # MW — écart vs biais attendu du mois
MIN_SLOTS_PER_MONTH    = 200    # slots minimum pour considérer la référence BQ fiable


_PSI_WEATHER_FEATURES = {
    "temperature_2m", "apparent_temperature", "precipitation", "cloud_cover",
    "heating_apparent", "cooling_apparent",
}


def _check_feature_psi():
    models_dir = os.getenv("MODELS_DIR", "")
    if not models_dir:
        log.info("PSI ignoré : MODELS_DIR non défini")
        return

    dist_path = Path(models_dir) / "feature_distributions.pkl"
    if not dist_path.exists():
        log.info("PSI ignoré : feature_distributions.pkl manquant")
        return

    try:
        import joblib
        distributions = joblib.load(dist_path)

        # raw_data.csv est mis à jour à chaque run ETL — source fraîche
        raw_csv = RAW_DIR / "raw_data.csv"
        df = pd.read_csv(
            raw_csv, sep=";", encoding="utf-8-sig",
            usecols=["Date et Heure", "temperature_2m", "apparent_temperature",
                     "precipitation", "cloud_cover"],
            low_memory=False,
        )
        df = df.dropna(subset=["temperature_2m"])
        df["heating_apparent"] = (17 - df["apparent_temperature"]).clip(lower=0)
        df["cooling_apparent"] = (df["apparent_temperature"] - 21).clip(lower=0)

        recent = df.tail(336)  # ~7 jours à 30-min
        month  = pd.to_datetime(recent["Date et Heure"], utc=True).dt.month.mode()[0]

        if month not in distributions:
            log.info(f"PSI : pas de référence pour le mois {month}")
            return

        ref = distributions[month]
        drift, watch, stable = [], [], []

        for col in _PSI_WEATHER_FEATURES:
            if col not in ref or col not in recent.columns:
                continue
            vals = recent[col].dropna().values
            if len(vals) < 10:
                continue
            bins    = np.array(ref[col]["bins"])
            ref_pct = np.array(ref[col]["ref_pct"])
            counts, _ = np.histogram(vals, bins=bins)
            if counts.sum() == 0:
                continue
            cur_pct = counts / counts.sum()
            cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

            if psi > 0.25:
                drift.append((col, psi))
            elif psi > 0.1:
                watch.append((col, psi))
            else:
                stable.append(col)

        if drift:
            drift_str = "  ".join(f"{col}={psi:.2f}" for col, psi in sorted(drift, key=lambda x: -x[1]))
            log.warning(f"PSI — dérive météo (mois={month}) : {drift_str}")
        if watch:
            watch_str = "  ".join(f"{col}={psi:.2f}" for col, psi in sorted(watch, key=lambda x: -x[1]))
            log.info(f"PSI — météo à surveiller (mois={month}) : {watch_str}")
        log.info(
            f"PSI mois {month} : {len(drift)} en dérive (>0.25)"
            f", {len(watch)} à surveiller (0.1-0.25)"
            f", {len(stable)} stables (<0.1)"
        )
    except Exception as e:
        log.warning(f"Contrôle PSI échoué : {e}")


def _check_prediction_drift():
    bq_table = os.getenv("BQ_TABLE", "")
    if not bq_table:
        log.info("Dérive ignorée : BQ_TABLE non défini")
        return
    try:
        from google.cloud import bigquery
        client = bigquery.Client()

        # Toutes les prédictions passées — dernière run_at par slot (dédup WRITE_APPEND)
        all_query = f"""
            SELECT predicted_at, prediction_mw
            FROM `{bq_table}`
            WHERE predicted_at < CURRENT_TIMESTAMP()
            QUALIFY ROW_NUMBER() OVER (PARTITION BY predicted_at ORDER BY run_at DESC) = 1
            ORDER BY predicted_at
        """
        all_pred_df = client.query(all_query).to_dataframe()
        if all_pred_df.empty:
            log.info("Dérive : aucune prédiction passée en BQ")
            return

        raw_csv = RAW_DIR / "raw_data.csv"
        actual_df = pd.read_csv(
            raw_csv, sep=";", encoding="utf-8-sig",
            usecols=["Date et Heure", "Consommation (MW)"], low_memory=False,
        )
        actual_df["predicted_at"] = pd.to_datetime(actual_df["Date et Heure"], utc=True)
        actual_df["actual_mw"] = pd.to_numeric(actual_df["Consommation (MW)"], errors="coerce")

        all_pred_df["predicted_at"] = pd.to_datetime(all_pred_df["predicted_at"], utc=True)
        all_merged = all_pred_df.merge(
            actual_df[["predicted_at", "actual_mw"]], on="predicted_at", how="inner"
        ).dropna(subset=["actual_mw"])

        if len(all_merged) < 10:
            log.info(f"Dérive : seulement {len(all_merged)} slots communs, skip")
            return

        # Références dynamiques par mois depuis l'historique BQ
        all_merged["month"] = all_merged["predicted_at"].dt.month
        all_merged["residual"] = all_merged["prediction_mw"] - all_merged["actual_mw"]
        monthly_stats = all_merged.groupby("month")["residual"].agg(
            rmse_ref=lambda r: float(np.sqrt((r ** 2).mean())),
            bias_ref="mean",
            count="count",
        )

        month = datetime.now(ZoneInfo("Europe/Paris")).month
        if month in monthly_stats.index and monthly_stats.loc[month, "count"] >= MIN_SLOTS_PER_MONTH:
            rmse_ref = float(monthly_stats.loc[month, "rmse_ref"])
            bias_ref = float(monthly_stats.loc[month, "bias_ref"])
            ref_source = f"BQ ({int(monthly_stats.loc[month, 'count'])} slots)"
        else:
            rmse_ref = _FALLBACK_RMSE_REF[month]
            bias_ref = _FALLBACK_BIAS_REF[month]
            ref_source = "fallback statique"

        # Métriques sur les 7 derniers jours
        recent = all_merged[
            all_merged["predicted_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
        ]
        if len(recent) < 10:
            log.info(f"Dérive : seulement {len(recent)} slots récents, skip")
            return

        bias = float(recent["residual"].mean())
        rmse = float(np.sqrt((recent["residual"] ** 2).mean()))
        rmse_threshold = rmse_ref * DRIFT_RMSE_MULTIPLIER

        log.info(
            f"Dérive 7j (mois={month}, réf={ref_source}) :"
            f" RMSE={rmse:.0f} MW (réf={rmse_ref:.0f}, seuil={rmse_threshold:.0f})"
            f"  biais={bias:+.0f} MW (réf={bias_ref:+.0f})  ({len(recent)} slots)"
        )

        if rmse > rmse_threshold:
            log.warning(f"DÉRIVE — RMSE {rmse:.0f} MW > {rmse_threshold:.0f} MW (1.5× réf mois {month})")
        if abs(bias - bias_ref) > DRIFT_BIAS_DELTA:
            log.warning(
                f"DÉRIVE — biais {bias:+.0f} MW s'écarte de {bias - bias_ref:+.0f} MW"
                f" vs attendu {bias_ref:+.0f} MW (seuil ±{DRIFT_BIAS_DELTA:.0f} MW)"
            )
    except Exception as e:
        log.warning(f"Contrôle dérive échoué : {e}")


def main():
    today = datetime.now(ZoneInfo("Europe/Paris")).date()
    log.info(f"--- ETL données du {today} ---")

    # 1. Combler les gaps des runs précédentes
    for since, until in _find_gaps():
        log.info(f"Gap détecté : {since} → {until - timedelta(days=1)}, backfill...")
        backfill(since, until)

    # 2. Fetch d'aujourd'hui — indépendants
    energy_exc = weather_exc = None
    energy = forecast = None

    try:
        energy = fetch_energy(today)
        if energy.empty:
            raise ValueError("Aucune donnée énergie disponible")
    except Exception as e:
        log.error(f"Énergie {today} indisponible : {e}")
        energy_exc = e

    try:
        forecast = fetch_weather_forecast()
    except Exception as e:
        log.error(f"Météo forecast indisponible : {e}")
        weather_exc = e

    if energy_exc and weather_exc:
        raise RuntimeError("Énergie et météo indisponibles") from energy_exc

    # 3. Upsert énergie (avec météo si dispo)
    if energy is not None:
        upsert_raw(merge_weather(energy, forecast) if forecast is not None else energy)

    update_datacard()
    _check_feature_psi()
    _check_prediction_drift()

    if forecast is not None:
        forecast.to_csv(RAW_DIR / "weather_forecast.csv")
        log.info(f"weather_forecast.csv sauvegardé ({len(forecast)} slots)")

    log.info("--- ETL données terminé ---")

    # Job fail si partiel — gap détecté au prochain run
    if energy_exc:
        raise RuntimeError("ETL partiel — énergie indisponible") from energy_exc
    if weather_exc:
        raise RuntimeError("ETL partiel — météo indisponible") from weather_exc


if __name__ == "__main__":
    main()
