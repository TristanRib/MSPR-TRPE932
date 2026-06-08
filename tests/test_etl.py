"""
Tests unitaires de l'ETL : retry, merge_weather, upsert_raw, _find_gaps, PSI.
Aucune dépendance réseau ni GCS — fonctions pures et filesystem via tmp_path.
"""
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Chargement explicite pour éviter de polluer sys.path avec apps/etl/
# et de casser le `import main` des autres fichiers (qui cible apps/api/main.py)
_ETL_PATH = Path(__file__).parent.parent / "apps" / "etl" / "main.py"
_spec = importlib.util.spec_from_file_location("etl_main", _ETL_PATH)
etl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(etl)


# ── helpers ──────────────────────────────────────────────────────────────────

def _slot(day: str, hour: int = 12, energy: float = 40000.0, temp: float = 15.0) -> dict:
    dt = pd.Timestamp(f"{day}T{hour:02d}:00:00", tz="UTC")
    return {
        "Date":              day,
        "Heure":             f"{hour:02d}:00",
        "Date et Heure":     dt.isoformat(),
        "Consommation (MW)": energy,
        "temperature_2m":    temp,
    }


def _write_csv(tmp_path: Path, rows: list) -> None:
    pd.DataFrame(rows).to_csv(
        tmp_path / "raw_data.csv", sep=";", encoding="utf-8-sig", index=False
    )


def _days(n: int) -> list[str]:
    """n jours allant jusqu'à hier inclus."""
    yesterday = date.today() - timedelta(days=1)
    return [(yesterday - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


# ── _with_retry ───────────────────────────────────────────────────────────────

class TestWithRetry:
    def test_succeed_first_try(self):
        result = etl._with_retry(lambda: "ok", label="test", retries=4)
        assert result == "ok"

    def test_retries_then_succeeds(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("fail")
            return "ok"
        with patch("time.sleep"):
            result = etl._with_retry(fn, label="test", retries=4)
        assert result == "ok"
        assert len(attempts) == 3

    def test_exhausted_raises_runtime_error(self):
        def always_fail():
            raise ValueError("always")
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="indisponible"):
                etl._with_retry(always_fail, label="test", retries=4)


# ── merge_weather ─────────────────────────────────────────────────────────────

class TestMergeWeather:
    def test_adds_weather_columns(self):
        ts = "2026-05-01T12:00:00+00:00"
        energy = pd.DataFrame({"Date et Heure": [ts], "Consommation (MW)": [40000.0]})
        weather = pd.DataFrame(
            {"temperature_2m": [18.0], "apparent_temperature": [17.0],
             "precipitation": [0.5], "cloud_cover": [20.0]},
            index=pd.to_datetime([ts], utc=True),
        )
        result = etl.merge_weather(energy, weather)
        assert result["temperature_2m"].iloc[0] == 18.0
        assert result["precipitation"].iloc[0] == 0.5

    def test_missing_slot_is_nan(self):
        energy = pd.DataFrame({"Date et Heure": ["2026-05-01T12:00:00+00:00"],
                                "Consommation (MW)": [40000.0]})
        weather = pd.DataFrame(
            {"temperature_2m": [15.0], "apparent_temperature": [14.0],
             "precipitation": [0.0], "cloud_cover": [30.0]},
            index=pd.to_datetime(["2026-05-02T12:00:00+00:00"], utc=True),  # mauvaise date
        )
        result = etl.merge_weather(energy, weather)
        assert pd.isna(result["temperature_2m"].iloc[0])


# ── upsert_raw ────────────────────────────────────────────────────────────────

class TestUpsertRaw:
    def test_adds_new_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(etl, "RAW_DIR", tmp_path)
        days = _days(3)
        _write_csv(tmp_path, [_slot(d) for d in days[:2]])

        new_row = pd.DataFrame([_slot(days[2])])
        etl.upsert_raw(new_row)

        result = pd.read_csv(tmp_path / "raw_data.csv", sep=";", encoding="utf-8-sig")
        assert len(result) == 3

    def test_deduplicates_on_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(etl, "RAW_DIR", tmp_path)
        day = _days(1)[0]
        _write_csv(tmp_path, [_slot(day)])

        etl.upsert_raw(pd.DataFrame([_slot(day)]))  # même slot

        result = pd.read_csv(tmp_path / "raw_data.csv", sep=";", encoding="utf-8-sig")
        assert len(result) == 1

    def test_dst_averages_numeric_duplicates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(etl, "RAW_DIR", tmp_path)
        day = _days(2)[0]
        _write_csv(tmp_path, [_slot(day, hour=0, energy=30000.0)])

        # Deux lignes avec le même timestamp UTC mais énergies différentes
        ts = pd.Timestamp(f"{day}T12:00:00", tz="UTC").isoformat()
        dupes = pd.DataFrame([
            {"Date": day, "Heure": "12:00", "Date et Heure": ts,
             "Consommation (MW)": 40000.0, "temperature_2m": 15.0},
            {"Date": day, "Heure": "12:00", "Date et Heure": ts,
             "Consommation (MW)": 60000.0, "temperature_2m": 15.0},
        ])
        etl.upsert_raw(dupes)

        result = pd.read_csv(tmp_path / "raw_data.csv", sep=";", encoding="utf-8-sig")
        ts_rows = result[pd.to_datetime(result["Date et Heure"], utc=True)
                         == pd.Timestamp(ts)]
        assert len(ts_rows) == 1
        assert ts_rows["Consommation (MW)"].iloc[0] == pytest.approx(50000.0)


# ── _find_gaps ────────────────────────────────────────────────────────────────

class TestFindGaps:
    def test_no_gap_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(etl, "RAW_DIR", tmp_path)
        days = _days(5)
        _write_csv(tmp_path, [_slot(d) for d in days])
        assert etl._find_gaps() == []

    def test_detects_missing_day(self, tmp_path, monkeypatch):
        monkeypatch.setattr(etl, "RAW_DIR", tmp_path)
        days = _days(5)
        missing = days[2]
        _write_csv(tmp_path, [_slot(d) for d in days if d != missing])
        gaps = etl._find_gaps()
        assert len(gaps) == 1
        assert gaps[0][0].isoformat() == missing

    def test_detects_nan_energy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(etl, "RAW_DIR", tmp_path)
        days = _days(5)
        rows = [_slot(d, energy=float("nan") if d == days[1] else 40000.0) for d in days]
        _write_csv(tmp_path, rows)
        gaps = etl._find_gaps()
        assert any(g[0].isoformat() == days[1] for g in gaps)


# ── _check_feature_psi ────────────────────────────────────────────────────────

class TestCheckFeaturePsi:
    def test_skips_when_no_models_dir(self, monkeypatch):
        monkeypatch.setenv("MODELS_DIR", "")
        # Ne doit pas lever d'exception
        etl._check_feature_psi()

    def test_skips_when_distributions_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODELS_DIR", str(tmp_path))
        # Dossier existe mais pas de feature_distributions.pkl
        etl._check_feature_psi()
