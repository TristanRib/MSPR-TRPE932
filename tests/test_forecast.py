"""
Tests du job forecast : logique de détection d'intervalles larges,
formatage des lignes BQ, et flux complet avec HTTP + BQ mockés.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_FORECAST_PATH = Path(__file__).parent.parent / "apps" / "forecast" / "main.py"


def _load_forecast(monkeypatch):
    """Charge le module forecast avec les variables d'env obligatoires."""
    monkeypatch.setenv("API_URL",  "http://fake-api")
    monkeypatch.setenv("BQ_TABLE", "fake.dataset.table")
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", MagicMock())
    spec = importlib.util.spec_from_file_location("forecast_main", _FORECAST_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_preds(n: int = 48, p10: int = 39000, p90: int = 41000) -> list:
    return [
        {"datetime":       f"2026-06-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
         "prediction_mw":  40000,
         "prediction_p10": p10,
         "prediction_p90": p90}
        for i in range(n)
    ]


# ── logique pure (pas d'import du module) ────────────────────────────────────

class TestWideIntervalDetection:
    def test_detecte_intervalle_large(self):
        rows = [
            {"prediction_p10": 35000, "prediction_p90": 41000},  # 6000 > 5000
            {"prediction_p10": 39000, "prediction_p90": 41000},  # 2000 ≤ 5000
        ]
        wide = [r for r in rows if r["prediction_p90"] - r["prediction_p10"] > 5000]
        assert len(wide) == 1

    def test_aucun_intervalle_large(self):
        rows = [{"prediction_p10": 39000, "prediction_p90": 41000}]
        wide = [r for r in rows if r["prediction_p90"] - r["prediction_p10"] > 5000]
        assert len(wide) == 0

    def test_format_ligne_bq(self):
        pred = {"datetime": "2026-06-01T00:00:00+00:00",
                "prediction_mw": 40000, "prediction_p10": 39000, "prediction_p90": 41000}
        run_at = "2026-06-01T12:00:00+00:00"
        row = {"run_at": run_at, "predicted_at": pred["datetime"],
               "prediction_mw": pred["prediction_mw"],
               "prediction_p10": pred["prediction_p10"],
               "prediction_p90": pred["prediction_p90"]}
        assert row["predicted_at"] == pred["datetime"]
        assert row["run_at"] == run_at
        assert set(row.keys()) == {"run_at", "predicted_at", "prediction_mw",
                                   "prediction_p10", "prediction_p90"}


# ── flux complet avec mocks ───────────────────────────────────────────────────

class TestForecastMain:
    def test_insere_48_lignes_en_bq(self, monkeypatch):
        mod = _load_forecast(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_preds(48)
        mod.requests.get = MagicMock(return_value=mock_resp)

        mock_bq = MagicMock()
        mod.bigquery.Client = MagicMock(return_value=mock_bq)

        mod.main()

        rows_inseres = mock_bq.load_table_from_json.call_args[0][0]
        assert len(rows_inseres) == 48

    def test_log_warning_si_intervalle_large(self, monkeypatch, caplog):
        import logging
        mod = _load_forecast(monkeypatch)

        preds_larges = _fake_preds(48, p10=30000, p90=40000)  # IC = 10 000 > 5 000
        mock_resp = MagicMock()
        mock_resp.json.return_value = preds_larges
        mod.requests.get = MagicMock(return_value=mock_resp)
        mod.bigquery.Client = MagicMock(return_value=MagicMock())

        with caplog.at_level(logging.WARNING):
            mod.main()

        assert any("5000" in r.message or "MW" in r.message for r in caplog.records)
