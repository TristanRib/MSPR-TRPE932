import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import train as tr


GOOD_METRICS = {
    "R2":             0.985,
    "RMSE":           1300.0,
    "MAPE":           0.019,
    "Accuracy (±5%)": 0.945,
}

BEST_BQ = {
    "r2":            0.984,
    "rmse":          1320.0,
    "mape":          0.020,
    "accuracy_5pct": 0.943,
}


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([40000.0, 42000.0, 38000.0])
        m = tr.compute_metrics(y, y)
        assert m["R2"] == pytest.approx(1.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["MAPE"] == pytest.approx(0.0)
        assert m["Accuracy (±5%)"] == pytest.approx(1.0)

    def test_all_keys_present(self):
        y = np.array([40000.0, 42000.0])
        m = tr.compute_metrics(y, y + 500)
        assert set(m.keys()) == {"R2", "RMSE", "MAPE", "Accuracy (±5%)"}

    def test_rmse_positive(self):
        y    = np.array([40000.0, 42000.0])
        pred = np.array([41000.0, 41000.0])
        m = tr.compute_metrics(y, pred)
        assert m["RMSE"] > 0


class TestCheckQuality:
    def test_passes_when_no_baseline(self):
        with patch.object(tr, "_load_best_from_bq", return_value=None):
            assert tr.check_quality(GOOD_METRICS) is True

    def test_fails_rmse_absolute(self):
        bad = {**GOOD_METRICS, "RMSE": tr.RMSE_FLOOR + 1}
        with patch.object(tr, "_load_best_from_bq", return_value=None):
            assert tr.check_quality(bad) is False

    def test_fails_r2_absolute(self):
        bad = {**GOOD_METRICS, "R2": tr.R2_FLOOR - 0.01}
        with patch.object(tr, "_load_best_from_bq", return_value=None):
            assert tr.check_quality(bad) is False

    def test_fails_mape_absolute(self):
        bad = {**GOOD_METRICS, "MAPE": tr.MAPE_FLOOR + 0.001}
        with patch.object(tr, "_load_best_from_bq", return_value=None):
            assert tr.check_quality(bad) is False

    def test_passes_with_baseline(self):
        with patch.object(tr, "_load_best_from_bq", return_value=BEST_BQ):
            assert tr.check_quality(GOOD_METRICS) is True

    def test_fails_rmse_regression(self):
        bad = {**GOOD_METRICS, "RMSE": BEST_BQ["rmse"] + tr.RMSE_NOISE + 1}
        with patch.object(tr, "_load_best_from_bq", return_value=BEST_BQ):
            assert tr.check_quality(bad) is False

    def test_fails_r2_regression(self):
        bad = {**GOOD_METRICS, "R2": BEST_BQ["r2"] - tr.R2_NOISE - 0.001}
        with patch.object(tr, "_load_best_from_bq", return_value=BEST_BQ):
            assert tr.check_quality(bad) is False

    def test_fails_acc_regression(self):
        bad = {**GOOD_METRICS, "Accuracy (±5%)": BEST_BQ["accuracy_5pct"] - tr.ACC_NOISE - 0.001}
        with patch.object(tr, "_load_best_from_bq", return_value=BEST_BQ):
            assert tr.check_quality(bad) is False

    def test_fails_mape_regression(self):
        bad = {**GOOD_METRICS, "MAPE": BEST_BQ["mape"] + tr.MAPE_NOISE + 0.001}
        with patch.object(tr, "_load_best_from_bq", return_value=BEST_BQ):
            assert tr.check_quality(bad) is False


class TestRunTag:
    def test_first_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "MODELS_DIR", tmp_path)
        tag = tr._run_tag("20260606")
        assert tag == "20260606_1"

    def test_increments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "MODELS_DIR", tmp_path)
        (tmp_path / "model_xgboost_20260606_1.pkl").touch()
        tag = tr._run_tag("20260606")
        assert tag == "20260606_2"

    def test_skips_quantile_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "MODELS_DIR", tmp_path)
        (tmp_path / "model_xgboost_p10_20260606_1.pkl").touch()
        (tmp_path / "model_xgboost_p90_20260606_1.pkl").touch()
        tag = tr._run_tag("20260606")
        assert tag == "20260606_1"

    def test_tag_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "MODELS_DIR", tmp_path)
        tag = tr._run_tag("20260606")
        assert re.match(r"^\d{8}_\d+$", tag)
