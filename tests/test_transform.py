import numpy as np
import pandas as pd
import pytest

from transform import add_lags, prepare_features


def _make_df(dates):
    return pd.DataFrame({
        "Date et Heure":       dates,
        "temperature_2m":      [10.0] * len(dates),
        "apparent_temperature": [8.0] * len(dates),
        "precipitation":       [0.0] * len(dates),
        "cloud_cover":         [50.0] * len(dates),
    })


class TestPrepareFeatures:
    def test_cyclical_columns_present(self):
        df = _make_df(["2024-01-15T12:00:00+00:00"])
        out = prepare_features(df)
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos",
                    "month_sin", "month_cos", "doy_sin", "doy_cos"]:
            assert col in out.columns

    def test_cyclical_values_in_range(self):
        dates = pd.date_range("2024-06-01", periods=48, freq="30min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00")
        out = prepare_features(_make_df(dates))
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert out[col].between(-1, 1).all(), f"{col} hors [-1, 1]"

    def test_holiday_detection(self):
        # 1er janvier = jour férié FR
        df = _make_df(["2024-01-01T10:00:00+00:00", "2024-01-02T10:00:00+00:00"])
        out = prepare_features(df)
        assert out["is_holiday"].iloc[0] == 1
        assert out["is_day_after_holiday"].iloc[1] == 1

    def test_non_holiday(self):
        df = _make_df(["2024-03-15T10:00:00+00:00"])
        out = prepare_features(df)
        assert out["is_holiday"].iloc[0] == 0

    def test_energy_crisis_flag(self):
        df = _make_df(["2022-09-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"])
        out = prepare_features(df)
        assert out["is_energy_crisis"].iloc[0] == 1
        assert out["is_energy_crisis"].iloc[1] == 0

    def test_heating_cooling_degree_days(self):
        df = _make_df(["2024-01-15T12:00:00+00:00"])
        df["apparent_temperature"] = [5.0]  # < 17 → heating, < 21 → no cooling
        out = prepare_features(df)
        assert out["heating_apparent"].iloc[0] == pytest.approx(12.0)
        assert out["cooling_apparent"].iloc[0] == pytest.approx(0.0)

    def test_cooling_degree_days(self):
        df = _make_df(["2024-07-15T12:00:00+00:00"])
        df["apparent_temperature"] = [30.0]  # > 21 → cooling, > 17 → no heating
        out = prepare_features(df)
        assert out["heating_apparent"].iloc[0] == pytest.approx(0.0)
        assert out["cooling_apparent"].iloc[0] == pytest.approx(9.0)

    def test_index_is_utc(self):
        df = _make_df(["2024-06-01T08:00:00+00:00"])
        out = prepare_features(df)
        assert out.index.tzinfo is not None
        assert str(out.index.tz) == "UTC"


class TestAddLags:
    def _make_history(self):
        idx = pd.date_range("2024-01-01", periods=24 * 14, freq="30min", tz="UTC")
        conso = pd.Series(range(len(idx)), index=idx, dtype=float)
        temp  = pd.Series([10.0] * len(idx), index=idx)
        return conso, temp

    def test_lag_h24_correct(self):
        conso, temp = self._make_history()
        slot = pd.Timestamp("2024-01-08T12:00:00", tz="UTC")
        df = pd.DataFrame(index=pd.DatetimeIndex([slot]))
        out = add_lags(df, conso_hist=conso, temp_hist=temp)
        expected = conso.get(slot - pd.Timedelta(hours=24))
        assert out["conso_h24"].iloc[0] == pytest.approx(expected)

    def test_lag_h168_correct(self):
        conso, temp = self._make_history()
        slot = pd.Timestamp("2024-01-08T12:00:00", tz="UTC")
        df = pd.DataFrame(index=pd.DatetimeIndex([slot]))
        out = add_lags(df, conso_hist=conso, temp_hist=temp)
        expected = conso.get(slot - pd.Timedelta(hours=168))
        assert out["conso_h168"].iloc[0] == pytest.approx(expected)

    def test_lag_columns_present(self):
        conso, temp = self._make_history()
        slot = pd.Timestamp("2024-01-08T00:00:00", tz="UTC")
        df = pd.DataFrame(index=pd.DatetimeIndex([slot]))
        out = add_lags(df, conso_hist=conso, temp_hist=temp)
        for col in ["conso_h24", "conso_h48", "conso_h168",
                    "conso_mean_7d", "conso_mean_12w", "conso_mean_52w",
                    "temp_h24", "temp_h48", "temp_h168"]:
            assert col in out.columns

    def test_missing_history_gives_nan(self):
        conso, temp = self._make_history()
        # Slot très proche du début — pas d'historique 168h
        slot = pd.Timestamp("2024-01-01T01:00:00", tz="UTC")
        df = pd.DataFrame(index=pd.DatetimeIndex([slot]))
        out = add_lags(df, conso_hist=conso, temp_hist=temp)
        assert np.isnan(out["conso_h168"].iloc[0])
