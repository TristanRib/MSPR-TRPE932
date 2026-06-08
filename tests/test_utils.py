import pytest

from utils import list_models, load_latest_model, load_quantile_models


class TestListModels:
    def test_empty_dir(self, tmp_path):
        assert list_models(str(tmp_path)) == []

    def test_excludes_quantile_files(self, tmp_path):
        (tmp_path / "model_xgboost_p10_20260606_1.pkl").touch()
        (tmp_path / "model_xgboost_p90_20260606_1.pkl").touch()
        assert list_models(str(tmp_path)) == []

    def test_includes_main_model(self, tmp_path):
        (tmp_path / "model_xgboost_20260606_1.pkl").touch()
        result = list_models(str(tmp_path))
        assert result == ["model_xgboost_20260606_1.pkl"]

    def test_sorted_most_recent_first(self, tmp_path):
        (tmp_path / "model_xgboost_20260601_1.pkl").touch()
        (tmp_path / "model_xgboost_20260606_1.pkl").touch()
        (tmp_path / "model_xgboost_20260603_2.pkl").touch()
        result = list_models(str(tmp_path))
        assert result[0] == "model_xgboost_20260606_1.pkl"

    def test_old_naming_still_matched(self, tmp_path):
        # L'ancien format YYYYMMDD_HHMMSS (ex: 20260601_143022) est toujours
        # matché par \d{8}_\d+ — compatibilité avec les modèles déjà en GCS.
        (tmp_path / "model_xgboost_20260601_143022.pkl").touch()
        result = list_models(str(tmp_path))
        assert "model_xgboost_20260601_143022.pkl" in result

    def test_excludes_non_pkl(self, tmp_path):
        (tmp_path / "model_xgboost_20260606_1.txt").touch()
        assert list_models(str(tmp_path)) == []

    def test_load_latest_raises_if_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest_model(str(tmp_path))

    def test_load_quantile_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_quantile_models(str(tmp_path))
