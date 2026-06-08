import logging
import os
import re

import joblib

log = logging.getLogger(__name__)

_MODEL_PATTERN    = re.compile(r"^model_(?!.+_(p10|p90)_).+_\d{8}_\d+\.pkl$")
_P10_PATTERN      = re.compile(r"^model_.+_p10_\d{8}_\d+\.pkl$")
_P90_PATTERN      = re.compile(r"^model_.+_p90_\d{8}_\d+\.pkl$")


def list_models(models_dir: str) -> list[str]:
    return sorted(
        [f for f in os.listdir(models_dir) if _MODEL_PATTERN.match(f)],
        reverse=True,
    )


def load_latest_model(models_dir: str = "../outputs"):
    models = list_models(models_dir)
    if not models:
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier.")
    latest = models[0]
    log.info(f"Modèle chargé : {latest}")
    return joblib.load(os.path.join(models_dir, latest)), latest


def load_quantile_models(models_dir: str):
    files = os.listdir(models_dir)
    p10 = sorted([f for f in files if _P10_PATTERN.match(f)], reverse=True)
    p90 = sorted([f for f in files if _P90_PATTERN.match(f)], reverse=True)
    if not p10 or not p90:
        raise FileNotFoundError("Modèles quantile p10/p90 introuvables.")
    log.info(f"Modèles quantile chargés : {p10[0]}, {p90[0]}")
    return joblib.load(os.path.join(models_dir, p10[0])), joblib.load(os.path.join(models_dir, p90[0]))