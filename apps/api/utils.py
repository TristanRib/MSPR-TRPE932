import logging
import os
import re

import joblib

log = logging.getLogger(__name__)

_MODEL_PATTERN = re.compile(r"^model_(.+)_(\d{8}_\d{6})\.pkl$")


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
    path = os.path.join(models_dir, latest)
    log.info(f"Modèle chargé : {latest}")
    return joblib.load(path), latest