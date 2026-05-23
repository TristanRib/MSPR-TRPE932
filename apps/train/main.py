import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

from transform import main as run_transform
from train import main as run_train


def main():
    log.info("--- Job entraînement ---")
    run_transform()
    try:
        run_train()
    except RuntimeError as e:
        log.warning(f"Qualité insuffisante, ancien modèle conservé.\n{e}")
    log.info("--- Entraînement terminé ---")


if __name__ == "__main__":
    main()
