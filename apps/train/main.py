import logging
import sys
from pathlib import Path

try:
    from google.cloud.logging.handlers import StructuredLogHandler
    logging.basicConfig(handlers=[StructuredLogHandler()], level=logging.INFO)
except Exception:
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
    run_train()
    log.info("--- Entraînement terminé ---")


if __name__ == "__main__":
    main()
