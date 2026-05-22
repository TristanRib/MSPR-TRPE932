import sys
from pathlib import Path

_here = Path(__file__).parent
if not (_here / "transform.py").exists():
    sys.path.insert(0, str(_here.parent.parent / "scripts"))

from transform import main as run_transform
from train import main as run_train


def main():
    print("--- Job entraînement ---")
    run_transform()
    try:
        run_train()
    except RuntimeError as e:
        print(f"WARN : qualité insuffisante, ancien modèle conservé.\n{e}")
    print("--- Entraînement terminé ---")


if __name__ == "__main__":
    main()
