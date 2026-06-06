import sys
from pathlib import Path

# Rend scripts/ et apps/api/ importables
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api"))
