import os, joblib, re

def load_latest_model(models_dir: str = "../model_dumps"):
    """Charge automatiquement le modèle avec le numéro de version le plus élevé."""

    # Lister tous les fichiers .pkl dans le dossier
    files = os.listdir(models_dir)

    # Filtrer uniquement les fichiers qui matchent le pattern model_XX.pkl
    pattern = re.compile(r"\S+_(\d+).pkl")
    model_files = [(f, int(m.group(1))) for f in files if (m := pattern.match(f))]

    if not model_files:
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier.")

    # Trier par numéro et prendre le plus grand
    latest_file = max(model_files, key=lambda x: x[1])
    latest_path = os.path.join(models_dir, latest_file[0])

    print(f"Modèle chargé : {latest_file[0]}")
    return joblib.load(latest_path), latest_file[0]