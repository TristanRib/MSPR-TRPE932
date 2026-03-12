# MSPR TRPE932

| Busin Thomas | Riboulet-Depret Tristan | Van-Duysen Nicolas |
|--------------|-------------------------|--------------------|

## Overview


---

## Dataset


---

## Methodology

1. Import et analyse exploratoire des données (EDA)
2. Feature engineering (extraction de tendances temporelles, clustering géographique)
3. Sélection et entraînement de modèles de prédiction
4. Évaluation des métriques (MAE, RMSE, R²)

---

## Project Structure

### [`/notebooks`](./notebooks)

Notebooks Jupyter organisés par phase du pipeline :

- **Exploration** : analyse exploratoire des séismes, distributions, patterns temporels et spatiaux.
- **Transformation** : prétraitement des variables, traitement des valeurs manquantes, feature engineering.
- **Training** : entraînement de modèles, validation croisée, comparaisons de performance.

### [`/models`](./models)

Contient des *model cards* documentant chaque modèle :

- Type et architecture du modèle
- Hyperparamètres
- Performance (mae, mse, r2, etc.)
- Conclusions et recommandations

### [`/data`](./data)

Contient :

- *Data cards* décrivant les datasets utilisés
- [transformed_data.csv](./data/transformed_data.csv) : version transformée prête à l’usage pour entraînement

---

## Usage

1. Cloner le dépôt
2. Installer les dépendances : `pip install -r requirements.txt`
3. Exécuter les notebooks dans l’ordre :
    - [01_exploration.ipynb](./notebooks/01_exploration.ipynb)
    - [02_transformation.ipynb](./notebooks/02_transformation.ipynb)
    - [03_training.ipynb](./notebooks/03_training.ipynb)

---

## Reproducibility

- Version Python recommandée : `>=3.9`
