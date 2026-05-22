import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).parent.parent

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT / "data")))
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    str(ROOT / "outputs")))


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "transformed_data.csv")
    X = df.drop(columns=["Consommation"])
    y = df["Consommation"]
    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def build_models() -> dict:
    return {
        "RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("rbf", RBFSampler(gamma=0.1, n_components=500)),
            ("ridge", Ridge()),
        ]),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        ),
    }


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "R2":             r2_score(y_true, y_pred),
        "RMSE":           np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE":           mean_absolute_percentage_error(y_true, y_pred),
        "Accuracy (±5%)": float(np.mean(np.abs((y_true - y_pred) / y_true) < 0.05)),
    }


def save_model(model, name: str):
    MODELS_DIR.mkdir(exist_ok=True)
    base = f"mspr_edf_{name.lower().replace(' ', '_')}"
    pattern = re.compile(rf"{re.escape(base)}_(\d+)\.pkl")
    numbers = [int(m.group(1)) for f in os.listdir(MODELS_DIR) if (m := pattern.match(f))]
    next_num = max(numbers) + 1 if numbers else 1
    path = MODELS_DIR / f"{base}_{next_num:02d}.pkl"
    joblib.dump(model, path, compress=3)
    print(f"Modèle sauvegardé : {path.name}")


def main():
    X_train, X_test, y_train, y_test = load_data()
    models = build_models()
    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = compute_metrics(y_test, model.predict(X_test))
        results.append({"Model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n" + results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    save_model(models[best_name], best_name)


if __name__ == "__main__":
    main()
