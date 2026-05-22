import os
import re
from datetime import datetime, timezone
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

RUN_NOISE_R2       = 0.005
RUN_NOISE_MAPE     = 0.005
RUN_NOISE_ACCURACY = 0.01


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
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_split=5,
            min_samples_leaf=1, max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=50, weights="distance", metric="manhattan", n_jobs=-1)),
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            random_state=42, n_jobs=-1,
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


def _load_prev_best_from_bq() -> dict | None:
    bq_table = os.getenv("BQ_METRICS_TABLE", "")
    if not bq_table:
        return None
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        query = f"""
            SELECT r2, mape, accuracy_5pct
            FROM `{bq_table}`
            WHERE run_at = (SELECT MAX(run_at) FROM `{bq_table}`)
              AND model_name = best_model
            LIMIT 1
        """
        rows = list(client.query(query).result())
        return dict(rows[0]) if rows else None
    except Exception as e:
        print(f"WARN : lecture BQ pour check_quality échouée : {e}")
        return None


def check_quality(metrics: dict):
    prev = _load_prev_best_from_bq()
    if prev is None:
        return

    errors = []
    if metrics["R2"] < prev["r2"] - RUN_NOISE_R2:
        errors.append(f"Régression R² : {metrics['R2']:.4f} < {prev['r2']:.4f} - {RUN_NOISE_R2}")
    if metrics["MAPE"] > prev["mape"] + RUN_NOISE_MAPE:
        errors.append(f"Régression MAPE : {metrics['MAPE']:.4f} > {prev['mape']:.4f} + {RUN_NOISE_MAPE}")
    if metrics["Accuracy (±5%)"] < prev["accuracy_5pct"] - RUN_NOISE_ACCURACY:
        errors.append(f"Régression Accuracy : {metrics['Accuracy (±5%)']:.4f} < {prev['accuracy_5pct']:.4f} - {RUN_NOISE_ACCURACY}")

    if errors:
        raise RuntimeError("Contrôle qualité échoué :\n" + "\n".join(f"  - {e}" for e in errors))


def save_metrics_to_bq(results: list[dict], best_name: str):
    bq_table = os.getenv("BQ_METRICS_TABLE", "")
    if not bq_table:
        print("WARN : BQ_METRICS_TABLE non défini, push BigQuery ignoré")
        return
    try:
        from google.cloud import bigquery

        run_at = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "run_at":        run_at,
                "best_model":    best_name,
                "model_name":    r["Model"],
                "r2":            r["R2"],
                "rmse":          r["RMSE"],
                "mape":          r["MAPE"],
                "accuracy_5pct": r["Accuracy (±5%)"],
            }
            for r in results
        ]
        client = bigquery.Client()
        errors = client.insert_rows_json(bq_table, rows)
        if errors:
            print(f"WARN : BigQuery insert errors : {errors}")
        else:
            print(f"Métriques pushées vers BigQuery ({len(rows)} lignes)")
    except Exception as e:
        print(f"WARN : push BigQuery échoué : {e}")


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

    best_row     = results_df.iloc[0]
    best_name    = best_row["Model"]
    best_metrics = best_row.drop("Model").to_dict()

    # check_quality(best_metrics)

    save_model(models[best_name], best_name)
    save_metrics_to_bq(results, best_name)
    print(f"Qualité validée — R²={best_metrics['R2']:.4f}, MAPE={best_metrics['MAPE']:.4f}")


if __name__ == "__main__":
    main()