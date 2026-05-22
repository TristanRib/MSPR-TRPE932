import json
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

# Planchers absolus : utilisés uniquement au premier run (pas d'historique)
FLOOR_R2       = 0.80
FLOOR_MAPE     = 0.10
FLOOR_ACCURACY = 0.70

# Bruit acceptable entre deux runs consécutifs (variation quotidienne des données)
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


def _extract_best_metrics(run: dict) -> dict:
    return run["all_models"][run["best_model"]]


def check_quality(metrics: dict, history: list[dict]):
    """
    Premier run  → planchers absolus (FLOOR_*).
    Runs suivants → plancher = meilleur modèle du run précédent ± bruit toléré (RUN_NOISE_*).
    """
    errors = []

    if not history:
        if metrics["R2"] < FLOOR_R2:
            errors.append(f"R² trop bas : {metrics['R2']:.4f} < {FLOOR_R2}")
        if metrics["MAPE"] > FLOOR_MAPE:
            errors.append(f"MAPE trop élevé : {metrics['MAPE']:.4f} > {FLOOR_MAPE}")
        if metrics["Accuracy (±5%)"] < FLOOR_ACCURACY:
            errors.append(f"Accuracy trop basse : {metrics['Accuracy (±5%)']:.4f} < {FLOOR_ACCURACY}")
    else:
        prev = _extract_best_metrics(history[-1])
        if metrics["R2"] < prev["R2"] - RUN_NOISE_R2:
            errors.append(f"Régression R² : {metrics['R2']:.4f} < {prev['R2']:.4f} - {RUN_NOISE_R2}")
        if metrics["MAPE"] > prev["MAPE"] + RUN_NOISE_MAPE:
            errors.append(f"Régression MAPE : {metrics['MAPE']:.4f} > {prev['MAPE']:.4f} + {RUN_NOISE_MAPE}")
        if metrics["Accuracy (±5%)"] < prev["Accuracy (±5%)"] - RUN_NOISE_ACCURACY:
            errors.append(f"Régression Accuracy : {metrics['Accuracy (±5%)']:.4f} < {prev['Accuracy (±5%)']:.4f} - {RUN_NOISE_ACCURACY}")

    if errors:
        raise RuntimeError("Contrôle qualité échoué :\n" + "\n".join(f"  - {e}" for e in errors))


def save_metrics_history(results: list[dict], best_name: str):
    MODELS_DIR.mkdir(exist_ok=True)
    history_path = MODELS_DIR / "metrics_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    entry = {
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "all_models": {r["Model"]: {k: v for k, v in r.items() if k != "Model"} for r in results},
    }
    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2))
    print(f"metrics_history.json mis à jour ({len(history)} runs)")


def push_model_metrics(metrics: dict, model_name: str):
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return

    try:
        import time
        from google.cloud import monitoring_v3

        client   = monitoring_v3.MetricServiceClient()
        now      = time.time()
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": int(now), "nanos": int((now % 1) * 1e9)}}
        )

        series_list = []
        for key, value in {
            "r2":            metrics["R2"],
            "rmse":          metrics["RMSE"],
            "mape":          metrics["MAPE"],
            "accuracy_5pct": metrics["Accuracy (±5%)"],
        }.items():
            s = monitoring_v3.TimeSeries()
            s.metric.type = f"custom.googleapis.com/model/{key}"
            s.metric.labels["model_name"] = model_name
            s.resource.type = "global"
            s.points = [monitoring_v3.Point({"interval": interval, "value": {"double_value": value}})]
            series_list.append(s)

        client.create_time_series(name=f"projects/{project_id}", time_series=series_list)
        print(f"Métriques pushées vers Cloud Monitoring ({project_id})")
    except Exception as e:
        print(f"WARN : push Cloud Monitoring échoué : {e}")


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

    history_path = MODELS_DIR / "metrics_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    check_quality(best_metrics, history)

    # if not history or best_metrics["R2"] > _extract_best_metrics(history[-1])["R2"]:
    save_model(models[best_name], best_name)

    save_metrics_history(results, best_name)
    push_model_metrics(best_metrics, best_name)
    print(f"Qualité validée — R²={best_metrics['R2']:.4f}, MAPE={best_metrics['MAPE']:.4f}")


if __name__ == "__main__":
    main()
