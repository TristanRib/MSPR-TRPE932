import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT / "data")))
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    str(ROOT / "outputs")))

RMSE_FLOOR    = 1600.0  # MW — seuil absolu (20% au-dessus de la baseline)
R2_FLOOR      = 0.97    # seuil absolu
MAPE_FLOOR    = 0.025   # seuil absolu (2.5%, baseline ~1.9%)
RMSE_NOISE    = 30.0    # MW — tolérance vs meilleur historique
R2_NOISE      = 0.002   # tolérance vs meilleur historique
ACC_NOISE     = 0.002   # tolérance vs meilleur historique (±5% accuracy)
MAPE_NOISE    = 0.001   # tolérance vs meilleur historique

MODEL_NAME       = "XGBoost"
MODEL_FILE_STEM  = "model_xgboost"
MODEL_PARAMS = dict(
    n_estimators=978, learning_rate=0.046, max_depth=6,
    subsample=0.75, colsample_bytree=0.93, min_child_weight=5,
    gamma=4.0, reg_alpha=0.65, reg_lambda=2.65,
    random_state=42, n_jobs=-1,
)


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "transformed_data.csv")
    X = df.drop(columns=["Consommation"])
    y = df["Consommation"]
    n = len(df)
    split_tr  = int(n * 0.70)
    split_cal = int(n * 0.80)
    return (
        X.iloc[:split_tr], X.iloc[split_tr:split_cal], X.iloc[split_cal:],
        y.iloc[:split_tr], y.iloc[split_tr:split_cal], y.iloc[split_cal:],
    )


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "R2":             r2_score(y_true, y_pred),
        "RMSE":           np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE":           mean_absolute_percentage_error(y_true, y_pred),
        "Accuracy (±5%)": float(np.mean(np.abs((y_true - y_pred) / y_true) < 0.05)),
    }


def _load_best_from_bq() -> dict | None:
    bq_table = os.getenv("BQ_METRICS_TABLE", "")
    if not bq_table:
        return None
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        query = f"""
            SELECT r2, rmse, mape, accuracy_5pct
            FROM `{bq_table}`
            WHERE model_name = '{MODEL_NAME}'
              AND rmse = (SELECT MIN(rmse) FROM `{bq_table}` WHERE model_name = '{MODEL_NAME}')
            LIMIT 1
        """
        rows = list(client.query(query).result())
        return dict(rows[0]) if rows else None
    except Exception as e:
        log.warning(f"Lecture BQ pour check_quality échouée : {e}")
        return None


def check_quality(metrics: dict):
    # Niveau 1 — seuils absolus, toujours vérifiés
    hard_errors = []
    if metrics["RMSE"] > RMSE_FLOOR:
        hard_errors.append(f"RMSE {metrics['RMSE']:.0f} MW > seuil absolu {RMSE_FLOOR:.0f} MW")
    if metrics["R2"] < R2_FLOOR:
        hard_errors.append(f"R² {metrics['R2']:.4f} < seuil absolu {R2_FLOOR}")
    if metrics["MAPE"] > MAPE_FLOOR:
        hard_errors.append(f"MAPE {metrics['MAPE']:.4f} > seuil absolu {MAPE_FLOOR}")
    if hard_errors:
        raise RuntimeError("Contrôle qualité — seuils absolus franchis :\n" + "\n".join(f"  - {e}" for e in hard_errors))

    # Niveau 2 — régression vs meilleur historique
    best = _load_best_from_bq()
    if best is None:
        log.info("check_quality : aucune baseline BQ, seuils absolus validés uniquement")
        return
    reg_errors = []
    if metrics["RMSE"] > best["rmse"] + RMSE_NOISE:
        reg_errors.append(f"RMSE {metrics['RMSE']:.0f} MW > meilleur {best['rmse']:.0f} + {RMSE_NOISE:.0f} MW")
    if metrics["R2"] < best["r2"] - R2_NOISE:
        reg_errors.append(f"R² {metrics['R2']:.4f} < meilleur {best['r2']:.4f} - {R2_NOISE}")
    if metrics["Accuracy (±5%)"] < best["accuracy_5pct"] - ACC_NOISE:
        reg_errors.append(f"Acc±5% {metrics['Accuracy (±5%)']:.4f} < meilleur {best['accuracy_5pct']:.4f} - {ACC_NOISE}")
    if metrics["MAPE"] > best["mape"] + MAPE_NOISE:
        reg_errors.append(f"MAPE {metrics['MAPE']:.4f} > meilleur {best['mape']:.4f} + {MAPE_NOISE}")
    if reg_errors:
        raise RuntimeError("Contrôle qualité — régression vs meilleur modèle :\n" + "\n".join(f"  - {e}" for e in reg_errors))


def save_metrics_to_bq(metrics: dict):
    bq_table = os.getenv("BQ_METRICS_TABLE", "")
    if not bq_table:
        log.warning("BQ_METRICS_TABLE non défini, push BigQuery ignoré")
        return
    try:
        from google.cloud import bigquery

        rows = [{
            "run_at":        datetime.now(timezone.utc).isoformat(),
            "best_model":    MODEL_NAME,
            "model_name":    MODEL_NAME,
            "r2":            metrics["R2"],
            "rmse":          metrics["RMSE"],
            "mape":          metrics["MAPE"],
            "accuracy_5pct": metrics["Accuracy (±5%)"],
        }]
        client = bigquery.Client()
        errors = client.insert_rows_json(bq_table, rows)
        if errors:
            log.warning(f"BigQuery insert errors : {errors}")
        else:
            log.info(f"Métriques pushées vers BigQuery")
    except Exception as e:
        log.warning(f"Push BigQuery échoué : {e}")


def main():
    X_train, X_calib, X_test, y_train, y_calib, y_test = load_data()
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    MODELS_DIR.mkdir(exist_ok=True)

    imputer = KNNImputer(n_neighbors=17, weights="distance")
    imputer.fit(X_train.iloc[-35040:].to_numpy())  # 2 dernières années (35040 slots 30-min)
    X_train = imputer.transform(X_train.to_numpy())
    X_calib = imputer.transform(X_calib.to_numpy())
    X_test  = imputer.transform(X_test.to_numpy())
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl", compress=3)
    log.info("imputer.pkl sauvegardé")

    sample_weight = np.exp(np.linspace(0, 2, len(X_train)))

    log.info(f"Training {MODEL_NAME} (MSE)...")
    model = XGBRegressor(objective="reg:squarederror", **MODEL_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    metrics = compute_metrics(y_test, model.predict(X_test))
    log.info(f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.1f}  MAPE={metrics['MAPE']:.4f}  Acc±5%={metrics['Accuracy (±5%)']:.4f}")

    check_quality(metrics)

    joblib.dump(model, MODELS_DIR / f"{MODEL_FILE_STEM}_{date_str}.pkl", compress=3)
    log.info(f"{MODEL_FILE_STEM}_{date_str}.pkl sauvegardé")

    quantile_models = {}
    for alpha, suffix in [(0.1, "p10"), (0.9, "p90")]:
        log.info(f"Training quantile {suffix}...")
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, **MODEL_PARAMS)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        joblib.dump(m, MODELS_DIR / f"{MODEL_FILE_STEM}_{suffix}_{date_str}.pkl", compress=3)
        log.info(f"{MODEL_FILE_STEM}_{suffix}_{date_str}.pkl sauvegardé")
        quantile_models[suffix] = m
    
    log.info("Calibration CQR...")
    p10_cal = quantile_models["p10"].predict(X_calib)
    p90_cal = quantile_models["p90"].predict(X_calib)
    scores  = np.maximum(p10_cal - y_calib.values, y_calib.values - p90_cal)
    q       = float(np.quantile(scores, 0.80))
    joblib.dump(q, MODELS_DIR / "cqr_correction.pkl")
    log.info(f"CQR q={q:.1f} MW sauvegardé")

    save_metrics_to_bq(metrics)


if __name__ == "__main__":
    main()
