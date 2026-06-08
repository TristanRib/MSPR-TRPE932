import logging
import os
from datetime import datetime, timezone

import requests
from google.cloud import bigquery

try:
    from google.cloud.logging.handlers import StructuredLogHandler
    logging.basicConfig(handlers=[StructuredLogHandler()], level=logging.INFO)
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

API_URL = os.environ["API_URL"]
BQ_TABLE = os.environ["BQ_TABLE"]


def main():
    preds = requests.get(f"{API_URL}/predict", timeout=120).json()

    run_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "run_at":         run_at,
            "predicted_at":   p["datetime"],
            "prediction_mw":  p["prediction_mw"],
            "prediction_p10": p["prediction_p10"],
            "prediction_p90": p["prediction_p90"],
        }
        for p in preds
    ]

    wide = [r for r in rows if r["prediction_p90"] - r["prediction_p10"] > 5000]
    if wide:
        log.warning(f"{len(wide)}/{len(rows)} slots avec intervalle p10-p90 > 5000 MW")

    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_json(rows, BQ_TABLE, job_config=job_config).result()

    log.info(f"{len(rows)} prédictions insérées ({run_at})")


if __name__ == "__main__":
    main()