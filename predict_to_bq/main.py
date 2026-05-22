import os
from datetime import datetime, timezone

import functions_framework
import requests
from google.cloud import bigquery

API_URL = os.environ["API_URL"]
BQ_TABLE = os.environ["BQ_TABLE"]


@functions_framework.http
def fetch_and_store(request):
    preds = requests.get(f"{API_URL}/predict", timeout=30).json()

    run_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "run_at":        run_at,
            "predicted_at":  p["datetime"],
            "prediction_mw": p["prediction_mw"],
        }
        for p in preds
    ]

    client = bigquery.Client()
    errors = client.insert_rows_json(BQ_TABLE, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

    return f"OK — {len(rows)} prédictions insérées ({run_at})", 200