import os
from datetime import datetime, timezone

import requests
from google.cloud import bigquery

API_URL = os.environ["API_URL"]
BQ_TABLE = os.environ["BQ_TABLE"]


def main():
    preds = requests.get(f"{API_URL}/predict", timeout=120).json()

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
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    client.load_table_from_json(rows, BQ_TABLE, job_config=job_config).result()

    print(f"OK — {len(rows)} prédictions insérées ({run_at})")


if __name__ == "__main__":
    main()