FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY etl/etl_pipeline.py .
COPY scripts/transform.py .
COPY scripts/train.py .
COPY scripts/datacard.py .

CMD ["python", "etl_pipeline.py"]
