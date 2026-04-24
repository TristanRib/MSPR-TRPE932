FROM python:3.11-slim

WORKDIR /app

COPY etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY etl/etl_pipeline.py .
COPY scripts/transform.py .
COPY scripts/train.py .

CMD ["python", "etl_pipeline.py"]
