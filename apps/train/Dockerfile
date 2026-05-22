FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY apps/train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/train/main.py .
COPY scripts/transform.py .
COPY scripts/train.py .

CMD ["python", "main.py"]
