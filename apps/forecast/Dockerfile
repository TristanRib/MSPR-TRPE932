FROM python:3.11-slim

WORKDIR /app

COPY apps/forecast/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/forecast/main.py .

CMD ["python", "main.py"]