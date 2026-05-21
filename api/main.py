from fastapi import FastAPI
import numpy as np
import os

from utils import load_latest_model

app = FastAPI()

MODELS_DIR = os.getenv("MODELS_DIR", "../outputs")
model, model_name = load_latest_model(MODELS_DIR)

@app.get("/health")
def health():
    """Endpoint utile pour Cloud Run pour vérifier que l'API tourne."""
    return {"status": "ok", "model_loaded": model_name}

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return {
        "prediction": float(prediction[0]),
        "model_used": model_name
    }

@app.post("/reload")
def reload_model():
    global model, model_name
    model, model_name = load_latest_model(MODELS_DIR)
    return {"status": "reloaded", "model_loaded": model_name}

@app.get("/model/info")
def model_info():
    """Endpoint dédié au tracking — utile pour ta MSPR."""
    return {
        "model_name": model_name,
        "models_available": os.listdir(MODELS_DIR)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)