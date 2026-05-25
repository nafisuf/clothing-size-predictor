# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import Dict, List

app = FastAPI(title="Clothing Size Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Izinkan semua origin (untuk development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load model ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'src/models/knn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'src/models/scaler.pkl'))
classes = np.load(os.path.join(BASE_DIR, 'src/models/classes.npy'), allow_pickle=True)

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Kurus"
    if bmi < 25: return "Normal"
    if bmi < 30: return "Gemuk"
    return "Obesitas"

def get_age_group(age: int) -> str:
    if age < 25: return "Muda (18-24)"
    if age < 40: return "Dewasa (25-39)"
    if age < 60: return "Paruh baya (40-59)"
    return "Lansia (60+)"

class Measurement(BaseModel):
    height: float
    weight: float
    age: int

class PredictionResponse(BaseModel):
    predicted_size: str
    confidence: float
    all_probabilities: Dict[str, float]
    body_analysis: Dict

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": {"type": "KNN", "k": model.n_neighbors, "accuracy": 0.9857}
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: Measurement):
    # Validasi input
    if not (140 <= data.height <= 210):
        raise HTTPException(400, "Tinggi harus 140-210 cm")
    if not (35 <= data.weight <= 140):
        raise HTTPException(400, "Berat harus 35-140 kg")
    if not (18 <= data.age <= 80):
        raise HTTPException(400, "Usia harus 18-80 tahun")

    X = np.array([[data.height, data.weight, data.age]])
    X_scaled = scaler.transform(X)

    # Dapatkan probabilitas dari model
    proba = model.predict_proba(X_scaled)[0]
    # Dapatkan prediksi kelas dengan probabilitas tertinggi
    pred_index = np.argmax(proba)
    pred_label = model.classes_[pred_index]
    confidence = proba[pred_index]

    # Buat dictionary probabilitas untuk semua kelas
    probs = {}
    for i, cls in enumerate(model.classes_):
        probs[cls] = float(proba[i])

    # Hitung BMI
    bmi = data.weight / ((data.height/100)**2)
    body_analysis = {
        "bmi": round(bmi, 2),
        "bmi_category": get_bmi_category(bmi),
        "age_group": get_age_group(data.age)
    }

    return {
        "predicted_size": pred_label,
        "confidence": round(confidence, 4),
        "all_probabilities": probs,
        "body_analysis": body_analysis
    }

@app.post("/predict/batch")
def predict_batch(measurements: List[Measurement]):
    results = []
    for m in measurements:
        try:
            results.append(predict(m).dict())
        except HTTPException as e:
            results.append({"error": e.detail, "input": m.dict()})
    return {"batch_results": results}