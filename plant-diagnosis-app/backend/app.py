"""
FastAPI backend serving the multi-task Swin plant diagnosis model.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import load_labels, predict
from model import load_model

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model_swin.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Plant Diagnosis API", version="1.0.0")

# Allow the frontend (served from a different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # loaded lazily on startup


@app.on_event("startup")
def startup_event():
    global model
    print(f"[app.py] Loading model on device: {DEVICE}")
    model = load_model(CHECKPOINT_PATH, device=DEVICE)
    print("[app.py] Model loaded and ready.")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": model is not None}


@app.get("/labels")
def get_labels():
    return load_labels()


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(model, image_bytes, device=DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result
