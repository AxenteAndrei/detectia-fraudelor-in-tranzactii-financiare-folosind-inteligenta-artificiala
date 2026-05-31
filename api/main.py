# -*- coding: utf-8 -*-
# Backend FastAPI pentru demo-ul de detectie a fraudelor (Capitolul 7).
#
# Rulare:
#   pip install fastapi uvicorn
#   uvicorn api.main:app --reload     (din radacina proiectului)
#   -> http://localhost:8000
#
# Acest fisier contine DOAR montarea FastAPI si rutele. Toata logica e in:
#   - api/schemas.py        (Pydantic models + constante)
#   - api/preprocessor.py   (scalare Time/Amount, statistici dataset)
#   - api/model.py          (FraudDetector — predict, explicabilitate, mesaj)
#   - api/generator.py      (TransactionGenerator — esantionare per-clasa)

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .generator import TransactionGenerator
from .model import FraudDetector
from .preprocessor import Preprocessor
from .schemas import (
    AUC_PR,
    FEATURE_NAMES,
    HealthResponse,
    PredictResponse,
    TransactionInput,
)

# ----------------------------------------------------------------- cai resurse
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "models" / "rf_smote.joblib"
DATA_PATH   = ROOT / "data" / "creditcard.csv"
STATS_CACHE = ROOT / "data" / "demo_stats.joblib"
STATIC_DIR  = Path(__file__).resolve().parent / "static"

# ----------------------------------------------------------------- app
app = FastAPI(
    title="API Detectie Frauda",
    version="1.0",
    description="Backend RandomForest + SMOTE pentru demo-ul din Cap. 7.",
)

# componente populate la startup
detector:     FraudDetector | None = None
preprocessor: Preprocessor | None = None
tx_generator: TransactionGenerator | None = None


@app.on_event("startup")
def load_resources():
    global detector, preprocessor, tx_generator
    detector = FraudDetector(MODEL_PATH)
    stats = Preprocessor.load_stats(STATS_CACHE, DATA_PATH)
    preprocessor = Preprocessor(stats)
    tx_generator = TransactionGenerator(stats)
    print(f"[startup] Model + statistici incarcate. Features: {len(FEATURE_NAMES)}")


# ================================================================ endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model": "RandomForest", "auc_pr": AUC_PR}


@app.post("/predict", response_model=PredictResponse)
def predict(tx: TransactionInput):
    if detector is None or preprocessor is None:
        raise HTTPException(503, "Model neincarcat.")

    raw = {name: float(getattr(tx, name)) for name in FEATURE_NAMES}
    scaled = preprocessor.scale(raw)
    x_scaled = np.array([scaled[name] for name in FEATURE_NAMES], dtype=float)

    proba = detector.predict_proba(x_scaled)
    este_frauda = detector.is_fraud(proba)
    top5 = detector.top_contributions(x_scaled, raw, FEATURE_NAMES, k=5)
    top_names = ", ".join(t.feature for t in top5[:3])

    return PredictResponse(
        label="FRAUDA" if este_frauda else "LEGITIMA",
        probability=round(proba, 4),
        top5_features=top5,
        message=FraudDetector.compose_message(este_frauda, top_names),
    )


@app.get("/generate/{class_type}", response_model=TransactionInput)
def generate(class_type: str):
    if tx_generator is None:
        raise HTTPException(503, "Generator neincarcat.")
    try:
        return tx_generator.generate(class_type)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ----------------------------------------------------------------- frontend SPA
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
