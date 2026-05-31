# -*- coding: utf-8 -*-
# Pydantic models + constante partajate.
# Pastrate intr-un modul separat ca sa fie importabile fara a trage si dependintele
# de joblib / numpy din model.py si preprocessor.py.

from pydantic import BaseModel, ConfigDict, create_model

# ordinea EXACTA in care modelul rf_smote astepta features la antrenare
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# top 5 dupa importanta RF (Capitolul 6 — rf_smote)
TOP_FEATURES = ["V14", "V10", "V12", "V17", "V16"]

# metrica raportata in /health si in footer-ul UI
AUC_PR = 0.8842


# ------------------------------------------------------------------ input request
# 30 de campuri float cu default 0.0, construite dinamic
TransactionInput = create_model(
    "TransactionInput",
    **{name: (float, 0.0) for name in FEATURE_NAMES},
)


# ------------------------------------------------------------------ output models
class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float
    contribution: float


class PredictResponse(BaseModel):
    label: str                                  # "FRAUDA" | "LEGITIMA"
    probability: float                          # P(frauda) in [0, 1]
    top5_features: list[FeatureContribution]
    message: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # permite campul "model"
    status: str
    model: str
    auc_pr: float
