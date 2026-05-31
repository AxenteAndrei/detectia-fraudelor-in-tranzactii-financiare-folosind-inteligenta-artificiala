# -*- coding: utf-8 -*-
# Wrapper subtire peste estimator-ul sklearn (rf_smote.joblib).
# Expune predict_proba, top_contributions si compose_message — toata logica
# pe care main.py o foloseste pentru a construi PredictResponse.

from pathlib import Path

import joblib
import numpy as np

from .schemas import FeatureContribution


class FraudDetector:
    """RandomForest + SMOTE (vezi Capitolul 6). Pragul de decizie e 0.5."""

    THRESHOLD = 0.5

    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)
        self.importances = self.model.feature_importances_

    # ---------------------------------------------------------------- predictie
    def predict_proba(self, x_scaled: np.ndarray) -> float:
        """Probabilitatea de frauda (clasa 1) pentru un vector de 30 features."""
        return float(self.model.predict_proba(x_scaled.reshape(1, -1))[0, 1])

    def is_fraud(self, proba: float) -> bool:
        return proba >= self.THRESHOLD

    # ---------------------------------------------------------------- explicabilitate
    def top_contributions(
        self,
        x_scaled: np.ndarray,
        raw: dict,
        feature_names: list[str],
        k: int = 5,
    ) -> list[FeatureContribution]:
        """
        Top-k features dupa contributie = importanta_RF * |valoare scalata|.
        Formula este aceeasi ca in analiza din Cap. 6, ca rezultatele
        sa fie reproductibile.
        """
        contributii = self.importances * np.abs(x_scaled)
        top_idx = np.argsort(contributii)[::-1][:k]
        return [
            FeatureContribution(
                feature=feature_names[i],
                value=round(raw[feature_names[i]], 3),
                importance=round(float(self.importances[i]), 4),
                contribution=round(float(contributii[i]), 4),
            )
            for i in top_idx
        ]

    # ---------------------------------------------------------------- mesaj UI
    @staticmethod
    def compose_message(este_frauda: bool, top_names: str) -> str:
        """Mesajul afisat sub badge in UI."""
        if este_frauda:
            return (
                f"Modelul considera ca aceasta tranzactie prezinta caracteristici "
                f"asociate fraudei. Cele mai influente variabile in decizie au fost: "
                f"{top_names}. Un scor peste pragul de 0.5 ar declansa o verificare "
                f"manuala sau blocarea tranzactiei."
            )
        return (
            f"Modelul estimeaza ca tranzactia este legitima. Cele mai relevante "
            f"variabile analizate au fost: {top_names}. Scorul ramane sub pragul de "
            f"0.5, deci nu se recomanda blocarea."
        )
