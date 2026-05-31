# -*- coding: utf-8 -*-
# Preprocesare aplicata inainte de predictie:
# Time si Amount sunt standardizate cu media + std din dataset, V1..V28 raman
# nemodificate (sunt deja PCA-standardizate la antrenare).

from pathlib import Path

import joblib
import pandas as pd


class Preprocessor:
    """Standardizare Time/Amount + materializare statistici per-clasa."""

    def __init__(self, stats: dict):
        # stats["global"] -> medie / std pe Time si Amount (folosit la scalare)
        # stats["legit"] / stats["frauda"] -> per-feature stats (folosit la generare)
        self.stats = stats

    # ---------------------------------------------------------------- runtime
    def scale(self, raw: dict) -> dict:
        """Aplica acelasi z-score ca la antrenare pe Time si Amount."""
        g = self.stats["global"]
        scaled = dict(raw)
        scaled["Time"] = (raw["Time"] - g["time_mean"]) / g["time_std"]
        scaled["Amount"] = (raw["Amount"] - g["amount_mean"]) / g["amount_std"]
        return scaled

    # ---------------------------------------------------------------- bootstrap
    @staticmethod
    def _per_class_stats(sub: pd.DataFrame) -> dict:
        return {
            "time_mean": float(sub["Time"].mean()),
            "time_std":  float(sub["Time"].std()),
            "amount_mean": float(sub["Amount"].mean()),
            "amount_std":  float(sub["Amount"].std()),
            "v_mean": {f"V{i}": float(sub[f"V{i}"].mean()) for i in range(1, 29)},
            "v_std":  {f"V{i}": float(sub[f"V{i}"].std())  for i in range(1, 29)},
        }

    @staticmethod
    def compute_stats(data_path: Path) -> dict:
        """Citeste creditcard.csv si extrage media/std global + per-clasa."""
        df = pd.read_csv(data_path)
        return {
            "global": {
                "time_mean": float(df["Time"].mean()),
                "time_std":  float(df["Time"].std()),
                "amount_mean": float(df["Amount"].mean()),
                "amount_std":  float(df["Amount"].std()),
            },
            "legit":  Preprocessor._per_class_stats(df[df["Class"] == 0]),
            "frauda": Preprocessor._per_class_stats(df[df["Class"] == 1]),
        }

    @staticmethod
    def load_stats(cache_path: Path, data_path: Path) -> dict:
        """Returneaza statistici din cache daca exista, altfel le calculeaza si le salveaza."""
        if cache_path.exists():
            return joblib.load(cache_path)
        stats = Preprocessor.compute_stats(data_path)
        try:
            joblib.dump(stats, cache_path)        # accelereaza pornirile urmatoare
        except Exception:
            pass
        return stats
