# -*- coding: utf-8 -*-
# Generare de tranzactii sintetice reprezentative pentru o clasa.
# Esantioane scoase din N(mean, std) per-feature, clip la intervalele
# observate in dataset.

import numpy as np


class TransactionGenerator:
    """Esantioneaza tranzactii dintr-o distributie normala per-clasa."""

    AMOUNT_MAX = 5000.0
    TIME_MAX   = 172792.0       # ~ 48h in secunde
    V_CLIP     = 5.0

    def __init__(self, stats: dict, rng: np.random.Generator | None = None):
        # stats trebuie sa contina "legit" si "frauda"
        self.stats = stats
        self.rng = rng or np.random.default_rng()

    def generate(self, class_type: str) -> dict:
        """
        Returneaza un dict cu 30 features. `class_type` accepta:
          - "fraud" / "frauda"
          - "legitimate" / "legitima" / "legit"
        Ridica ValueError pentru alte valori (main.py il converteste in HTTP 400).
        """
        cs = self._stats_for(class_type)
        out: dict = {}
        for i in range(1, 29):
            name = f"V{i}"
            out[name] = round(float(np.clip(
                self.rng.normal(cs["v_mean"][name], cs["v_std"][name]),
                -self.V_CLIP, self.V_CLIP,
            )), 4)
        out["Amount"] = round(float(np.clip(
            abs(self.rng.normal(cs["amount_mean"], cs["amount_std"])),
            0.0, self.AMOUNT_MAX,
        )), 2)
        out["Time"] = round(float(np.clip(
            abs(self.rng.normal(cs["time_mean"], cs["time_std"])),
            0.0, self.TIME_MAX,
        )), 0)
        return out

    # ---------------------------------------------------------------- helpers
    def _stats_for(self, class_type: str) -> dict:
        ct = class_type.lower()
        if ct in ("fraud", "frauda"):
            return self.stats["frauda"]
        if ct in ("legitimate", "legitima", "legit"):
            return self.stats["legit"]
        raise ValueError("class_type trebuie sa fie 'fraud' sau 'legitimate'.")
