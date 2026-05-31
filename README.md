# Detectia fraudelor in tranzactii financiare folosind inteligenta artificiala

Lucrare de licenta — Universitatea „Lucian Blaga" din Sibiu, Facultatea de Stiinte,
Specializarea Informatica, 2026.

Detectarea tranzactiilor frauduloase cu modele de machine learning, pe un set de
date puternic dezechilibrat.

**Dataset:** ULB Credit Card Fraud Detection — 284.807 tranzactii, dintre care
492 fraude (0.17%).

---

## Structura

```
.
├── notebooks/
│   ├── 01_eda.ipynb              # Analiza exploratorie a datelor (EDA)
│   ├── 02_preprocessing.ipynb    # SMOTE, ADASYN, Random Undersampling
│   ├── 03_models.ipynb           # RF, XGBoost, LightGBM + evaluare
│   └── 04_explainability.ipynb   # Interpretabilitate (SHAP, importanta features)
├── api/                          # Aplicatia demo (Capitolul 7) — FastAPI + SPA
│   ├── main.py                   # Montare FastAPI + rute REST
│   ├── schemas.py                # Modele Pydantic + constante
│   ├── preprocessor.py           # Scalare Time/Amount, statistici dataset
│   ├── model.py                  # FraudDetector — predictie, explicabilitate
│   ├── generator.py              # Generare tranzactii sintetice per-clasa
│   └── static/                   # Frontend SPA (index.html, app.js, style.css)
├── figures/                      # Figuri exportate pentru lucrare (PNG, 300 dpi)
├── data/                         # Nu e pe Git — adauga manual creditcard.csv
└── models/                       # Nu e pe Git — generate de notebooks
```

---

## Setup

Python via Anaconda (base environment).

Instaleaza dependintele:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap joblib
```

Descarca datasetul de pe Kaggle:
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Pune `creditcard.csv` in folderul `data/`.

---

## Rularea notebook-urilor

```bash
jupyter notebook
# sau, direct cu interpretorul Anaconda:
# "C:/Users/Gabi-Del/anaconda3/python.exe" -m jupyter notebook
```

Ordinea recomandata:
`01_eda` → `02_preprocessing` → `03_models` → `04_explainability`

Pasul `03_models` antreneaza si salveaza modelele in `models/` (necesare pentru demo).

---

## Demo interactiv (FastAPI + SPA)

Aplicatia demo din Capitolul 7: backend REST (FastAPI) + frontend SPA static.
Are nevoie de modelul antrenat (`models/rf_smote.joblib`) si de dataset
(`data/creditcard.csv`) — ! ruleaza mai intai `03_models.ipynb`.

Instaleaza dependintele:
```bash
pip install fastapi uvicorn
```

Porneste aplicatia din radacina proiectului:
```bash
uvicorn api.main:app --reload
```

Se deschide la `http://localhost:8000`:
- `/` — frontend SPA: introdu parametrii tranzactiei sau genereaza automat exemple
  legitime / frauda, apoi **Verifica tranzactie** pentru predictie + top 5 features.
- `/docs` — Swagger UI cu endpoint-urile REST.

Endpoint-uri principale:
- `POST /predict` — predictie pentru o tranzactie (eticheta, probabilitate, top 5 features, mesaj).
- `GET /generate/{class_type}` — genereaza o tranzactie sintetica (`legitimate` / `fraud`).
- `GET /health` — status model.

---

## Modele antrenate

- Random Forest (SMOTE) — modelul folosit in demo
- XGBoost (SMOTE)
- LightGBM (SMOTE)

Metrica principala: **AUC-PR** (aria sub curba Precision-Recall) — relevanta pe
seturi de date dezechilibrate.

---

## Autor

Axente Andrei-Eugen
Coordonator stiintific: Conf. univ. dr. Stoica Florin
