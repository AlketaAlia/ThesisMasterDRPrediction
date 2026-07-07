# Uncertainty-Aware Diabetic Retinopathy Grading

Master thesis project — **Uncertainty-Aware Diabetic Retinopathy Grading Using Conformal Prediction and Bayesian Deep Learning**, Polytechnic University of Tirana · Université de Technologie de Compiègne, Faculty of Information Technology.

The project builds a diabetic-retinopathy (DR) screening system that not only classifies fundus images but also **quantifies and communicates its own uncertainty** — through calibration, Monte Carlo Dropout, conformal prediction, and out-of-distribution detection — and defers ambiguous cases to a clinician.

---

## Key results

| Task | Metric | Result |
|------|--------|--------|
| Binary (DR / No DR) | Ensemble test accuracy | **96.55 %** |
| Binary | Best single model (resnet50_mcd) | 96.18 % |
| 5-stage grading | Quadratic-Weighted Kappa | **0.847** ("excellent") |
| Conformal triage | Auto-classify / refer split | **71 % / 29 %** |
| OOD detection | AUROC (Mahalanobis, cosine) | 1.0 (synthetic noise) |

The five strongest deep architectures are statistically indistinguishable (pairwise McNemar p > 0.5); the differentiating value comes from the uncertainty-quantification layer, not raw accuracy.

---

## Repository structure

```
├── app.py                       # Streamlit decision-support application
├── requirements.txt             # Pinned dependencies
├── lib/                         # Inference + UI modules
│   ├── inference.py             #   binary inference
│   ├── multiclass_inference.py  #   5-class inference
│   ├── uncertainty_inference.py #   conformal + UQ at inference
│   ├── quality.py               #   image-quality heuristics
│   ├── report.py                #   PDF report generation
│   └── i18n.py                  #   English / Albanian UI
├── scripts/
│   ├── train.py                 # Unified training driver
│   ├── helpers.py               # Splits, preprocessing, class weights
│   └── ML.py                    # Classical classifiers (DT, RF, SVM)
├── master/
│   ├── uncertainty/             # Calibration, conformal, MCD, OOD, ensemble, stats
│   ├── run_*.py                 # Analysis drivers
│   └── results/                 # Figures, CSVs, JSON summaries
├── translations/                # en.json, sq.json
└── inputs/labels.csv            # Image labels
```

> **Note:** Trained model weights (`*.keras`, `*.h5`) and the raw fundus images are **not** included in this repository because of their size. See *Setup* below.

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download the **APTOS 2019 Blindness Detection** dataset from Kaggle and place the images under `inputs/images/`:

https://www.kaggle.com/c/aptos2019-blindness-detection

### 3. Train a model

```bash
python scripts/train.py --arch resnet50
```

Available architectures: `resnet50`, `xception`, `densenet121`, `vgg16`, `cnn`, `cnn_tanh`, `cnn_mcd`, `resnet50_mcd`, `cnn_5class`, `resnet50_5class`.

### 4. Run the analyses

```bash
python master/run_uncertainty_analysis.py    # calibration, ensemble, McNemar
python master/run_phase2_analysis.py         # conformal + OOD
python master/run_mc_dropout_analysis.py     # MC Dropout
python master/run_kfold_cv.py                # 5-fold CV
python master/run_multiclass_analysis.py     # 5-stage grading
```

### 5. Launch the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Tech stack

Python 3.9 · TensorFlow 2.15 / Keras · scikit-learn · Streamlit · CPU-only (no GPU required).

---

## Author

**Alketa Alia** — Master in Artificial Intelligence and Optimization, Polytechnic University of Tirana · Université de Technologie de Compiègne.
