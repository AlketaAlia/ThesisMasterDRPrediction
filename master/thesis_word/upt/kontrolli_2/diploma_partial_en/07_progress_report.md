# POLYTECHNIC UNIVERSITY OF TIRANA

# FACULTY OF INFORMATION TECHNOLOGY

# DEPARTMENT OF COMPUTER ENGINEERING

---

# PROGRESS REPORT — SECOND CONTROL OF THE MASTER THESIS

**Thesis title:** Uncertainty-Aware Diabetic Retinopathy Grading Using Conformal Prediction and Bayesian Deep Learning

**Candidate:** Alketa Alia

**Supervisor:** Prof. Dr. [Name Surname]

**Submission date:** [Date]

**Overall status:** **Approximately 50% complete**

---

## 1. Status summary

The thesis is progressing according to the original plan approved at the First Control. The first experimental phase (binary classification) has been completed with concrete results; the second phase (advanced uncertainty methods) is under active development; the theoretical chapter (Kreu II) is nearly complete, and the methodology (Kreu III) is a first stable draft.

The table below summarises the status by chapter and experimental component.

| Component | Status | Percent |
|-----------|--------|--------:|
| Kreu I — Introduction | First draft completed | 90% |
| Kreu II — Theoretical Background | First draft, under review | 80% |
| Kreu III — Methodology | Working draft, stabilising | 70% |
| Kreu IV — Phase 1 results (binary) | Experiments completed, tables drafted | 65% |
| Kreu IV — Phase 2 results (UQ) | Implementation in progress | 30% |
| Kreu IV — Phase 3 results (multi-class) | Planned | 0% |
| Kreu V — Conclusions | Planned | 0% |
| Streamlit Application | Functional prototype | 40% |
| Bibliography | 20 current refs, target 40+ | 50% |
| **Overall total** | — | **~50%** |

---

## 2. What has been done so far

### 2.1 Theoretical part (Kreu I and II)

- The introduction is fully written, with the description of the clinical problem of diabetic retinopathy, the motivation for using uncertainty in medical AI, and the statement of the **four research questions** that guide the work.
- Kreu II — Theoretical Background — has been written in first draft, including:
  - Pathophysiology and clinical scale of DR (5 stages)
  - APTOS 2019 dataset (3,662 images)
  - Description of the 6 evaluated architectures
  - Concepts of calibration, Monte Carlo Dropout, conformal prediction, and OOD detection
  - Regulatory framework (FDA SaMD, EU AI Act)

### 2.2 Methodology (Kreu III)

- The **unified training pipeline** has been built (`scripts/train.py`)
- The **stratified 70/15/15 split** with fixed random_state has been implemented (`scripts/helpers.py`)
- **Architecture-specific preprocessing** for every backbone has been configured
- **Standard callbacks** are in place: EarlyStopping (patience=10), ReduceLROnPlateau, ModelCheckpoint
- Experimental protocols for calibration, conformal, MCD, and OOD have been documented

### 2.3 Experiments — Phase 1 (binary classification)

Fully completed. **Six binary models** have been trained, and preliminary results are:

| Model | Test acc | 95% CI |
|-------|---------:|:------:|
| ResNet50 | 95.27% | [93.45, 97.09] |
| Xception | 95.45% | [93.82, 97.09] |
| DenseNet121 | 95.64% | [94.00, 97.27] |
| VGG16 | 95.64% | [94.00, 97.27] |
| CNN (from scratch) | 96.00% | [94.36, 97.45] |
| CNN (Tanh+ReLU) | 92.91% | [90.72, 94.91] |

Preliminary findings:
- The five strong models are statistically indistinguishable (McNemar p > 0.5)
- CNN(Tanh+ReLU) is significantly worse (p < 0.05)
- The pipeline shows high stability in the reported metrics

### 2.4 Streamlit Application — prototype

A functional prototype has been built (`app.py`), supporting:
- Single image upload
- Inference with the selected model
- Display of raw probabilities
- Multilingual UI (English / Albanian)

---

## 3. What is currently in progress

### 3.1 Phase 2 — Advanced uncertainty methods

- **Conformal prediction** (`master/uncertainty/conformal.py`): LAC implementation is complete; APS and per-class diagnostics are being finalised. Expected to finish within 2 weeks.
- **Monte Carlo Dropout**: Both variants (`cnn_mcd`, `resnet50_mcd`) have been designed; training is in progress. Once trained, T = 30 forward-pass analysis will follow.
- **K-Fold Cross-Validation**: The script is ready (`master/run_kfold_cv.py`); execution for ResNet50 is planned (~2 hours on CPU).
- **OOD Detection**: Implementation of all 4 methods (MSP, Energy, Mahalanobis, Cosine) is in progress.

### 3.2 Statistical analysis

- Bootstrap CI computation is ready and has been executed for Phase 1.
- Pairwise McNemar tests have been executed.
- Calibration analysis (ECE, MCE, reliability diagrams) is half-completed.

### 3.3 Streamlit app expansion

Work in progress to integrate:
- Calibrated probabilities (after T-scaling)
- Conformal sets in real time
- Image-quality heuristics
- PDF export

---

## 4. What remains (~50% remaining)

### 4.1 Remaining experiments

- **Full Phase 2**: conformal evaluation, MC Dropout analysis, K-Fold CV, OOD detection
- **Phase 3 — Multi-class 5-stage grading**:
  - Training of `cnn_5class` and `resnet50_5class`
  - Quadratic Weighted Kappa, ordinal distance
  - Multi-class conformal with per-class coverage
  - Multi-class calibration
- **Classical classifiers** (Decision Tree, Random Forest, SVM) on DenseNet features

### 4.2 Remaining writing

- Kreu IV — sections for Phase 2 and Phase 3
- Kreu IV — synthesis of findings and sensitivity analysis
- Kreu V — Conclusions, limitations, and future work
- Abstract in Albanian and English
- Discussion of clinical implications

### 4.3 Supporting components

- Finalisation of the Streamlit application with all UQ features
- Expansion of the bibliography to 40+ references (Vancouver style)
- Creation of all final figures and tables
- Formatting according to the UPT standard (Times New Roman 12pt, 2.54/3.8 cm margins, etc.)
- Table of Contents, List of Figures, List of Tables

---

## 5. Timeline for completion

| Period | Main activities |
|--------|-----------------|
| May 2026 (weeks 2-3) | Finalisation of Phase 2: conformal evaluation, MC Dropout, OOD |
| May 2026 (week 4) | K-Fold cross-validation for ResNet50 |
| June 2026 (weeks 1-2) | Training and analysis for Phase 3 (5-class grading) |
| June 2026 (weeks 3-4) | Writing Kreu IV (full results) |
| July 2026 (weeks 1-2) | Writing Kreu V (conclusions) + Kreu IV synthesis |
| July 2026 (weeks 3-4) | Finalising the Streamlit application with all UQ features |
| August 2026 (weeks 1-2) | Bibliography expansion, Kreu II revision |
| August 2026 (weeks 3-4) | Final UPT formatting, TOC + figure/table lists |
| September 2026 (week 1) | Full review with the supervisor |
| September 2026 (week 2) | Final corrections and printing for defence |
| September 2026 (week 3) | **Final submission and defence** |

---

## 6. Identified challenges and resolutions

### 6.1 CPU-only training

**Challenge**: The development environment is CPU-only (MacBook Air M-series, 8 GB RAM). This constrains the architectures and training regimes that are realistically feasible.

**Resolution**: This has been turned into a methodological point — the system is designed to be deployable in low-resource settings (primary-care clinics). Larger models such as Vision Transformers are left for future work.

### 6.2 Dataset imbalance

**Challenge**: APTOS has a very skewed distribution (50% No DR, only 13% for stages 3+4 combined).

**Resolution**: Use of `class_weight='balanced'` in training; per-class analysis in Phase 3 to surface performance on minority classes; conformal prediction with per-class coverage.

### 6.3 Real OOD validation

**Challenge**: No second dataset is currently available for cross-dataset evaluation.

**Workaround for now**: Use of synthetic noise as OOD; the infrastructure for cross-dataset evaluation (`master/run_cross_dataset.py`) is ready for when Messidor-2 or IDRiD become available.

---

## 7. Summary statistics

| Indicator | Current value | Final target |
|-----------|--------------:|-------------:|
| Pages written | ~25 | ~60 |
| Word count | ~8,500 | ≥17,000 |
| Chapters drafted | 3 (Kreu I, II, III) | 5 |
| Result tables | 3 | ~12 |
| Figures | 6 | ~24 |
| Trained models | 6 binary | 10 (6 binary + 2 MCD + 2 multi-class) |
| Bibliography references | 20 | 40+ |

---

## 8. Conclusion

The thesis is progressing according to the original plan. The first experimental phase has been completed with competitive results (ensemble ~96.5%), and the code infrastructure for the subsequent phases is built and documented. The proposed timeline makes final submission in **September 2026** realistically achievable, leaving sufficient time for review with the supervisor and final corrections.

Thank you for your attention and guidance.

---

**Alketa Alia**

Candidate, Master in Computer Engineering

[Email: aivalanche.2023@gmail.com]

Tirana, [Submission date]
