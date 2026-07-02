# CHAPTER 3: METHODOLOGY

This chapter describes the unified experimental protocol used across the three phases of the thesis. Section 3.1 describes the dataset and splits. Section 3.2 covers preprocessing and augmentation. Section 3.3 describes the training pipeline shared by all architectures. Section 3.4 lists the evaluation metrics. Section 3.5 describes the conformal prediction protocol, Section 3.6 the Monte Carlo Dropout setup, and Section 3.7 the OOD detection protocol. Section 3.8 details the multi-class extension. Section 3.9 describes the implementation, code organisation, and reproducibility.

## 3.1 Dataset and Splits

### 3.1.1 APTOS 2019

All experiments use the APTOS 2019 Blindness Detection dataset, comprising 3,662 retinal fundus images at 224×224 pixels, each labelled by a clinician on the 0–4 severity scale. The dataset is loaded once at the start of each experiment and cached as a single pandas DataFrame on disk to ensure that every experiment sees the same image order and identical pixel values.

### 3.1.2 Stratified 70/15/15 Split

The bachelor thesis used an 80/20 train/test split with no separate validation set. The present thesis uses a stratified 70/15/15 train/validation/test split with `random_state = 123`:

- **Train (2,562 images)**: used for parameter updates.
- **Validation (550 images)**: used for early stopping, learning-rate scheduling, temperature-scaling fit, and conformal threshold fit. *Never* used for final evaluation.
- **Test (550 images)**: held out fixed across every experiment in this thesis. Every reported test metric is computed on the same 550 images.

Stratification preserves the class proportions across splits, which matters because the dataset is imbalanced. Without stratification, the validation or test set could under- or over-represent the minority severity classes by chance, making the test estimate unreliable.

### 3.1.3 Why a Separate Test Set Matters

The bachelor pipeline used the same data for early-stopping selection and for reporting "validation" accuracy, which is a mild form of data leakage: the held-out set is implicitly used for model selection through the choice of the early-stopping epoch. The present pipeline cleanly separates concerns: validation drives all selection decisions, and the test set is locked away until the end of training. The test set is also used unchanged across all phases (binary, multi-class, conformal, OOD), so methods can be directly compared.

## 3.2 Image Preprocessing

### 3.2.1 Architecture-Specific Preprocessing

A subtle but important methodological correction relative to the bachelor pipeline is the use of architecture-specific input preprocessing. Each ImageNet-pretrained backbone in Keras has its own `preprocess_input` function:

- **ResNet50**: BGR channel order, mean subtraction with `[103.939, 116.779, 123.68]` (ImageNet means in BGR).
- **DenseNet121**: same as ResNet50 (caffe-style).
- **Xception**: simple scaling to `[-1, 1]`.
- **VGG16**: BGR mean subtraction (caffe-style).
- **CNN (from scratch)**: grayscale conversion + division by 255 (matches the bachelor pipeline for backward compatibility).

The bachelor pipeline applied a uniform `1/255` rescaling to every architecture. This is a serious mismatch for the ImageNet backbones, whose pretrained weights expect mean-subtracted inputs; using `1/255` instead can drop test accuracy by 10 percentage points or more. The corrected protocol fixes this.

### 3.2.2 Data Augmentation

The training data generator applies a moderate set of geometric augmentations to combat overfitting:

- Rotation range: ±20°
- Width and height shift: ±10%
- Shear range: 10%
- Zoom range: 10%
- Horizontal flip: enabled

These are applied **only to the training set**. The validation and test generators apply the same architecture-specific preprocessing but no augmentation, so that all metrics are computed on un-perturbed images. The bachelor pipeline mistakenly augmented its test set as well; the present pipeline corrects this in `lib/helpers.py` by exposing `build_train_datagen` and `build_eval_datagen` as separate functions.

## 3.3 Training Pipeline

### 3.3.1 Common Components

Every model in this thesis is trained with the same core pipeline:

- **Optimizer**: Adam with initial learning rate $1 \times 10^{-3}$.
- **Loss**: binary cross-entropy for binary classifiers; categorical cross-entropy for the 5-class models.
- **Batch size**: 32.
- **Class weights**: computed via `sklearn.utils.class_weight.compute_class_weight(class_weight='balanced')` and passed into `model.fit(class_weight=...)`. This is critical for the heavily imbalanced 5-class problem.
- **Maximum epochs**: 100, with early stopping triggered if validation loss does not improve for 10 consecutive epochs.

### 3.3.2 Callbacks

Three callbacks are standard:

1. **EarlyStopping** (`monitor='val_loss', patience=10, restore_best_weights=True`): halts training once validation loss plateaus and restores the best weights. Most experiments stop between epochs 15 and 35.
2. **ReduceLROnPlateau** (`factor=0.5, patience=5, min_lr=1e-6`): halves the learning rate if validation loss stagnates for 5 epochs. This typically fires once or twice during a run.
3. **ModelCheckpoint** (`monitor='val_accuracy', save_best_only=True`): saves the best-validation model to disk. We always evaluate the *best* model rather than the final one.

### 3.3.3 Architecture Heads

For transfer-learning models, the ImageNet base is loaded with `include_top=False`, frozen, and topped with a Global Average Pooling layer, a 128-unit Dense layer with ReLU activation, and a final classification layer (sigmoid for binary, softmax for multi-class). The from-scratch CNN uses three convolution-pool blocks (32 → 64 → 128 filters with 3×3 kernels and 2×2 max-pooling) followed by a Flatten layer, a 128-unit Dense ReLU, and the classification layer.

## 3.4 Evaluation Metrics

Multiple metrics are computed for every model, organised as follows.

### 3.4.1 Discrimination Metrics

- **Accuracy**: fraction of correct predictions on the test set.
- **AUROC** (Area Under the Receiver Operating Characteristic curve): for binary tasks, measures discrimination at all thresholds.
- **Macro and weighted F1**: for multi-class, capture per-class precision-recall balance with and without class-frequency weighting.

### 3.4.2 Calibration Metrics

- **Expected Calibration Error (ECE)** with 15 equal-width confidence bins, computed both before and after temperature scaling.
- **Maximum Calibration Error (MCE)**: worst-case bin-level calibration gap.
- **Reliability diagram**: per-bin accuracy vs confidence, with a perfect-calibration diagonal as reference.

### 3.4.3 Ordinal Metrics (Multi-class Only)

For 5-stage grading, two additional metrics capture the ordinal structure of the labels:

- **Quadratic-Weighted Kappa (QWK)**: penalises errors by their squared distance on the severity scale. This is the metric used by the original APTOS 2019 Kaggle competition leaderboard.
- **Mean Ordinal Distance**: $\mathbb{E}[|y - \hat y|]$, a simpler companion to QWK.

### 3.4.4 Statistical Inference

- **Bootstrap 95% confidence intervals** on test accuracy, computed with 1,000 resamples and seed 42.
- **Pairwise McNemar tests** between every model pair, with a continuity correction. p < 0.05 indicates a statistically significant difference in error rates.
- **Cohen's kappa** between pairs of classifiers, as a chance-corrected agreement measure.

## 3.5 Conformal Prediction Protocol

### 3.5.1 Calibration Set

The validation split (550 images) is used as the conformal calibration set. Both LAC and APS scores are evaluated. The non-conformity threshold $\hat q$ is computed as the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile (with `method='higher'`), giving the finite-sample correction needed for the marginal coverage guarantee.

### 3.5.2 Coverage Targets

Two miscoverage levels are evaluated: $\alpha = 0.10$ (target 90% coverage) and $\alpha = 0.05$ (target 95% coverage). For each, both LAC and APS are computed.

### 3.5.3 Reported Diagnostics

For each (model, score, $\alpha$) triple, the following are recorded:

- **Empirical marginal coverage**: fraction of test points whose true label falls inside the conformal set.
- **Mean set size**: average $|C(x)|$.
- **Set size distribution**: histogram of |C(x)| values (0, 1, 2 for binary; 0, 1, 2, 3, 4, 5 for 5-class).
- **Singleton correct rate**: among single-class sets, fraction equal to the truth.
- **Abstain rate**: fraction of two-class sets (binary) or sets covering more than one class (multi-class).
- **Per-class conditional coverage** (multi-class): fraction of true-class points whose set includes that class, computed per class.

## 3.6 Monte Carlo Dropout Protocol

### 3.6.1 Architecture Variants

Two MC-Dropout-equipped variants are introduced for Phase 2:

- **`cnn_mcd`**: the from-scratch 3-block CNN with `SpatialDropout2D(rate=0.3)` after each convolution-pool pair and `Dropout(rate=0.3)` before the final classifier.
- **`resnet50_mcd`**: ResNet50 transfer learning with `Dropout(rate=0.3)` before the final dense layer; the convolutional base remains frozen.

### 3.6.2 Inference

At inference time, the model is called with `training=True` to keep dropout active. T = 30 stochastic forward passes are performed per test input. The per-pass probabilities are aggregated into:

- **Mean prediction**: $\bar p = \tfrac{1}{T} \sum_t p_t$, used for the binary decision.
- **Standard deviation**: $\sigma = \text{std}_t(p_t)$, an epistemic uncertainty estimate.
- **Predictive entropy**: $H(\bar p)$, total uncertainty.
- **Mean per-pass entropy**: $\mathbb{E}_t [H(p_t)]$, aleatoric proxy.
- **Mutual information** (BALD): predictive entropy minus mean per-pass entropy, epistemic component.

## 3.7 Out-of-Distribution Detection Protocol

### 3.7.1 In-Distribution and Out-of-Distribution Sources

For Phase 2, the in-distribution (ID) inputs are the 550 fundus images of the APTOS test set. Out-of-distribution (OOD) inputs are 300 synthetic images of uniform random noise, preprocessed identically to ID inputs (for ResNet50, this means BGR mean subtraction). This is admittedly the *easy* case for OOD detection; a proper cross-dataset evaluation on Messidor-2 or IDRiD is left for future work, with the infrastructure to run it already in place (`master/run_cross_dataset.py`).

### 3.7.2 Feature Extractor

DenseNet121 with ImageNet-pretrained weights and a frozen base produces 1024-dimensional features for both ID and OOD inputs. The validation set is used to fit per-class means and a shared covariance for Mahalanobis, and to compute a single ID centroid for cosine distance.

### 3.7.3 Scoring Methods

Four scores are evaluated:

- **Maximum Softmax Probability (MSP)**: $\max_k \hat p_k$ on the trained classifier.
- **Energy score**: $-T \cdot \log \sum_k \exp(z_k / T)$, with $T = 1$.
- **Mahalanobis distance** to the closest class mean in DenseNet feature space.
- **Cosine distance** to the ID centroid in DenseNet feature space.

For each score, ID and OOD distributions are reported, along with **AUROC** and **FPR at TPR = 95%**.

## 3.8 K-Fold Cross-Validation

To answer the question "how stable is the test accuracy estimate?" Phase 2 also runs a 5-fold cross-validation of ResNet50. The training+validation pool (3,112 images) is split into 5 stratified folds; the test set (550 images) is held fixed across folds. Each fold trains for up to 30 epochs with early stopping (patience = 6); the final test metrics are the mean and standard deviation across folds. The relative simplicity of this design — only ResNet50, only one architecture — is dictated by training time on CPU, but it is sufficient to put a tight bound on the variability of the single-split estimate used in Phase 1.

## 3.9 Multi-Class Extension

### 3.9.1 Label Reformulation

Phase 3 returns to the original 5-class severity labels (0 to 4) instead of the binary collapse used in the bachelor work and in Phases 1–2. The same 70/15/15 split is used, but stratified on the 5-class label so that each split contains the appropriate proportion of each severity level. The class distribution of the test set is heavily imbalanced: No DR 271, Mild 56, Moderate 150, Severe 29, PDR 44.

### 3.9.2 Architecture Heads

The classification head is changed from a single sigmoid unit to a 5-unit softmax output; the loss is changed from binary cross-entropy to categorical cross-entropy. The data generators use `class_mode='categorical'` and emit one-hot labels.

### 3.9.3 Architectures Trained

Two architectures are evaluated in the multi-class setting:

- **`cnn_5class`**: the same 3-block CNN as the binary CNN, with the head changed.
- **`resnet50_5class`**: ResNet50 transfer learning with the head changed.

### 3.9.4 Multi-Class Conformal Prediction

The conformal-prediction modules (`master/uncertainty/conformal.py`) generalise straightforwardly to multi-class via the LAC and APS scores defined per-class. The threshold $\hat q$ is fitted on the validation set as in the binary case. Conditional per-class coverage becomes especially informative: it tells us whether the marginal guarantee is honoured uniformly or whether the system is implicitly biased toward the majority "No DR" class.

### 3.9.5 Multi-Class Calibration

A multi-class generalisation of ECE is implemented in `master/uncertainty/calibration_mc.py`. Confidence is taken to be the maximum softmax probability $\max_k \hat p_k$, and predictions are bucketed into 15 confidence bins between $1/K$ and 1. Temperature scaling is generalised to softmax via the standard NLL minimisation.

## 3.10 Implementation and Reproducibility

### 3.10.1 Software Environment

- Python 3.9.6
- TensorFlow 2.15.0, Keras 2.15
- scikit-learn 1.4.2, scikit-image 0.22, OpenCV 4.7.0.72
- pandas 2.2.2, numpy 1.26.4, matplotlib 3.8.4, seaborn 0.13.2
- streamlit 1.33.0

The full pinned environment is captured in `requirements.txt`.

### 3.10.2 Hardware

All experiments run on an Apple Silicon (M-series) MacBook Air with 8 GB RAM, CPU-only. Training a transfer-learning binary classifier takes approximately 15 to 30 minutes; the from-scratch CNN takes approximately 30 to 60 minutes; the 5-fold CV of ResNet50 takes approximately 2 hours total.

### 3.10.3 Code Organisation

- `lib/`: Streamlit user interface and inference modules.
- `scripts/`: unified training driver (`train.py --arch <name>`) and the legacy per-architecture scripts retained for reproducibility of the bachelor numbers.
- `master/uncertainty/`: calibration, conformal, ensemble, MC Dropout, and OOD modules.
- `master/run_*.py`: analysis drivers that produced every CSV, JSON, and figure referenced in this thesis.
- `master/results/`: all experimental outputs, organised by experiment phase.
- `master/thesis/`: the LaTeX skeleton of this document, with auto-generated figures and tables.

### 3.10.4 Reproducibility Conventions

- **Random seeds**: stratified split uses `random_state=123`; bootstrap CIs use seed 42; the conformal `predict_sets` random-tie-breaking RNG uses seed 42.
- **Saved models**: every trained model is saved as a `.keras` file with the corresponding history pickle; analysis scripts reload these without retraining.
- **Conformal thresholds**: precomputed and saved as `master/results/multiclass/app_conformal_thresholds.json` so that the Streamlit app can produce conformal sets without re-fitting the threshold at inference time.
- **Documentation**: every analysis script includes a docstring describing inputs, outputs, and intended use.

The next four chapters present the experimental results.
