# KREU III

# METHODOLOGY

This chapter describes the unified experimental protocol used across the three phases of the thesis. Section 3.1 describes the dataset and splits. Section 3.2 covers preprocessing and augmentation. Section 3.3 describes the training pipeline shared by all architectures. Section 3.4 lists the evaluation metrics. Section 3.5 introduces the three classical-baseline classifiers. Sections 3.6 to 3.9 describe the experimental protocols for the subsequent phases (conformal prediction, MC Dropout, OOD, K-fold) — for the Second Control submission, these are in the implementation phase. Section 3.10 details the multi-class extension (planned for Phase 3). Section 3.11 describes implementation, code organisation, and reproducibility.

> *Second Control note: Sections 3.1–3.5 and 3.11 are fully stabilised. Sections 3.6–3.10 describe protocols that are under active implementation.*

## 3.1 Dataset and Splits

### 3.1.1 APTOS 2019

All experiments use the APTOS 2019 Blindness Detection dataset, comprising 3,662 retinal fundus images at 224×224 pixels, each labelled by a clinician on the 0–4 severity scale. The dataset is loaded once at the start of each experiment and cached as a single pandas DataFrame on disk to ensure that every experiment sees the same image order and identical pixel values. The class distribution is heavily skewed, dominated by the No DR class.

### 3.1.2 Stratified 70/15/15 Split

A stratified 70/15/15 train/validation/test split with `random_state = 123` is used:

- **Train (2,562 images)**: used for parameter updates.
- **Validation (550 images)**: used for early stopping, learning-rate scheduling, temperature-scaling fit, and conformal threshold fit. *Never* used for final evaluation.
- **Test (550 images)**: held out fixed across every experiment in this thesis. Every reported test metric is computed on the same 550 images.

Stratification preserves the class proportions across splits, which matters because the dataset is imbalanced. Without stratification, the validation or test set could under- or over-represent the minority severity classes by chance, making the test estimate unreliable.

### 3.1.3 Why a Separate Test Set Matters

It is methodologically important to separate the data used for model selection from the data used for final evaluation. Re-using the held-out set for early-stopping selection and for reported "validation" accuracy is a mild form of data leakage: the held-out set is implicitly used for model selection through the choice of the early-stopping epoch. The pipeline used here cleanly separates concerns: validation drives all selection decisions, and the test set is locked away until the end of training. The test set is also used unchanged across all phases.

## 3.2 Image Preprocessing

### 3.2.1 Architecture-Specific Preprocessing

A subtle but important methodological consideration is the use of architecture-specific input preprocessing. Each ImageNet-pretrained backbone in Keras has its own `preprocess_input` function:

- **ResNet50**: BGR channel order, mean subtraction with `[103.939, 116.779, 123.68]` (ImageNet means in BGR).
- **DenseNet121**: same as ResNet50 (caffe-style).
- **Xception**: simple scaling to `[-1, 1]`.
- **VGG16**: BGR mean subtraction (caffe-style).
- **CNN (from scratch)**: grayscale conversion + division by 255.

A naïve uniform `1/255` rescaling applied to every architecture is a serious mismatch for the ImageNet backbones, whose pretrained weights expect mean-subtracted inputs; using `1/255` instead can drop test accuracy by 10 percentage points or more. Each ImageNet backbone in this thesis therefore receives its own canonical preprocessing function.

### 3.2.2 Data Augmentation

The training data generator applies a moderate set of geometric augmentations to combat overfitting:

- Rotation range: ±20°
- Width and height shift: ±10%
- Shear range: 10%
- Zoom range: 10%
- Horizontal flip: enabled

These are applied **only to the training set**. The validation and test generators apply the same architecture-specific preprocessing but no augmentation, so that all metrics are computed on un-perturbed images.

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

1. **EarlyStopping** (`monitor='val_loss', patience=10, restore_best_weights=True`): halts training once validation loss plateaus and restores the best weights.
2. **ReduceLROnPlateau** (`factor=0.5, patience=5, min_lr=1e-6`): halves the learning rate if validation loss stagnates for 5 epochs.
3. **ModelCheckpoint** (`monitor='val_accuracy', save_best_only=True`): saves the best-validation model to disk.

### 3.3.3 Architecture Heads

For transfer-learning models, the ImageNet base is loaded with `include_top=False`, frozen, and topped with a Global Average Pooling layer, a 128-unit Dense layer with ReLU activation, and a final classification layer (sigmoid for binary, softmax for multi-class). The from-scratch CNN uses three convolution-pool blocks (32 → 64 → 128 filters with 3×3 kernels and 2×2 max-pooling) followed by a Flatten layer, a 128-unit Dense ReLU, and the classification layer.

## 3.4 Evaluation Metrics

Multiple metrics are computed for every model, organised as follows.

### 3.4.1 Discrimination Metrics

- **Accuracy**: fraction of correct predictions on the test set.
- **AUROC**: for binary tasks, measures discrimination at all thresholds.
- **Macro and weighted F1**: for multi-class, capture per-class precision-recall balance.

### 3.4.2 Calibration Metrics

- **Expected Calibration Error (ECE)** [2] with 15 equal-width confidence bins.
- **Maximum Calibration Error (MCE)** [7]: worst-case bin-level calibration gap.
- **Reliability diagram**: per-bin accuracy vs confidence.

### 3.4.3 Ordinal Metrics (Multi-class only)

For 5-stage grading, two additional metrics capture the ordinal structure of the labels:

- **Quadratic-Weighted Kappa (QWK)**: penalises errors by their squared distance on the severity scale.
- **Mean Ordinal Distance**: $\mathbb{E}[|y - \hat y|]$.

### 3.4.4 Statistical Inference

- **Bootstrap 95% confidence intervals** on test accuracy, computed with 1,000 resamples and seed 42.
- **Pairwise McNemar tests** between every model pair, with a continuity correction.
- **Cohen's kappa** between pairs of classifiers, as a chance-corrected agreement measure.

## 3.5 Classical Baseline Classifiers

To anchor the comparison of deep architectures against simpler methods, three classical machine-learning classifiers are also evaluated: a Decision Tree, a Random Forest, and a Support Vector Machine. Rather than training them on raw 224×224×3 pixels — which would yield 150,528-dimensional feature vectors — the classical models are trained on **deep features extracted from a frozen DenseNet121 backbone with ImageNet pretrained weights**. The base network is loaded with `include_top=False, pooling='avg'`, producing a 1024-dimensional feature vector per input image.

The three classifiers, all from `scikit-learn`:

- **Decision Tree**: default Gini split criterion, `class_weight='balanced'`, `random_state=123`.
- **Random Forest**: 300 trees, `class_weight='balanced'`, `random_state=123`.
- **Support Vector Machine**: RBF kernel, `probability=True`, `class_weight='balanced'`.

## 3.6 Conformal Prediction Protocol *(under implementation)*

The split conformal procedure of Vovk et al. [10] and Angelopoulos & Bates [11] is adopted. The validation set (550 images) will be used as the conformal calibration set. Both Least Ambiguous Classifier (LAC) and Adaptive Prediction Sets (APS) [12] scores will be evaluated. The non-conformity threshold $\hat q$ will be computed as the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile, giving the finite-sample correction needed for the marginal coverage guarantee.

Two miscoverage levels will be evaluated: $\alpha = 0.10$ (target 90% coverage) and $\alpha = 0.05$ (target 95% coverage).

## 3.7 Monte Carlo Dropout Protocol *(under implementation)*

Following Gal and Ghahramani [8], two MC-Dropout-equipped variants will be introduced for Phase 2:

- **`cnn_mcd`**: the from-scratch 3-block CNN with `SpatialDropout2D(rate=0.3)` after each convolution-pool pair.
- **`resnet50_mcd`**: ResNet50 transfer learning with `Dropout(rate=0.3)` before the final dense layer.

At inference time, the model will be called with `training=True` to keep dropout active. T = 30 stochastic forward passes will be performed per test input.

## 3.8 Out-of-Distribution Detection Protocol *(under implementation)*

For Phase 2, the in-distribution (ID) inputs will be the 550 fundus images of the APTOS test set. Out-of-distribution (OOD) inputs will be 300 synthetic images of uniform random noise; a planned extension is cross-dataset evaluation against the Messidor [19] and IDRiD [20] datasets. DenseNet121 with ImageNet-pretrained weights and a frozen base will produce 1024-dimensional features for both ID and OOD inputs. The validation set will be used to fit per-class means and a shared covariance for Mahalanobis.

Four scores will be evaluated: Maximum Softmax Probability (MSP) [13], Energy score [14], Mahalanobis distance [15], and Cosine distance to the in-distribution centroid.

## 3.9 K-Fold Cross-Validation *(planned)*

Phase 2 will also run a 5-fold cross-validation of ResNet50. The training+validation pool (3,112 images) will be split into 5 stratified folds; the test set (550 images) will be held fixed across folds. The final test metrics will be the mean and standard deviation across folds.

## 3.10 Multi-Class Extension *(Phase 3 — planned)*

Phase 3 will reformulate the task using the original 5-class severity labels (0 to 4) instead of the binary collapse. The same 70/15/15 split will be used, but stratified now on the 5-class label.

Two architectures will be evaluated in the multi-class setting: `cnn_5class` and `resnet50_5class`.

## 3.11 Implementation and Reproducibility

### 3.11.1 Software Environment

- Python 3.9.6
- TensorFlow 2.15.0, Keras 2.15
- scikit-learn 1.4.2, pandas 2.2.2, numpy 1.26.4
- streamlit 1.33.0

The full pinned environment is captured in `requirements.txt`.

### 3.11.2 Hardware

All experiments run on an Apple Silicon (M-series) MacBook Air with 8 GB RAM, CPU-only. Training a transfer-learning binary classifier takes approximately 15 to 30 minutes; the from-scratch CNN takes approximately 30 to 60 minutes.

### 3.11.3 Code Organisation

- `lib/`: Streamlit user interface and inference modules.
- `scripts/`: unified training driver (`train.py --arch <name>`).
- `master/uncertainty/`: calibration, conformal, ensemble, MC Dropout, and OOD modules.
- `master/run_*.py`: analysis drivers that produced every CSV, JSON, and figure referenced in this thesis.
- `master/results/`: all experimental outputs.

### 3.11.4 Reproducibility Conventions

- **Random seeds**: stratified split uses `random_state=123`; bootstrap CIs use seed 42.
- **Saved models**: every trained model is saved as a `.keras` file with the corresponding history pickle.

\newpage
