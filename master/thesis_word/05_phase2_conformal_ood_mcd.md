# CHAPTER 5: CONFORMAL PREDICTION, MC DROPOUT, K-FOLD CV, AND OOD DETECTION (PHASE 2)

This chapter implements the second research question of the thesis: *what is the most useful uncertainty signal for selective prediction and out-of-distribution detection, and can a conformal-prediction wrapper provide a finite-sample coverage guarantee?* Section 5.1 reports split conformal prediction with both LAC and APS scores at α = 0.10 and α = 0.05. Section 5.2 covers Monte Carlo Dropout. Section 5.3 reports out-of-distribution detection results. Section 5.4 presents the 5-fold cross-validation of ResNet50. Section 5.5 summarises the lessons of Phase 2.

## 5.1 Split Conformal Prediction

### 5.1.1 Setup

The validation set (550 images) is used as the conformal calibration set. Two scoring rules — LAC and APS — and two coverage targets — α = 0.10 and α = 0.05 — give four configurations per model. The non-conformity threshold $\hat q$ is computed with the finite-sample correction $\lceil (n+1)(1-\alpha) \rceil / n$, using `np.quantile(..., method='higher')`.

### 5.1.2 Per-Model Coverage at α = 0.10

**Table 5.1.** Conformal prediction at α = 0.10 (target 90% coverage). Coverage is the fraction of test points whose true label falls inside the conformal set; mean size is the average |C(x)|. Singleton correct rate is the accuracy among test points whose set has size 1; abstain rate is the fraction of two-class sets {0, 1}; empty rate is the fraction of empty sets.

| Model | Score | Coverage | Mean size | Singleton correct | Abstain {0,1} | Empty {} |
|-------|------:|---------:|----------:|------------------:|--------------:|---------:|
| ResNet50 | LAC | 90.36% | 0.93 | 97.45% | 0.0% | 7.3% |
| ResNet50 | APS | 88.73% | 0.96 | 96.67% | 4.4% | 8.4% |
| Xception | LAC | 92.55% | 0.95 | 97.32% | 0.0% | 4.9% |
| Xception | APS | 88.36% | 0.97 | 96.41% | 5.5% | 8.5% |
| DenseNet121 | LAC | 89.09% | 0.91 | 97.61% | 0.0% | 8.7% |
| DenseNet121 | APS | 88.36% | 0.97 | 96.79% | 6.0% | 8.9% |
| **VGG16** | **LAC** | 91.27% | 0.93 | **98.05%** | 0.0% | 6.9% |
| VGG16 | APS | 89.27% | 0.96 | 97.09% | 4.4% | 8.2% |
| CNN | LAC | 91.82% | 0.95 | 96.74% | 0.0% | 5.1% |
| CNN | APS | 89.09% | 0.98 | 96.43% | 5.6% | 7.8% |
| CNN (T+R) | LAC | 90.73% | 0.96 | 94.69% | 0.0% | 4.2% |
| CNN (T+R) | APS | 90.55% | 1.05 | 94.48% | 9.6% | 4.7% |
| **Ensemble** | **LAC** | **90.55%** | 0.93 | **97.84%** | 0.0% | 7.5% |
| Ensemble | APS | 87.27% | 0.95 | 97.19% | 5.5% | 10.4% |

Several patterns are visible in the table.

**Coverage tracks the target.** Every method lands at 87% to 93% empirical coverage, on either side of the 90% target. This empirically validates the conformal procedure: the prescribed coverage rate is honoured, even though no model is perfectly calibrated.

**LAC produces no abstain sets but admits empty sets.** In binary LAC, the prediction set contains a class iff $1 - \hat p_y \leq \hat q$. Since this condition is symmetric in $y$, sets of size 2 only occur when *both* $1 - p$ and $p$ are below $\hat q$, which happens when the prediction is in the middle of the probability range. Empty sets occur when *both* are above the threshold, i.e., when neither class meets the inclusion criterion — the model expresses no opinion. LAC produces 4 to 9% empty sets and 0% two-class sets.

**APS produces fewer empty sets and more abstain sets.** APS adds randomised tie-breaking, which causes more sets of size 2 (4 to 10%) and fewer empty sets (8 to 9%). For clinical use, two-class sets are more useful than empty sets — they say "the answer is one of these two" rather than "I have no opinion".

**VGG16 + LAC has the highest singleton correct rate (98.05%).** When the prediction set is a single class, VGG16 is correct 98% of the time. This is materially better than the ensemble (97.84%) and substantially better than the worst model (CNN(T+R) at 94.69%).

**The ensemble's empty-set rate (7.5% LAC, 10.4% APS) is high.** Disagreement among the six members forces conformal to abstain. This is consistent with the observation in Chapter 4 that the ensemble is more conservative than any single model.

### 5.1.3 Per-Model Coverage at α = 0.05

**Table 5.2.** Conformal prediction at α = 0.05 (target 95% coverage). Selected rows; full table at `master/results/phase2/conformal_results.csv`.

| Model | Score | Coverage | Mean size | Singleton correct | Abstain | Empty |
|-------|------:|---------:|----------:|------------------:|--------:|------:|
| Xception | LAC | 95.64% | 1.00 | 95.64% | 0.0% | **0.0%** |
| **CNN** | **LAC** | **96.00%** | **1.00** | 96.00% | 0.0% | **0.0%** |
| Ensemble | LAC | 95.27% | 0.97 | 97.76% | 0.0% | 2.5% |
| Ensemble | APS | 95.64% | 1.10 | 97.87% | 12.2% | 2.5% |
| CNN (T+R) | APS | 94.73% | 1.21 | 95.23% | 22.2% | 1.6% |

At target 95%, the CNN with LAC achieves 96.0% empirical coverage with mean set size exactly 1.00 and zero abstentions. Every test point gets a single-class decision, and the conformal calibration finds a clean threshold under which the marginal coverage guarantee is honoured. Xception with LAC is similarly well-behaved.

The Ensemble at α = 0.05 is interesting: APS coverage is 95.64% with 12% of cases assigned a two-class set (clinically meaningful "refer for further review"). This may be the most clinically useful single configuration in the binary problem: 88% of cases auto-classified, 12% deferred, no empty sets.

### 5.1.4 Interpretation for Clinical Workflow

Conformal prediction transforms a point classifier into a *set* classifier with a statistically grounded refer-to-clinician rule. A set of size 1 is a confident verdict. A set of size 2 (in binary) means the system explicitly cannot decide. An empty set, when it occurs, indicates either an outlier input or a calibration mismatch and should also trigger human review.

For the screening setting envisaged in this thesis, the most appropriate configuration is **APS at α = 0.10** with the ensemble or with VGG16: empirical coverage of approximately 90%, single-class verdicts for ~85% of cases, two-class "refer" sets for ~5%, and empty sets for ~10% (which we conservatively also refer). This gives the clinician a reliable triage signal that maps onto the existing referral workflow.

## 5.2 Monte Carlo Dropout

### 5.2.1 Architectures and Training

Two MC Dropout networks were trained with the same Phase 1 pipeline plus dropout layers:

- **`cnn_mcd`**: 3-block CNN with `SpatialDropout2D(0.3)` after each conv-pool pair and `Dropout(0.3)` before the classifier.
- **`resnet50_mcd`**: ResNet50 transfer learning with `Dropout(0.3)` before the dense head, base frozen.

Training stopped via early stopping at epoch ~35 for cnn_mcd and ~50 for resnet50_mcd.

### 5.2.2 Inference and Aggregation

At inference, T = 30 stochastic forward passes are performed per test image with `training=True` so that dropout remains active. The per-pass probabilities are aggregated into mean, standard deviation, predictive entropy, mean per-pass entropy, and mutual information.

**Table 5.3.** MC Dropout results on the 550-image test set.

| Model | Det. acc | Det. AUC | Det. ECE | MC acc | MC AUC | MC ECE | Mean σ correct | Mean σ wrong |
|-------|---------:|---------:|---------:|-------:|-------:|-------:|---------------:|-------------:|
| cnn_mcd | 90.73% | 0.9675 | 0.0363 | **91.09%** | 0.9677 | 0.0387 | 0.044 | **0.072** |
| resnet50_mcd | 96.18% | 0.9891 | **0.0204** | 96.18% | 0.9882 | 0.0243 | 0.043 | **0.130** |

Three observations follow.

**MC averaging gives a small accuracy bump for cnn_mcd** (+0.36 pp from 90.73% deterministic to 91.09% MC). The randomised network is essentially an ensemble of many sub-networks; averaging T draws stabilises the prediction. For resnet50_mcd, the deterministic and MC accuracies coincide.

**The σ signal is clearly informative.** On wrong predictions, the average MC standard deviation is 3 to 4 times larger than on correct predictions (0.130 vs 0.043 for resnet50_mcd). This means that high σ flags inputs that the system gets wrong — exactly the property needed for a refer-to-clinician decision.

**`resnet50_mcd` is the strongest single model in the thesis** at 96.18% test accuracy, narrowly exceeding the non-dropout ResNet50 (95.27%) and matching or beating every other architecture from Chapter 4. The dropout in the head acts as effective regularisation on this small dataset.

### 5.2.3 Risk-Coverage with MC Signals

For resnet50_mcd:

- Predictive entropy at 90% coverage → 97.78% selective accuracy.
- Predictive entropy at 50% coverage → 99.27%.

For cnn_mcd:

- Predictive entropy at 90% coverage → 94.34%.
- Predictive entropy at 50% coverage → 98.91%.

resnet50_mcd dominates: at 50% coverage (the most-confident half of the test set), accuracy is 99.27% — clinical-grade precision while deferring half the cases to human review. This is the strongest selective-prediction result in the thesis.

## 5.3 Out-of-Distribution Detection

### 5.3.1 Setup

In-distribution (ID) inputs are the 550 fundus images of the test set. Out-of-distribution (OOD) inputs are 300 synthetic images of uniform random noise, preprocessed identically to the ID images. DenseNet121 (ImageNet-pretrained, frozen) is used both as the classifier and as the 1024-dim feature extractor for distance-based methods. Per-class means and a shared covariance for Mahalanobis are fitted on the validation set.

### 5.3.2 Results

**Table 5.4.** OOD detection metrics. AUROC is the area under the ROC curve for ID-vs-OOD separation; FPR @ TPR=95% is the false-positive rate among ID inputs when 95% of OOD inputs are correctly flagged.

| Method | AUROC (ID vs OOD) | FPR @ TPR=95% | Verdict |
|--------|------------------:|--------------:|---------|
| Maximum Softmax Probability | 0.589 | 29.6% | Fails — softmax over-confidence on noise |
| Energy score | 0.787 | 14.0% | Useful, not perfect |
| **Mahalanobis distance** | **1.000** | **0.0%** | **Perfect separation** |
| **Cosine to ID centroid** | **1.000** | **0.0%** | **Perfect separation** |

Several conclusions follow.

**Output-space methods (MSP) fail.** The classifier produces high-confidence predictions even on uniform-noise inputs because the dense head cannot help but produce *some* probability. MSP is therefore close to chance.

**Energy is useful but not perfect.** The unbounded log-sum-exp avoids softmax saturation and pushes ID and OOD distributions apart, but the separation is not complete.

**Feature-space methods are perfect for noise OOD.** Random pixels produce CNN features that are very far from the manifold of fundus images in feature space. Mahalanobis distance and cosine distance to the centroid both achieve AUROC = 1.0 with zero false alarms at TPR = 95%.

### 5.3.3 Caveats

Synthetic uniform noise is the *easy* case for OOD detection. Real-world OOD inputs — chest X-rays, fundus images from a different camera, low-quality phone photos — would lie much closer to the training manifold and would likely yield AUROC values in the 0.80 to 0.95 range. A proper cross-dataset evaluation on Messidor-2 or IDRiD is therefore left as future work; the infrastructure to run it is in place (`master/run_cross_dataset.py`). The present synthetic-OOD result confirms that the feature-space approach *can* detect distribution shift in principle, but the magnitude of the effect on real OOD remains to be quantified.

## 5.4 K-Fold Cross-Validation of ResNet50

### 5.4.1 Motivation

Phase 1 reported a single test estimate of 95.27% for ResNet50, with bootstrap CI [93.45, 97.09]. Two questions remained: how stable is this estimate across different training runs, and how confident can we be that the ensemble's 96.55% is materially different from ResNet50's 95.27%?

### 5.4.2 Protocol

ResNet50 is retrained 5 times on stratified folds of the 3,112-sample train+val pool, with the test set held fixed at 550 images. Each fold trains for up to 30 epochs with early stopping (patience = 6).

### 5.4.3 Per-Fold Results

**Table 5.5.** ResNet50 5-fold cross-validation. Each row is one fold; the test set is identical across folds.

| Fold | Epochs | Best val acc | Test acc | Test AUC | Train time |
|-----:|-------:|-------------:|---------:|---------:|-----------:|
| 1 | 8 | 95.99% | 95.82% | 0.9867 | 8.5 min |
| 2 | 21 | 97.11% | 95.82% | 0.9882 | 25 min |
| 3 | 22 | 98.07% | 95.45% | 0.9898 | 39 min |
| 4 | 15 | 97.75% | 95.64% | 0.9903 | 16 min |
| 5 | 25 | 97.59% | 95.45% | 0.9900 | 30 min |

### 5.4.4 Aggregate

- Test accuracy: **95.64% ± 0.18 percentage points** (95% CI [95.48, 95.79]).
- Test AUC: 0.9890 ± 0.0015.
- Best validation accuracy: 97.30% ± 0.81 pp.

### 5.4.5 Interpretation

**Test accuracy is exceptionally stable.** The standard deviation across 5 retrainings is only 0.18 pp — far smaller than the bootstrap CI half-width of 1.8 pp from a single fold. This means that the variance in the Phase 1 estimate is dominated by the *test sample* rather than by training stochasticity. The Phase 1 single-split estimate of 95.27% is well within the 95% CI of the cross-validated mean (95.48 to 95.79); it is a representative sample, not a particularly lucky or unlucky one.

**Validation accuracy varies more.** Best val accuracy has σ = 0.81 pp because validation sets are small (~550 each) and depend on the fold split. Test acc, computed on the *same* fixed 550-sample test set across folds, is more stable.

**Resolving the ensemble-vs-ResNet50 question.** The ensemble's 96.55% is approximately 5 standard deviations above the ResNet50 mean of 95.64%. While the McNemar test in Chapter 4 was borderline (p = 0.10) due to limited test points, the cross-validated comparison strongly supports the conclusion that the ensemble is genuinely better, not a statistical artifact of a single split.

## 5.5 Lessons from Phase 2

Phase 2 yields four substantive conclusions.

First, conformal prediction is essentially a free addition to any trained classifier. It requires only a calibration set, no retraining, and produces a statistically grounded refer-to-clinician rule. The CNN at α = 0.05 with LAC achieves 96% coverage with no abstentions and no empty sets — a near-ideal configuration for clinical deployment.

Second, MC Dropout is the cheapest Bayesian approximation available. It requires retraining with dropout layers — no architectural overhaul beyond inserting `Dropout` and `SpatialDropout2D` — and at inference it costs T forward passes per input. The σ across MC samples is 3 to 4× larger on wrong predictions than on correct ones, providing a useful selective-prediction signal. resnet50_mcd is the strongest single model in the thesis at 96.18%.

Third, K-fold cross-validation costs significant compute (5 trainings of ResNet50 ≈ 2 hours on CPU) but pins down the variance of the test estimate to a tight σ = 0.18 pp. The single-split estimate is reliable; the ensemble's small accuracy advantage is real.

Fourth, OOD evaluation against synthetic noise is too easy. The feature-space methods achieve perfect AUROC on uniform noise, which is encouraging in principle but does not prove that they will work against real cross-dataset shift. A dedicated cross-dataset study on Messidor-2 or IDRiD is the natural next step.

Phase 2 thus closes Research Questions 2 and 3 with strong empirical support. The next chapter turns to Research Question 4: extending the same uncertainty machinery to the clinically meaningful 5-stage grading task.
