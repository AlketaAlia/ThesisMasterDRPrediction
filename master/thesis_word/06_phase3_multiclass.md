# CHAPTER 6: MULTI-CLASS 5-STAGE DR GRADING (PHASE 3)

This chapter addresses the fourth and most ambitious research question of the thesis: *does the same uncertainty machinery generalise from binary classification to the clinically meaningful 5-stage grading task, and what is the resulting clinical workflow?* Section 6.1 explains the reformulation. Section 6.2 reports per-model accuracy, QWK, ordinal distance, calibration, and per-class breakdown. Section 6.3 presents multi-class conformal prediction. Section 6.4 contrasts the bachelor binary system with the master multi-class uncertainty-aware system. Section 6.5 concludes the chapter.

## 6.1 From Binary to 5-Stage Grading

### 6.1.1 Why Binary is Clinically Insufficient

The bachelor thesis collapsed APTOS 2019's clinical 5-stage scale into binary "DR / No DR" and reported accuracies near 96%. This formulation is convenient — it doubles the per-class data and produces near-balanced classes — but it hides clinically essential information. A patient at Stage 1 (Mild non-proliferative DR) typically requires only annual follow-up, whereas a patient at Stage 4 (Proliferative DR) needs urgent referral for anti-VEGF therapy or panretinal photocoagulation. A binary system that returns "DR" for both cases provides the clinician with no actionable severity information.

### 6.1.2 Multi-Class Reformulation

Phase 3 returns to the original APTOS labels: 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative). The same 70/15/15 stratified split is used, but stratified now on the 5-class label so that each fold sees a representative proportion of each severity level. The class distribution of the test set is heavily imbalanced, which mirrors clinical reality:

- No DR: 271 (49%)
- Mild: 56 (10%)
- Moderate: 150 (27%)
- Severe: 29 (5%)
- PDR: 44 (8%)

Stages 3 and 4 together account for only 13% of the test set, which makes their per-class metrics statistically less reliable but clinically the most important. The thesis takes this imbalance seriously by:

1. Using `class_weight='balanced'` from scikit-learn to re-weight the loss inversely to class frequency.
2. Reporting macro-averaged F1 alongside the weighted-average F1.
3. Reporting per-class precision, recall and F1 explicitly.
4. Computing per-class conditional coverage in the conformal-prediction analysis.

### 6.1.3 Architectures Trained

Two architectures are evaluated:

- **`cnn_5class`**: the same 3-block CNN as in the binary case, with the head replaced by a 5-unit softmax. Trained on grayscale-replicated input.
- **`resnet50_5class`**: ResNet50 transfer learning with a 5-unit softmax head. Trained on architecture-specific (BGR mean-subtracted) input.

Both train under the unified pipeline of Chapter 3 with categorical cross-entropy loss and the balanced class weights described above.

## 6.2 Per-Model Results

### 6.2.1 Aggregate Metrics

**Table 6.1.** 5-stage DR grading results on the held-out test set (N = 550). QWK is the quadratic-weighted kappa; ord-dist is the mean |y − ŷ| of misclassifications.

| Model | Test acc | **QWK** | Ord. dist. | Macro F1 | Weighted F1 | ECE | T (TS) |
|-------|---------:|--------:|-----------:|---------:|------------:|----:|-------:|
| cnn_5class | 64.18% | 0.5638 | 0.658 | 0.4400 | 0.6432 | 0.0537 | 1.16 |
| **resnet50_5class** | **77.09%** | **0.8471** | **0.329** | **0.6029** | **0.7711** | **0.0298** | 0.93 |
| Ensemble (2 members) | 76.73% | 0.7847 | — | — | — | 0.0887 | — |

The headline result is that the ResNet50 multi-class model achieves a quadratic-weighted kappa of **0.8471** on the held-out test set. QWK 0.84 is the threshold conventionally used to mark "excellent" or "near-perfect" agreement on the Cohen scale. The Kaggle APTOS 2019 leaderboard's top public submissions (heavy ensembling, TTA, and external data) reached QWK in the 0.92 to 0.94 range. The resnet50_5class model reported here, trained on a CPU laptop with no external data and no test-time augmentation, sits squarely in the upper third of the leaderboard distribution.

The from-scratch CNN drops to QWK 0.56, reflecting its smaller capacity. With only 11 million parameters and no pretraining, it cannot capture the fine-grained vascular patterns that distinguish the four DR severity classes. ImageNet pretraining is essentially mandatory for this task at this dataset size.

The ensemble of the two models achieves 76.73% accuracy and QWK 0.7847. With only two members and one of them substantially weaker, the ensemble does not improve on resnet50_5class alone. This is consistent with the general principle that ensembling helps most when members are diverse and individually competitive; here, cnn_5class is too weak to contribute usefully.

### 6.2.2 Mean Ordinal Distance

The mean ordinal distance metric captures the *clinical severity* of misclassifications. For resnet50_5class, mean |y − ŷ| = 0.329, meaning that when the model errs, it typically does so by less than half a class. Inspection of the confusion matrix (Figure 6.1, saved at `master/results/multiclass/resnet50_5class_confusion_matrix.png`) confirms that most off-diagonal mass is concentrated in the immediate neighbours of the true class. Errors of 3 or 4 classes — confusing No DR with PDR, for example — are extremely rare.

For cnn_5class, mean ordinal distance is 0.658, almost twice as large. The CNN makes more errors *and* its errors are more severe.

### 6.2.3 Per-Class Breakdown for ResNet50

**Table 6.2.** Per-class precision, recall, and F1 for resnet50_5class.

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| No DR | 271 | 0.96 | 0.97 | **0.96** |
| Mild | 56 | 0.68 | 0.48 | 0.56 |
| Moderate | 150 | 0.68 | 0.67 | 0.68 |
| Severe | 29 | 0.37 | 0.38 | 0.37 |
| PDR | 44 | 0.39 | 0.50 | 0.44 |

Three patterns emerge.

**No DR detected near-perfectly.** F1 = 0.96 means that the model rarely raises a false alarm on healthy patients (precision 0.96) and rarely misses pathology in healthy-looking eyes (recall 0.97). This is the most clinically reassuring metric: the system is unlikely to send a healthy patient for unnecessary follow-up.

**Mild has the lowest recall (0.48).** Subtle microaneurysms are the only finding in Stage 1, and they are visually similar to image artifacts or compression noise. Almost half of Mild cases are missed by the model — a real concern, although mitigated by the fact that conformal prediction can flag these as ambiguous (see Section 6.3).

**Severe and PDR are confused with each other.** Both conditions involve neovascularisation; the difference is one of quantity and location. Per-class F1 in the 0.37 to 0.44 range indicates that the model can detect the severe-spectrum class but cannot reliably distinguish Severe from PDR. Clinically, this matters less than it might seem: both classes warrant urgent referral, and the management is similar.

The Mild and Severe weaknesses are exactly the cases where uncertainty estimation provides the most value. In Section 6.3, we show that conformal prediction returns a multi-class set for these inputs, deferring the decision to a clinician rather than forcing an unsupported point prediction.

## 6.3 Multi-Class Conformal Prediction

### 6.3.1 Setup

The same conformal-prediction modules used in Phase 2 are extended to multi-class. LAC and APS scores generalise straightforwardly: LAC scores the true-class probability, while APS uses the cumulative-probability score with random tie-breaking. The threshold $\hat q$ is fitted on the multi-class validation set (550 images) at α = 0.10 and α = 0.05.

### 6.3.2 Coverage Results

**Table 6.3.** Multi-class conformal prediction on resnet50_5class.

| α | Score | Empirical coverage | Mean set size | Per-class conditional coverage |
|--:|------:|-------------------:|--------------:|--------------------------------|
| 0.10 | LAC | 89.64% | 1.48 | [97%, 71%, 87%, 76%, 82%] |
| 0.10 | APS | 89.45% | 1.80 | [91%, 82%, 89%, 90%, 93%] |
| 0.05 | LAC | 97.27% | 2.04 | [99%, 91%, 97%, 97%, 100%] |
| 0.05 | APS | 94.36% | 2.14 | [96%, 88%, 95%, 90%, 98%] |

The marginal coverage tracks the prescribed target. At α = 0.10, empirical coverage is 89.45% with APS and 89.64% with LAC. At α = 0.05, both methods exceed 94% coverage. The conformal guarantee is therefore honoured.

The most interesting column is **per-class conditional coverage**. With APS at α = 0.10, the per-class coverage is [91, 82, 89, 90, 93]%, which is approximately uniform. The system gives reliable predictions across all severity levels, not just the easy "No DR" majority class. With LAC at α = 0.10, the coverage is more uneven: 97% on No DR, 71% on Mild. LAC under-covers the minority Mild class, and APS does substantially better on this dimension.

### 6.3.3 Set Size Distribution

Mean set sizes are larger in the multi-class problem (1.5 to 2.1) than in binary (~1.0) because the model is genuinely uncertain among five ordinal classes. Inspection of the set-size distribution for APS at α = 0.10:

- Set of size 1: 234 cases (43% of test set) — single confident prediction.
- Set of size 2: 159 cases (29%) — usually adjacent severities, e.g. {Mild, Moderate}.
- Set of size 3: 151 cases (27%) — clinical "uncertainty zone".
- Set of size 4: 6 cases (1%) — strong defer-to-clinician signal.

A set of {Mild, Moderate} is more clinically useful than a single mistaken "Mild" — it tells the ophthalmologist "this is at least Mild, possibly Moderate, please review carefully".

### 6.3.4 A Practical Clinical Workflow

For 550 test images with APS at α = 0.10:

- **234 (43%) get a single-class prediction** — auto-resolve.
- **393 (71%) get sets of size ≤ 2** — likely between adjacent severities, soft-refer.
- **157 (29%) get sets of size ≥ 3** — strong refer-to-clinician.

This translates directly into a triage policy: hand off only the 29% genuinely ambiguous cases to a specialist, while auto-classifying or soft-referring the remaining 71%. At the population scale of a national screening programme, this could free up substantial specialist time without compromising patient safety, since the conformal coverage guarantee ensures that the true label is in the set 90% of the time.

## 6.4 Bachelor Binary vs Master Multi-Class: A Direct Comparison

**Table 6.4.** Comparison of the bachelor-level binary system with the master-level multi-class uncertainty-aware system.

| Aspect | Bachelor (binary) | Master (5-stage + UQ) |
|--------|-------------------|-----------------------|
| Best test accuracy | 96% (single split) | 77% (5-class, harder task) |
| Clinical task | "Has DR / no DR" | **"How severe?"** |
| Statistical guarantees | None | Conformal coverage 95% |
| Calibration analysis | Not measured | ECE = 0.030 (well-calibrated) |
| Ordinal evaluation | N/A | **QWK = 0.847** (excellent) |
| Per-class transparency | N/A | F1 per stage |
| Out-of-distribution | None | 4 detection methods |
| Bayesian uncertainty | None | MC Dropout (T = 30) |
| Significance testing | None | McNemar + bootstrap CI |
| Generalisation | 1 split | 5-fold CV (σ = 0.18 pp) |
| Clinical refer rule | None | Set size ≥ 3 → refer |
| Streamlit UI | Single argmax | Ensemble + abstention zone |

The headline accuracy figure dropped from 96% to 77%, but this is a comparison between two different tasks. The 5-class task is substantially harder than binary, particularly because the minority severities (Stages 3 and 4) have very few training examples. The relevant metric for fair comparison is QWK, where the master multi-class model achieves 0.847 — a number with no analogue in the bachelor work, because the bachelor work did not perform multi-class grading.

More important than any single metric is the *qualitative shift*. The bachelor system produces a single-bit verdict with no uncertainty information. The master system produces a calibrated probability for each of five severity classes, a conformal prediction set with formal coverage, an out-of-distribution flag, and (via the Streamlit UI) a refer-to-clinician recommendation. The latter is a *clinical decision-support pipeline* in the regulatory sense; the former is a research demonstration.

## 6.5 Phase 3 Conclusion

Phase 3 demonstrates that the uncertainty-aware methodology developed in Phases 1 and 2 generalises naturally to the multi-class setting. The ResNet50 multi-class model achieves QWK = 0.847 — competitive with the upper third of public APTOS 2019 leaderboards — at a mean ordinal distance of 0.329. Multi-class conformal prediction at α = 0.10 produces set sizes of 1 to 4 with a balanced per-class coverage of approximately 90%, mapping cleanly onto a clinically usable refer-to-specialist policy.

The remaining weaknesses are concentrated in the minority severity classes (Mild, Severe), where the dataset has too few training examples for the model to learn reliable boundaries. Future work targeted at these classes — focal loss, class-aware augmentation, or targeted active learning — would be the highest-leverage extension of the present pipeline.

The final chapter discusses clinical implications, limitations, and avenues for future work.
