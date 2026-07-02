# KREU IV

# PRELIMINARY RESULTS — PHASE 1

> *Second Control note: this chapter presents the results of Phase 1 (binary classification), which has been completed. Phase 2 (conformal prediction, MC Dropout, K-fold, OOD detection) is under active development. Phase 3 (multi-class grading) is planned after Phase 2 completion. The final synthesis and sensitivity discussion are planned for the final submission.*

This chapter reports the preliminary results of the first experimental phase of the thesis — binary classification on APTOS 2019 with all six deep architectures. Section 4.1 reports per-model performance with confidence intervals. Section 4.2 covers pairwise McNemar statistical tests. Section 4.3 presents the initial calibration analysis. Section 4.4 reports preliminary ensemble results. Section 4.5 illustrates training dynamics. Section 4.6 concludes with an outline of the next steps.

## 4.1 Per-Model Performance with Confidence Intervals

The six binary architectures introduced in Kreu II were trained with the unified pipeline of Kreu III. Table 4.1 reports test accuracy with bootstrap 95% confidence intervals (1,000 resamples), together with AUROC and raw ECE.

::: {custom-style="Table Caption"}
**Table 4.1.** Per-model performance on the held-out test set (N = 550). 95% CIs are computed via bootstrap with 1,000 resamples and seed 42.
:::

| Model | Test acc | 95% CI | AUC | ECE raw |
|-------|---------:|:------:|----:|--------:|
| ResNet50 | 95.27% | [93.45, 97.09] | 0.9888 | 0.0199 |
| Xception | 95.45% | [93.82, 97.09] | 0.9852 | 0.0174 |
| DenseNet121 | 95.64% | [94.00, 97.27] | 0.9919 | 0.0288 |
| VGG16 | 95.64% | [94.00, 97.27] | 0.9900 | 0.0255 |
| **CNN** | **96.00%** | [94.36, 97.45] | 0.9867 | 0.0317 |
| CNN (Tanh+ReLU) | 92.91% | [90.72, 94.91] | 0.9699 | 0.0460 |

![**Figure 4.1.** Per-model test accuracy with 95% bootstrap CIs on the held-out test set (N = 550). The five strong models cluster within a 0.7-percentage-point band; CNN(Tanh+ReLU) lies clearly below them.](../../../../thesis/figures/fig_phase1_accuracy_ci.png)

The five strong models — ResNet50, Xception, DenseNet121, VGG16, and the from-scratch CNN — cluster between 95.27% and 96.00% test accuracy. Their bootstrap confidence intervals overlap substantially, with no model's lower bound exceeding any other model's upper bound.

The from-scratch CNN, with only 11 million parameters, ties or marginally beats the much larger transfer-learning models. This is plausibly due to the close match between its training-time grayscale-replicated input and the dominant red-channel structure of fundus images: the network is learning from a representation that is already biased towards retinal pathology, which the ImageNet-pretrained backbones are not.

The CNN (Tanh+ReLU) variant is consistently the weakest model, with a test accuracy 3 percentage points below the others and a confidence interval that is fully separate from those of the strong models.

## 4.2 Statistical Significance: Pairwise McNemar Tests

Table 4.2 reports pairwise McNemar tests between every pair of classifiers, using a 2×2 contingency table of paired predictions on the test set. The statistic uses a continuity correction; p-values below 0.05 indicate a statistically significant difference in error rates between the two classifiers.

::: {custom-style="Table Caption"}
**Table 4.2.** Pairwise McNemar test results (summary). Statistically significant differences (p < 0.05) are bolded.
:::

| Comparison | b | c | χ² | p-value | Significant? |
|------------|--:|--:|---:|--------:|:-------------|
| ResNet50 vs Xception | 11 | 12 | 0.00 | 1.000 | No |
| ResNet50 vs DenseNet121 | 8 | 10 | 0.06 | 0.814 | No |
| ResNet50 vs VGG16 | 5 | 7 | 0.08 | 0.773 | No |
| ResNet50 vs CNN | 8 | 12 | 0.45 | 0.502 | No |
| **ResNet50 vs CNN(T+R)** | 24 | 11 | 4.11 | **0.043** | **Yes** |
| Xception vs DenseNet121 | 11 | 12 | 0.00 | 1.000 | No |
| Xception vs VGG16 | 11 | 12 | 0.00 | 1.000 | No |
| **DenseNet121 vs CNN(T+R)** | 26 | 11 | 5.30 | **0.021** | **Yes** |
| **VGG16 vs CNN(T+R)** | 25 | 10 | 5.60 | **0.018** | **Yes** |
| **CNN vs CNN(T+R)** | 19 | 2 | 12.19 | **0.00048** | **Highly Yes** |

![**Figure 4.2.** Heatmap of pairwise McNemar p-values across the six binary classifiers. Cells with p < 0.05 (statistically significant differences) are highlighted; CNN(Tanh+ReLU) is the only model that differs significantly from any of the five strong models.](../../../../results/mcnemar_pvalues.png)

The pattern is striking. Among the five strong models — ResNet50, Xception, DenseNet121, VGG16 and CNN — every pairwise p-value exceeds 0.5. None of the differences in Table 4.1 between these five models are statistically significant. Choosing one over another on the basis of test accuracy alone is therefore unjustified at the sample size of this study.

By contrast, every comparison involving the CNN(Tanh+ReLU) variant against any of the other five models yields p < 0.05, with p-values ranging from 0.018 (vs VGG16) to 0.043 (vs ResNet50). This strongly supports the conclusion that the substitution of tanh for ReLU in the second convolutional block and the dense head materially harms performance.

## 4.3 Preliminary Calibration Analysis

The ECE columns of Table 4.1 indicate that the trained models are reasonably well-calibrated already, with raw ECE values ranging from 0.017 to 0.046. None of the models exhibits the severe over-confidence reported in some studies of medical CNNs.

However, deeper analysis — including temperature scaling [2] and reliability diagrams [7] — is currently in progress. The implementation in `master/uncertainty/calibration.py` is complete; analysis of individual bins and per-architecture MCE will be reported in the final submission.

Preliminary note: VGG16 has MCE = 0.7397, dramatically larger than its ECE of 0.0255. This indicates that VGG16's calibration is good on average but very poor in some confidence range — a small bin where the model is heavily over-confident. This will be investigated fully in the final Kreu IV.

## 4.4 Preliminary Ensemble Results

A simple heterogeneous ensemble [9] is built by averaging the per-model probabilities across all six binary classifiers. Test accuracy is **96.55%**, AUROC is **0.9906**.

::: {custom-style="Table Caption"}
**Table 4.3.** Heterogeneous ensemble of 6 binary models on the held-out test set.
:::

| Metric | Value |
|--------|------:|
| Accuracy | 96.55% |
| AUROC | 0.9906 |
| ECE | 0.0283 |

The ensemble outperforms every individual model (95.27% – 96.00%), as predicted by theory. The improvement is small (~0.55 pp over the best member), but it is consistent with the general result that ensembling reduces variance.

Deeper analysis — selective accuracy as a function of ensemble disagreement, risk-coverage diagrams, and comparison of uncertainty signals (predictive entropy vs mutual information vs std) — is in progress and will be presented in the final Kreu IV.

## 4.5 Training Dynamics

To verify that each binary classifier converged appropriately and was halted by the early-stopping callback, the train and validation curves of accuracy and loss were recorded epoch-by-epoch.

![**Figure 4.3.** ResNet50 training dynamics — accuracy (left) and loss (right). Validation accuracy stabilises around 0.95–0.96 within the first six epochs.](../../../../../results/resnet50_accuracy.png)

![**Figure 4.4.** DenseNet121 training dynamics. The dense-connectivity backbone reaches ~96% validation accuracy quickly and shows a smoother loss trajectory.](../../../../../results/densenet121_accuracy.png)

![**Figure 4.5.** From-scratch 3-block CNN training dynamics. Without pretrained weights the model takes longer to converge than the transfer-learning backbones, but eventually achieves comparable validation accuracy.](../../../../../results/cnn_accuracy.png)

![**Figure 4.6.** CNN(Tanh+ReLU) variant training dynamics. Compared with the all-ReLU baseline of Figure 4.5, the mixed-activation variant shows visibly slower convergence and a larger train–validation gap, consistent with its statistically significant 3-percentage-point gap on the test set.](../../../../../results/tanh_relu_accuracy.png)

In every case the training accuracy continues to climb after the validation accuracy plateaus, which is the canonical pattern that early stopping with `restore_best_weights=True` is designed to handle. The patience-10 setting halted most runs between epochs 15 and 35.

## 4.6 Next Steps

Phase 1 has confirmed two central findings: (i) deep architectures are statistically indistinguishable on the binary task once preprocessing and class weighting are properly configured, and (ii) a heterogeneous ensemble provides a small but consistent improvement. These results establish the baseline on which the uncertainty methods will be built.

Active work in progress for the final Kreu IV:

- **Full calibration analysis**: temperature scaling for all six architectures, raw vs TS reliability diagrams, per-architecture MCE reporting, evaluation of the ECE improvement.
- **Conformal prediction** (Phase 2): evaluation of both LAC and APS [12] scoring, following the split conformal framework [10, 11], two coverage levels (α = 0.10 and 0.05), set-size diagnostics, and per-class conditional coverage.
- **Monte Carlo Dropout** [8] (Phase 2): training of `cnn_mcd` and `resnet50_mcd`, analysis with T = 30 forward passes, uncertainty decomposition into epistemic and aleatoric components, comparison of performance against the deterministic models.
- **K-Fold cross-validation** (Phase 2): five-fold for ResNet50, reporting mean ± std as confirmation of stability.
- **OOD detection** (Phase 2): evaluation of all four methods (MSP [13], Energy [14], Mahalanobis [15], Cosine) against 300 synthetic OOD images, reporting AUROC and FPR @ TPR = 95%.
- **Classical classifiers** (extended Phase 1): evaluation of DT, RF, SVM on 1024-dim DenseNet features.
- **Phase 3 — Multi-class**: reformulating the task as 5-stage grading, training of `cnn_5class` and `resnet50_5class`, computing QWK and ordinal distance, multi-class conformal with per-class coverage.
- **Final synthesis**: cross-phase comparison, full answer to RQ1-RQ4, sensitivity analysis with respect to design choices, statement of the limits of empirical validation.

Prior work on uncertainty quantification in DR — including MCD selective prediction [16], calibrated regression in medical imaging [17], and the JAMA benchmark of CNN-based DR detection [18] — provides the broader context within which these results will be positioned. The detailed timeline for this work is presented in the accompanying Second Control progress report.

\newpage
