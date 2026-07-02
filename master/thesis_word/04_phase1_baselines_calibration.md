# CHAPTER 4: BINARY BASELINES, CALIBRATION, AND ENSEMBLING (PHASE 1)

This chapter reports the first set of experimental results, addressing Research Questions 1 and 2 from the introduction. Section 4.1 presents per-model test accuracy with bootstrap confidence intervals. Section 4.2 reports pairwise McNemar significance tests. Section 4.3 covers calibration analysis and post-hoc temperature scaling. Section 4.4 introduces the heterogeneous ensemble and selective prediction. Section 4.5 summarises the lessons learned from Phase 1.

## 4.1 Per-Model Performance with Confidence Intervals

The six binary architectures introduced in Chapter 2 were trained with the unified pipeline of Chapter 3. Table 4.1 reports test accuracy with bootstrap 95% confidence intervals (1,000 resamples), together with AUROC, raw and temperature-scaled ECE, MCE, and the fitted temperature.

**Table 4.1.** Per-model test performance with bootstrap 95% confidence intervals on the held-out test set (N = 550). ECE is reported before and after temperature scaling; T is the temperature fitted on the validation set.

| Model | Test acc | 95% CI | AUC | ECE raw | ECE TS | MCE | T |
|-------|---------:|:------:|----:|--------:|-------:|----:|--:|
| ResNet50 | 95.27% | [93.45, 97.09] | 0.9888 | 0.0199 | 0.0182 | 0.3515 | 1.19 |
| Xception | 95.45% | [93.82, 97.09] | 0.9852 | 0.0174 | 0.0224 | 0.5767 | 1.16 |
| DenseNet121 | 95.64% | [94.00, 97.27] | 0.9919 | 0.0288 | 0.0269 | 0.4470 | 1.03 |
| VGG16 | 95.64% | [94.00, 97.27] | 0.9900 | 0.0255 | **0.0153** | 0.7397 | 1.44 |
| **CNN** | **96.00%** | [94.36, 97.45] | 0.9867 | 0.0317 | 0.0317 | 0.4229 | 1.00 |
| CNN (Tanh+ReLU) | 92.91% | [90.72, 94.91] | 0.9699 | 0.0460 | 0.0415 | 0.3871 | 1.21 |
| **Ensemble (6 models)** | **96.55%** | — | **0.9906** | 0.0283 | — | — | — |

Several observations follow immediately from the table.

The five strong models — ResNet50, Xception, DenseNet121, VGG16, and the from-scratch CNN — cluster between 95.27% and 96.00% test accuracy. Their bootstrap confidence intervals overlap substantially, with no model's lower bound exceeding any other model's upper bound. Visually, this can be displayed as horizontal error bars on a bar chart (Figure 4.1, generated automatically by `master/generate_thesis_figures.py`).

The from-scratch CNN, with only 11 million parameters, ties or marginally beats the much larger transfer-learning models. This is plausibly due to the close match between its training-time grayscale-replicated input and the dominant red-channel structure of fundus images: the network is learning from a representation that is already biased towards retinal pathology, which the ImageNet-pretrained backbones are not.

The CNN (Tanh+ReLU) variant is consistently the weakest model, with a test accuracy 3 percentage points below the others and a confidence interval that is fully separate from those of the strong models. The bachelor thesis reported the two CNN variants as approximately equivalent; the present analysis shows that they are, in fact, statistically distinguishable.

The MCE column is informative: VGG16's MCE of 0.74 is dramatically larger than its ECE of 0.026. This indicates that VGG16's calibration is good *on average* but very poor in *some* confidence range — a small bin where the model is heavily over-confident. The reliability diagram in Figure 4.2 confirms this: the high-confidence end of the bar plot is well below the diagonal.

Finally, the ensemble of all six members achieves 96.55%, beating every individual model. This improvement is small (~0.55 pp over the best member) but consistent with the general result that ensembling reduces variance.

## 4.2 Statistical Significance: Pairwise McNemar Tests

Table 4.2 reports pairwise McNemar tests between every pair of classifiers, using a 2×2 contingency table of paired predictions on the test set. The statistic uses a continuity correction; p-values below 0.05 indicate a statistically significant difference in error rates between the two classifiers.

**Table 4.2.** Pairwise McNemar test results. *b* counts test points where model A is right and B is wrong; *c* vice versa. Statistically significant differences (p < 0.05) are bolded.

| Comparison | b | c | χ² | p-value | Significant? |
|------------|--:|--:|---:|--------:|:-------------|
| ResNet50 vs Xception | 11 | 12 | 0.00 | 1.000 | No |
| ResNet50 vs DenseNet121 | 8 | 10 | 0.06 | 0.814 | No |
| ResNet50 vs VGG16 | 5 | 7 | 0.08 | 0.773 | No |
| ResNet50 vs CNN | 8 | 12 | 0.45 | 0.502 | No |
| **ResNet50 vs CNN(T+R)** | 24 | 11 | 4.11 | **0.043** | **Yes** |
| ResNet50 vs Ensemble | 3 | 10 | 2.77 | 0.096 | Borderline |
| Xception vs DenseNet121 | 11 | 12 | 0.00 | 1.000 | No |
| Xception vs VGG16 | 11 | 12 | 0.00 | 1.000 | No |
| Xception vs CNN | 7 | 10 | 0.24 | 0.628 | No |
| **Xception vs CNN(T+R)** | 25 | 11 | 4.69 | **0.030** | **Yes** |
| DenseNet121 vs VGG16 | 7 | 7 | 0.00 | 1.000 | No |
| DenseNet121 vs CNN | 10 | 12 | 0.05 | 0.831 | No |
| **DenseNet121 vs CNN(T+R)** | 26 | 11 | 5.30 | **0.021** | **Yes** |
| VGG16 vs CNN | 10 | 12 | 0.05 | 0.831 | No |
| **VGG16 vs CNN(T+R)** | 25 | 10 | 5.60 | **0.018** | **Yes** |
| **CNN vs CNN(T+R)** | 19 | 2 | 12.19 | **0.00048** | **Highly Yes** |
| CNN vs Ensemble | 3 | 6 | 0.44 | 0.505 | No |
| **CNN(T+R) vs Ensemble** | 2 | 22 | 15.04 | **0.00011** | **Highly Yes** |

The pattern is striking. Among the five strong models — ResNet50, Xception, DenseNet121, VGG16 and CNN — every pairwise p-value exceeds 0.5. None of the differences in Table 4.1 between these five models are statistically significant. Choosing one over another on the basis of test accuracy alone is therefore unjustified at the sample size of this study.

By contrast, every comparison involving the CNN(Tanh+ReLU) variant against any of the other five models yields p < 0.05, with p-values ranging from 0.018 (vs VGG16) to 0.043 (vs ResNet50). This strongly supports the conclusion that the substitution of tanh for ReLU in the second convolutional block and the dense head materially harms performance — a result that the bachelor thesis did not report because it lacked statistical testing.

The ensemble's advantage over its members is borderline: vs ResNet50, p = 0.096; vs CNN, p = 0.50. With only 550 test points, the McNemar test cannot conclusively distinguish a 0.55 pp improvement from noise. Section 5.4 of Chapter 5 returns to this question with a 5-fold cross-validation that puts a tighter bound on the ResNet50 estimate.

A side observation: the CNN vs CNN(T+R) comparison has b = 19, c = 2. That is, on 19 test images, the all-ReLU CNN was correct and the tanh+ReLU variant was wrong; on only 2 images was the situation reversed. This is the strongest signal in the table and explains the very low p-value (0.00048).

## 4.3 Calibration Analysis

### 4.3.1 ECE Before and After Temperature Scaling

The ECE columns of Table 4.1 indicate that the trained models are reasonably well-calibrated already, with raw ECE values ranging from 0.017 to 0.046. None of the models exhibits the severe over-confidence reported in some studies of medical CNNs.

Temperature scaling, fitted on the validation set, produces small but consistent improvements for ResNet50, DenseNet121, and CNN(T+R), and a substantial improvement for VGG16, whose ECE drops from 0.0255 to 0.0153 — a 40% relative reduction. The fitted temperature for VGG16 is 1.44, indicating that the raw model was significantly over-confident.

Two cases are interesting:

- **CNN**: T = 1.00 to within numerical tolerance. The from-scratch CNN was already calibrated. Temperature scaling provides no benefit because the raw probabilities already track frequencies well. This is consistent with the general finding that simpler architectures with less regularisation are better calibrated by default.
- **Xception**: ECE *increases* slightly under temperature scaling (from 0.0174 to 0.0224). This is unusual but not unprecedented. It can occur when the validation set is small enough that the temperature fit overfits to validation-specific calibration patterns that do not generalise to the test set. A larger validation set or cross-validated temperature fit would likely eliminate this effect.

### 4.3.2 Reliability Diagrams

Reliability diagrams for each model, before and after temperature scaling, are provided as supplementary figures (Appendix B, Figures B.1–B.7) and saved at `master/results/reliability_<model>_{raw,temp_scaled}.png`. The general pattern is mild over-confidence in the high-confidence bins, with VGG16 being the most extreme example. After temperature scaling, the bars settle close to the diagonal across the entire confidence range.

### 4.3.3 Practical Implications

For deployment, temperature scaling is essentially free: it requires fitting one scalar parameter on a calibration set, and it does not change the argmax prediction (so accuracy is exactly preserved). The 40% ECE reduction for VGG16 alone justifies its routine application. For Xception, where TS marginally hurts, the operator can choose to skip it; the raw probabilities are good enough.

## 4.4 Heterogeneous Ensemble and Selective Prediction

### 4.4.1 Ensemble Construction

A heterogeneous ensemble is built by averaging the per-model probabilities $p_k(x)$ across the six binary classifiers:

$$\bar p(x) = \frac{1}{6} \sum_{k=1}^{6} p_k(x).$$

The ensemble's binary decision uses threshold 0.5. Test accuracy is 96.55%, AUROC is 0.9906, and ECE is 0.0283.

Importantly, the ensemble can also be queried for *disagreement*-based uncertainty signals:

- **Mean probability** $\bar p$ — the point prediction.
- **Standard deviation across members** — an epistemic-like spread; high values indicate that the constituent models disagree on the input.
- **Predictive entropy** $H(\bar p)$ — total uncertainty.
- **Mean per-member entropy** $\mathbb{E}_k[H(p_k)]$ — aleatoric proxy.
- **Mutual information** = predictive entropy − mean per-member entropy.
- **Vote agreement** — fraction of members predicting the majority class.

These signals are saved to `master/results/summary.json` and used in the next subsection.

### 4.4.2 Risk-Coverage Curves

A risk-coverage curve plots selective accuracy as a function of coverage. Coverage is the fraction of test points retained after sorting by ascending uncertainty and rejecting the most uncertain. At 100% coverage (no rejection), selective accuracy equals the ensemble accuracy of 96.55%. At lower coverages, accuracy on the auto-classified subset rises.

**Table 4.3.** Selective accuracy of the ensemble at varying coverage, for each uncertainty signal.

| Coverage | Std (epistemic) | Predictive entropy | Mutual information | 1 − max prob |
|---------:|----------------:|-------------------:|-------------------:|-------------:|
| 50% | 99.64% | 99.64% | 99.27% | 99.64% |
| 60% | 99.39% | 99.39% | 99.39% | 99.39% |
| 70% | 98.44% | 99.48% | 98.44% | 99.48% |
| 80% | 98.41% | 98.64% | 97.73% | 98.64% |
| 90% | 97.78% | 98.18% | 97.78% | 98.18% |
| 100% | 96.55% | 96.55% | 96.55% | 96.55% |

The clinical interpretation is direct: if the system *defers* the 10% most uncertain cases to a clinician, selective accuracy on the remaining 90% rises from 96.55% to 98.18%. At 50% coverage, selective accuracy reaches 99.64% — close to perfect.

Predictive entropy and 1 − max prob slightly outperform variance and mutual information at low coverages. Mutual information attempts to isolate the *epistemic* component of uncertainty, but for binary classification the predictive entropy is more directly tied to errors and is therefore the most useful signal here.

### 4.4.3 Histogram of Uncertainty for Correct vs Wrong Predictions

A second diagnostic is the histogram of an uncertainty signal split by whether the prediction was correct. If the signal is informative, wrong predictions should systematically have higher uncertainty. Figures 4.3 and 4.4 (saved as `uncertainty_hist_std.png` and `uncertainty_hist_entropy.png` in `master/results/`) confirm this. For the standard-deviation signal, mean σ on correct predictions is 0.04, vs 0.07 on wrong predictions — almost a 2x ratio.

## 4.5 Lessons from Phase 1

Several conclusions follow from Phase 1.

First, the bachelor-level framing — "model X is better than model Y by 0.5 percentage points" — is not supported by the data once proper statistical testing is applied. With 550 test images, a 0.5 pp difference is below the noise floor. The correct framing is that the five strong architectures form an equivalence class on this benchmark. Choosing among them should be guided by other criteria: parameter count (favours DenseNet121), inference speed (favours the from-scratch CNN), or memory footprint (also CNN).

Second, calibration is a free improvement. Even a 40% relative ECE reduction from temperature scaling does not change argmax predictions, so accuracy is preserved exactly. For deployment, temperature scaling should be considered the default rather than an optional add-on.

Third, the ensemble's small accuracy advantage (0.55 pp) is at the edge of significance with the present test set, but its larger contribution is the disagreement-based uncertainty signal. At 90% coverage, predictive entropy lifts selective accuracy from 96.55% to 98.18%. This is a clinically meaningful gain — at the population scale of a screening programme, it corresponds to thousands fewer false negatives and false positives per million screened.

Fourth, a methodological aside: the bachelor thesis's failure to detect the CNN(T+R) underperformance is a consequence of evaluating accuracy without confidence intervals or pairwise tests. Future studies should adopt the McNemar test as a default step.

The next chapter extends this analysis with conformal prediction, Monte Carlo Dropout, K-fold cross-validation, and OOD detection.
