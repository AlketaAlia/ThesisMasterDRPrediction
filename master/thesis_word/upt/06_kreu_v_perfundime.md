# KREU V

# CONCLUSIONS

This chapter consolidates the contributions of the thesis, discusses clinical implications, acknowledges the limitations of the present work, and proposes avenues for future research.

## 5.1 Summary of Contributions

This thesis set out to develop a clinically grounded, statistically rigorous, uncertainty-aware multi-class grading system for diabetic retinopathy on the APTOS 2019 dataset. The work was carried out in three experimental phases.

**Phase 1** built the binary classification pipeline with stratified train/validation/test splits, architecture-specific input preprocessing, balanced class weighting, and unified training callbacks. The five strong architectures cluster between 95.27% and 96.00% test accuracy and are statistically indistinguishable from one another (every pairwise McNemar p > 0.5). The CNN(Tanh+ReLU) variant is shown to be significantly worse (p < 0.05 against every strong model). A heterogeneous ensemble across all six binary architectures achieves 96.55% test accuracy. Calibration analysis reveals that all models are reasonably well-calibrated (ECE < 5%), with VGG16 benefiting most from temperature scaling (40% relative ECE reduction). At 90% coverage, ensemble selective prediction lifts accuracy from 96.55% to 98.18%.

**Phase 2** introduced four advanced uncertainty mechanisms. Split conformal prediction with both LAC and APS scores produced empirical coverage that tracks the prescribed targets at α = 0.10 and α = 0.05. The CNN at α = 0.05 achieves 96% coverage with no abstentions and zero empty sets, the cleanest configuration of any model. Two Monte Carlo Dropout networks were retrained: cnn_mcd reached 91.09% test accuracy after T = 30 stochastic forward passes, and resnet50_mcd reached 96.18% — the highest accuracy of any single model in the thesis. The MC standard deviation is 3 to 4 times larger on incorrect predictions than on correct ones, providing a strong selective-prediction signal. A 5-fold cross-validation of ResNet50 gave a tight estimate of 95.64 ± 0.18 percentage points, ruling out single-split luck. Out-of-distribution detection against synthetic noise inputs showed that feature-space methods (Mahalanobis distance, cosine to centroid) achieve perfect separation (AUROC = 1.0).

**Phase 3** reformulated the task to the original 5-stage clinical scale. The ResNet50 multi-class model reached **QWK = 0.847**, a "near-perfect" agreement on the Cohen scale and competitive with the upper third of the public APTOS 2019 Kaggle leaderboard. Mean ordinal distance was 0.329, indicating that misclassifications are typically only one severity class away from the true label. Multi-class conformal prediction at α = 0.10 with APS produced a marginal coverage of 89.45% and a balanced per-class conditional coverage of [91, 82, 89, 90, 93]%. The implied clinical workflow — auto-classify 71% of cases, refer 29% to a specialist — matches the way ophthalmologists already triage in tele-screening programmes.

## 5.2 Clinical Implications

The most important conclusion of the thesis is that a screening tool deployed in clinical practice should not return a single argmax verdict. It should return:

1. A calibrated probability over the five severity classes.
2. A conformal prediction set whose coverage rate is configurable to the clinical risk tolerance (typically 90% to 95%).
3. An out-of-distribution flag that triggers when the input is not a fundus image, when image quality is too low, or when the input lies outside the training distribution.
4. A clear refer-to-clinician recommendation when any of the above signals warrant human review.

The infrastructure developed in this thesis — and the open-source Streamlit application that exposes all four signals — demonstrates that this is feasible without retraining state-of-the-art models. The marginal cost of uncertainty quantification, once the model is trained, is small: temperature scaling fits a single scalar; conformal prediction fits a single threshold; MC Dropout requires retraining with dropout layers but no architectural overhaul. Compared with the cost of a clinical study, these are negligible.

The implied division of labour with a screening programme is also worth noting. With APS at α = 0.10 producing a 71%/29% auto-classify/refer split, an ophthalmologist embedded in a screening programme would handle only the 29% genuinely ambiguous cases — a substantial reduction in workload that could enable screening at scale in low-resource settings.

## 5.3 Limitations

The thesis has several limitations that should be made explicit.

**Single dataset.** All experiments were performed on APTOS 2019 [1]. Real-world deployment requires evaluation across multiple data sources — different cameras, different patient populations, different screening protocols. The scaffolding for cross-dataset evaluation is in place but the experiments themselves are left for future work, since they require downloading and curating Messidor-2 [41] or IDRiD [42].

**Synthetic OOD only.** The OOD detection experiments used uniform-noise inputs as the OOD source. This is the easy case for distance-based methods. Realistic OOD scenarios — chest X-rays, low-quality phone photos, fundus images from a different camera — would lie much closer to the training manifold and would likely yield AUROC values in the 0.80 to 0.95 range rather than the 1.00 reported here. A targeted study on real OOD inputs is the natural next step.

**Minority class performance is weak.** The 5-class model has F1 = 0.37 on Severe and F1 = 0.44 on PDR. With only 29 and 44 test images respectively, the per-class metrics also have wide confidence intervals. A larger and more balanced dataset would substantially improve these numbers.

**CPU-only training.** All experiments ran on a CPU laptop, which constrained the architectures and training regimes that were feasible. State-of-the-art APTOS submissions use heavy ensembling, test-time augmentation, and external pretraining, none of which were attempted here. The relative advantage of the present pipeline lies in its uncertainty quantification, not in raw accuracy.

**Single random seed for splits.** The train/val/test split uses a fixed random_state = 123. While the 5-fold CV of ResNet50 confirmed that the test estimate is stable across training-data perturbations, the test set itself is fixed.

**No clinician validation.** The "refer-to-clinician" decisions implied by the conformal-set policy have not been validated by an actual clinician. A retrospective study in which an ophthalmologist reviews the deferred 29% would establish whether the system's notion of "ambiguous" matches clinical intuition.

## 5.4 Future Work

Several extensions of the present work would substantially strengthen its clinical case.

**Cross-dataset evaluation.** The infrastructure is in place; the next concrete step is to download Messidor-2 [41] or IDRiD [42] and run the cross-dataset evaluation script. Reporting QWK and conformal coverage on the second dataset will establish the system's domain-shift robustness.

**Clinician-in-the-loop validation.** A retrospective study in which an ophthalmologist reviews the deferred cases would establish whether the conformal-set policy correctly identifies clinically ambiguous inputs. This study should also collect inter-rater agreement statistics on the deferred cases, which often have weak ground truth in any case.

**Active learning.** The conformal abstention set is a natural candidate for active learning: query labels for the most ambiguous cases first, retrain, and iterate. Initial experiments could be run on the existing dataset by withholding a fraction of training labels and using the conformal sets to drive a labelling sequence.

**Multi-task learning with lesion segmentation.** IDRiD provides pixel-level annotations of microaneurysms, hemorrhages, and exudates. A multi-task model that simultaneously predicts severity grade and segments lesions could exploit these annotations to learn richer representations and produce explanations (e.g., via Grad-CAM-style saliency maps [43]) that ophthalmologists can verify.

**Federated learning.** Privacy-preserving training across multiple clinics is a logical next step for a deployable system. The conformal threshold is per-clinic by construction (it is fitted on a calibration set), so the system already adapts to each deployment site without sharing raw data.

**Vision transformers and self-supervised pretraining.** ViT [44], Swin Transformer, and DINO-style self-supervised pretraining on unlabelled retinal images are likely to outperform the convolutional backbones used here. The uncertainty machinery developed in this thesis is architecture-agnostic and would transfer directly.

**Calibration drift monitoring.** A deployed system needs to monitor calibration over time to detect distribution shift, model degradation, or population drift. The reliability-diagram and ECE infrastructure developed here can be re-run on a rolling production set to flag drift.

**Foundation models for retinal imaging.** The very recent RETFound model [12] — a vision transformer pretrained self-supervised on more than 1.6 million unlabelled retinal images — provides a substantially stronger feature extractor than the ImageNet-pretrained backbones used here. Replacing DenseNet121 with RETFound features for the OOD experiments and the classical-baseline experiments of Section 4.1.6 would likely produce both higher accuracy and stronger feature-space OOD discrimination. This would be a one-day experiment given a GPU and the published RETFound checkpoint.

**Demographic-shift evaluation.** APTOS 2019 ships without per-image demographic metadata, so the per-subgroup performance breakdown that the WHO ethics guidance [37] requires cannot be computed from APTOS alone. EyePACS, in contrast, includes patient-level metadata for ethnicity and age in some releases. Cross-evaluating on EyePACS with demographic stratification would directly address the equity gap.

**Multi-modal integration.** Real screening pathways combine fundus imaging with patient history, glycaemic control, and sometimes optical coherence tomography. A multi-modal model that ingests an HbA1c value alongside the fundus image is likely to outperform a fundus-only model on the harder Mild and Moderate cases, where visible signs are subtle and recent glycaemic control is informative. The uncertainty methodology developed here transfers directly to multi-modal inputs.

**Conformal risk control.** Beyond the marginal coverage guarantee, recent work [24] has extended conformal prediction to control arbitrary loss functions — for example, the cost of a missed PDR diagnosis is far higher than the cost of an unnecessary referral. Implementing a cost-sensitive conformal wrapper would let the deployed thresholds be tuned to the asymmetric clinical costs of false negatives and false positives in a principled way.

**Integration with electronic health records.** Once cleared as a SaMD device under the regulatory framework (FDA SaMD [36], EU AI Act [45], WHO ethics [37]) of Section 2.8, the system would need to integrate with the electronic health-record systems used in screening clinics. The Streamlit application is a research-grade interface; a deployed version would expose the same outputs through HL7 FHIR or DICOM-SR endpoints rather than a web UI.

**Time-to-event modelling.** The single-image classification framing of this thesis ignores the longitudinal dimension of DR: patients are screened repeatedly over time, and the rate of progression is itself diagnostically informative. A time-to-event model that predicts the probability of progression to vision-threatening DR within (say) 24 months from the baseline image would be a natural and clinically more useful output. The infrastructure for this — labelled longitudinal cohorts — is again the bottleneck rather than the methodology.

**Hyperparameter sensitivity study.** A deliberate hyperparameter sweep over learning rate, batch size, dropout rate, conformal α, and temperature-fitting set size would put confidence intervals on the headline numbers and identify the points at which the pipeline is fragile. Section 4.4.1 surveys the most visible of these levers informally; a proper sweep with multiple seeds would strengthen the empirical case.

## 5.5 Concluding Remarks

The thesis demonstrates that a from-scratch CNN and a transfer-learning ResNet50 on APTOS 2019, combined with split conformal prediction, Monte Carlo Dropout, and feature-space out-of-distribution detection, form a complete clinical decision-support pipeline. The headline binary accuracy is statistically indistinguishable across the five strong architectures, so the differentiating value of the system does not come from squeezing additional points of accuracy out of any single classifier. The substantive contribution lies elsewhere: in the addition of statistical guarantees on prediction-set coverage, of calibration analysis, of multi-class severity grading at QWK 0.847, of disagreement-based and Bayesian uncertainty signals, and of an explicit refer-to-specialist rule that maps onto the existing screening workflow.

The cumulative effect is a transition from a black-box classifier reporting a single accuracy figure to a transparent, statistically grounded triage system that knows when it does not know — and communicates that to the clinician.

Modern medical AI must do more than maximise accuracy. It must know when to defer.
