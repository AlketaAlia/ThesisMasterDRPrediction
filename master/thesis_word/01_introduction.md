# CHAPTER 1: INTRODUCTION

## 1.1 Problem Description

Diabetic retinopathy (DR) is a microvascular complication of diabetes mellitus and remains one of the leading causes of preventable blindness in working-age adults worldwide. The disease arises when sustained hyperglycaemia damages the small blood vessels supplying the retina; the resulting cycle of vascular leakage, ischaemia and aberrant neovascularisation, if undetected, ends in retinal detachment and irreversible vision loss. Because the early stages are largely asymptomatic, regular fundus screening is essential — yet the global prevalence of diabetes (estimated at over 537 million people in 2024) far exceeds the capacity of trained ophthalmologists, especially in low-income and rural settings. Automated screening tools that can triage patients before specialist review have therefore become an active research area.

The clinical staging system used by ophthalmologists distinguishes five severity levels: No DR (Stage 0), Mild non-proliferative (Stage 1), Moderate non-proliferative (Stage 2), Severe non-proliferative (Stage 3), and Proliferative DR (Stage 4). Each step has different management implications, ranging from observation in the mild stage to urgent vitreoretinal intervention in the proliferative stage. A clinically useful screening tool must therefore not only discriminate "DR present" from "DR absent" but also indicate severity in a way that maps onto the existing referral and treatment pathway.

The bachelor thesis on which the present master work builds approached the same dataset, the Kaggle APTOS 2019 Blindness Detection collection, as a binary classification task: "DR" vs "No DR". Six deep architectures and three classical classifiers were compared, and an XGBoost-style narrative around a winning model was reported. While these results were sufficient at the bachelor level, they leave open three deeper questions that any deployment of such a system would have to answer:

1. **Are the reported differences between models real, or are they sampling noise?** No statistical significance test was performed.
2. **When the model is wrong, does it know?** No calibration analysis or uncertainty estimation was provided.
3. **What does the system do when the input is not a fundus image, or when the patient sits between two severity stages?** No abstention or out-of-distribution mechanism was implemented.

The present thesis treats these three questions as its primary research agenda.

## 1.2 Motivation: Why Uncertainty Matters in Medical AI

Modern convolutional networks are notoriously over-confident. A binary classifier that assigns "DR" with probability 0.99 may be wrong much more often than 1% of the time, particularly under distribution shift or for atypical inputs. In benign domains (advertising, recommendation), such miscalibration is mildly inconvenient. In screening medicine, it is dangerous: a confidently incorrect "No DR" can delay vision-saving intervention, while a confidently incorrect "DR" can trigger unnecessary referrals that erode clinical trust in the tool.

Recent regulatory developments have begun to make uncertainty quantification a *requirement* rather than a research nicety. The U.S. Food and Drug Administration's framework for Software as a Medical Device, the European Union's Artificial Intelligence Act (in force from 2026), and the World Health Organization's guidance on AI for health all stipulate that high-risk medical AI must provide evidence of calibration, robustness to out-of-distribution inputs, and interpretable confidence estimates. A retinal screening pipeline that emits only an "accuracy = 96%" figure is not deployable under these frameworks; one that exposes calibrated probabilities, conformal prediction sets, and feature-space novelty scores is.

A second, equally important argument is clinical. Ophthalmologists routinely refer ambiguous cases for second opinions, and the entire structure of telehealth retinal screening is built around triage — confident cases are auto-resolved while uncertain ones are escalated. A system that produces a single argmax verdict with no explicit "I don't know" output cannot integrate cleanly into this workflow. The system developed in this thesis explicitly produces three layers of output:

1. A point prediction, as in the bachelor work.
2. A calibrated probability for each class, validated by reliability diagrams and post-hoc temperature scaling.
3. A conformal prediction set that, with provable coverage guarantee, contains the true label at least 90% (or 95%) of the time.

Combined, these allow a downstream clinical workflow to filter cases by confidence, escalating only the genuinely ambiguous ones to a specialist.

## 1.3 Research Questions and Contributions

This thesis is organised around four research questions, each addressed in a dedicated chapter:

**RQ1.** *Do the six architectures evaluated in the bachelor thesis differ in performance once preprocessing and training are corrected, and what is the role of model ensembling?*
Chapter 4 retrains each of the six binary classifiers with architecture-specific input preprocessing, balanced class weighting, and stratified train/validation/test splits. Bootstrap confidence intervals on test accuracy and pairwise McNemar tests are used to assess significance. A heterogeneous ensemble across all six members is constructed.

**RQ2.** *How well-calibrated are the trained models, and to what extent does post-hoc temperature scaling reduce calibration error?*
Chapter 4 also reports Expected and Maximum Calibration Error, reliability diagrams (raw and after temperature scaling), and per-model temperatures fitted on the validation set.

**RQ3.** *What is the most useful uncertainty signal for selective prediction and out-of-distribution detection, and can a conformal prediction wrapper provide a finite-sample coverage guarantee?*
Chapter 5 implements split conformal prediction with both LAC and APS scores, Monte Carlo Dropout retraining for the CNN and ResNet50 backbones, and four out-of-distribution scoring methods (Maximum Softmax Probability, Energy, Mahalanobis, cosine to ID centroid). A 5-fold cross-validation of ResNet50 quantifies the variability of the test estimate.

**RQ4.** *Does the same uncertainty machinery generalise from binary classification to the clinically meaningful 5-stage grading task, and what is the resulting clinical workflow?*
Chapter 6 retrains the CNN and ResNet50 backbones on the original 5-class labels, evaluates them with quadratic-weighted kappa and ordinal distance, and constructs multi-class conformal prediction sets that map directly onto a refer-to-specialist rule.

The cumulative contributions are summarised below.

1. A reproducible, statistically rigorous evaluation pipeline for the APTOS 2019 dataset, addressing the methodological gaps of the bachelor study (test-set augmentation, lack of stratification, missing class weights, mismatched preprocessing).
2. A calibration analysis for six DR classifiers, revealing previously unreported over-confidence in VGG16 and demonstrating a 40% relative ECE reduction via temperature scaling.
3. The first application, to our knowledge, of split conformal prediction with formal coverage guarantees to multi-stage DR grading on APTOS 2019.
4. A side-by-side comparison of Monte Carlo Dropout, deep-ensemble disagreement, and feature-space OOD detection on a single benchmark.
5. A 5-stage grading model achieving QWK = 0.847, competitive with public APTOS 2019 leaderboards, with conformal sets that translate into a 71%/29% auto-classify/refer split clinically.
6. An open-source Streamlit application that surfaces calibrated probabilities, ensemble disagreement, and conformal prediction sets to end users in real time.

## 1.4 Methodology Overview

The dataset is the same as in the bachelor thesis: 3,662 retinal fundus images at 224×224 pixels, sourced from the Kaggle APTOS 2019 competition. All experiments use a single, fixed stratified 70/15/15 train/validation/test split with random seed 123, so every metric reported in the thesis is computed on the *same* 550 held-out test images. The bachelor pipeline used an 80/20 split with no separate validation set; the present work adds an explicit validation set used for early stopping, learning-rate scheduling, temperature scaling, and conformal threshold fitting.

All training is performed on a CPU-only consumer laptop (Apple Silicon M-series, 8 GB RAM), with TensorFlow 2.15. No GPU was used. This is a deliberate choice: it constrains the experiments to architectures and training regimes that are realistically deployable in low-resource settings, including the kind of computer used in primary-care clinics.

Code is organised into four top-level Python packages: `lib/` for the inference and Streamlit user interface, `scripts/` for the unified training driver, `master/uncertainty/` for the calibration, conformal, ensemble, OOD and Monte Carlo Dropout modules, and `master/run_*.py` for the analysis scripts that produced every CSV, JSON and figure in the present document. The complete codebase is open-source.

## 1.5 Thesis Outline

The remainder of the thesis is organised as follows. Chapter 2 surveys the clinical and technical background needed to motivate the methodology: the pathophysiology and grading of DR, the convolutional architectures used as backbones, calibration of deep classifiers, Bayesian approximations via Monte Carlo Dropout, the conformal prediction framework, and out-of-distribution detection. Chapter 3 describes the unified experimental protocol — dataset, splits, augmentation, training callbacks, evaluation metrics — and the implementation of each uncertainty module. Chapters 4, 5 and 6 report the results of the three experimental phases. Chapter 7 discusses clinical implications, limitations, and avenues for future work. Two appendices document hyperparameters and present additional reliability and confusion-matrix figures.
