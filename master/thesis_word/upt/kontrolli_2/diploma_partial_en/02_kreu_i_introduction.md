# KREU I

# INTRODUCTION

## 1.1 Problem Description

Diabetic retinopathy (DR) is a microvascular complication of diabetes mellitus and remains one of the leading causes of preventable blindness in working-age adults worldwide. The disease arises when sustained hyperglycaemia damages the small blood vessels supplying the retina; the resulting cycle of vascular leakage, ischaemia and aberrant neovascularisation, if undetected, ends in retinal detachment and irreversible vision loss [1]. Because the early stages are largely asymptomatic, regular fundus screening is essential — yet the global prevalence of diabetes (estimated at over 537 million people in 2024) far exceeds the capacity of trained ophthalmologists, especially in low-income and rural settings. Automated screening tools that can triage patients before specialist review have therefore become an active research area.

The clinical staging system used by ophthalmologists distinguishes five severity levels: No DR (Stage 0), Mild non-proliferative (Stage 1), Moderate non-proliferative (Stage 2), Severe non-proliferative (Stage 3), and Proliferative DR (Stage 4). Each step has different management implications, ranging from observation in the mild stage to urgent vitreoretinal intervention in the proliferative stage. A clinically useful screening tool must therefore not only discriminate "DR present" from "DR absent" but also indicate severity in a way that maps onto the existing referral and treatment pathway.

A meaningful screening pipeline must address three deeper questions that any deployment of such a system would have to answer:

1. Are the differences between candidate models statistically meaningful, or are they sampling noise?
2. When a model is wrong, does it know? Is its confidence calibrated, and does it provide an uncertainty signal that can drive human-in-the-loop decisions?
3. What does the system do when the input is not a fundus image, or when the patient sits between two severity stages?

The present thesis treats these three questions as its primary research agenda.

## 1.2 Motivation: Why Uncertainty Matters in Medical AI

Modern convolutional networks are notoriously over-confident. A binary classifier that assigns "DR" with probability 0.99 may be wrong much more often than 1% of the time, particularly under distribution shift or for atypical inputs. In benign domains (advertising, recommendation), such miscalibration is mildly inconvenient. In screening medicine, it is dangerous: a confidently incorrect "No DR" can delay vision-saving intervention, while a confidently incorrect "DR" can trigger unnecessary referrals that erode clinical trust in the tool [2].

Recent regulatory developments have begun to make uncertainty quantification a requirement rather than a research nicety. The U.S. Food and Drug Administration's framework for Software as a Medical Device, the European Union's Artificial Intelligence Act (in force from 2026), and the World Health Organization's guidance on AI for health all stipulate that high-risk medical AI must provide evidence of calibration, robustness to out-of-distribution inputs, and interpretable confidence estimates. A retinal screening pipeline that emits only an "accuracy = 96%" figure is not deployable under these frameworks; one that exposes calibrated probabilities, conformal prediction sets, and feature-space novelty scores is.

A second, equally important argument is clinical. Ophthalmologists routinely refer ambiguous cases for second opinions, and the entire structure of telehealth retinal screening is built around triage — confident cases are auto-resolved while uncertain ones are escalated. A system that produces a single argmax verdict with no explicit "I don't know" output cannot integrate cleanly into this workflow. The system developed in this thesis will explicitly produce three layers of output:

1. A point prediction with a hard decision threshold.
2. A calibrated probability for each class, validated by reliability diagrams and post-hoc temperature scaling.
3. A conformal prediction set that, with provable coverage guarantee, contains the true label at least 90% (or 95%) of the time.

Combined, these allow a downstream clinical workflow to filter cases by confidence, escalating only the genuinely ambiguous ones to a specialist.

## 1.3 Research Questions and Contributions

This thesis is organised around four research questions:

**RQ1.** Do alternative deep architectures differ in performance on the APTOS 2019 task once preprocessing and training are properly configured, and what is the role of model ensembling?

**RQ2.** How well-calibrated are the trained models, and to what extent does post-hoc temperature scaling reduce calibration error?

**RQ3.** What is the most useful uncertainty signal for selective prediction and out-of-distribution detection, and can a conformal prediction wrapper provide a finite-sample coverage guarantee?

**RQ4.** Does the same uncertainty machinery generalise from binary classification to the clinically meaningful 5-stage grading task, and what is the resulting clinical workflow?

The planned cumulative contributions of this work are summarised below:

1. A reproducible, statistically rigorous evaluation pipeline for the APTOS 2019 dataset, with stratified splits, architecture-specific preprocessing, balanced class weighting, and unified training callbacks. *(Completed.)*

2. A calibration analysis for six DR classifiers, expected to reveal previously unreported over-confidence in VGG16 and demonstrate a ~40% relative ECE reduction via temperature scaling. *(In progress.)*

3. The first application, to our knowledge, of split conformal prediction with formal coverage guarantees to multi-stage DR grading on APTOS 2019. *(In progress.)*

4. A side-by-side comparison of Monte Carlo Dropout, deep-ensemble disagreement, and feature-space OOD detection on a single benchmark. *(In progress.)*

5. A 5-stage grading model expected to achieve QWK ≈ 0.85, competitive with public APTOS 2019 leaderboards, with conformal sets that translate into approximately a 70%/30% auto-classify/refer split clinically. *(Planned — Phase 3.)*

6. An open-source Streamlit application that surfaces calibrated probabilities, ensemble disagreement, and conformal prediction sets to end users in real time. *(Functional prototype.)*

## 1.4 Methodology Overview

The dataset is the Kaggle APTOS 2019 Blindness Detection collection [1]: 3,662 retinal fundus images at 224 × 224 pixels, each labelled by an experienced clinician on the 0–4 severity scale. All experiments use a single, fixed stratified 70/15/15 train/validation/test split with random seed 123, so every metric reported in the thesis is computed on the same 550 held-out test images.

All training is performed on a CPU-only consumer laptop (Apple Silicon M-series, 8 GB RAM), with TensorFlow 2.15. No GPU was used. This is a deliberate choice: it constrains the experiments to architectures and training regimes that are realistically deployable in low-resource settings, including the kind of computer used in primary-care clinics.

Code is organised into four top-level Python packages: `lib/` for the inference and Streamlit user interface, `scripts/` for the unified training driver, `master/uncertainty/` for the calibration, conformal, ensemble, OOD and Monte Carlo Dropout modules, and `master/run_*.py` for the analysis scripts that produced every CSV, JSON and figure in this document. The complete codebase is open-source.

## 1.5 Thesis Outline

The remainder of the thesis is organised as follows:

**Kreu II** surveys the clinical and technical background needed to motivate the methodology: the pathophysiology and grading of DR, the convolutional architectures used as backbones, calibration of deep classifiers, Bayesian approximations via Monte Carlo Dropout, the conformal prediction framework, and out-of-distribution detection.

**Kreu III** describes the unified experimental protocol — dataset, splits, augmentation, training callbacks, evaluation metrics — and the implementation of each uncertainty module.

**Kreu IV** reports and discusses the experimental results. In this Second Control draft, Phase 1 (binary classification and preliminary calibration analysis) is included; Phase 2 (conformal prediction, MC Dropout, K-fold, and OOD detection) and Phase 3 (multi-class grading) are under active development and will be included in the final submission.

**Kreu V** (planned) will conclude with a summary of contributions, clinical implications, limitations, and avenues for future work.

\newpage
