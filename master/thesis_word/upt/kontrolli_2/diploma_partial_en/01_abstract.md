# ABSTRACT

**UNCERTAINTY-AWARE DIABETIC RETINOPATHY GRADING USING CONFORMAL PREDICTION AND BAYESIAN DEEP LEARNING**

*Draft status: Second Control — approximately 50% complete*

This thesis aims to develop an uncertainty-aware system for the automated grading of diabetic retinopathy (DR) from retinal fundus images. While modern convolutional neural networks routinely achieve accuracies above 95% on the Kaggle APTOS 2019 Blindness Detection benchmark, deployment in clinical screening remains hindered by their tendency to produce confidently incorrect predictions. This work argues that a screening tool intended for non-specialist clinicians must do more than maximise accuracy: it must also know, and communicate, when its predictions are unreliable.

The methodology establishes a statistically rigorous evaluation protocol over the 3,662-image APTOS 2019 dataset, with stratified 70/15/15 train/validation/test splits, architecture-specific input pre-processing, class-balanced loss weighting, and a unified training pipeline. **The first experimental phase has been completed**: six deep architectures are evaluated as binary classifiers — ResNet50, Xception, DenseNet121, VGG16, a from-scratch convolutional neural network, and a CNN variant combining tanh and ReLU activations. Once preprocessing and class weighting are corrected, the five strongest deep models cluster between **95.27% and 96.00% test accuracy**, and pairwise McNemar tests show that they are **statistically indistinguishable**. A heterogeneous ensemble of all six members achieves **96.55%**.

**Phases two and three are under active development**: implementation of calibration analysis with temperature scaling, split conformal prediction with finite-sample coverage guarantees, Monte Carlo Dropout with T = 30 stochastic forward passes, K-fold cross-validation, and out-of-distribution detection are in progress. The third phase — reformulating the task to the original 5-stage clinical scale (No DR, Mild, Moderate, Severe, PDR) with quadratic-weighted kappa as the headline metric — is planned for after Phase 2 completion.

The expected cumulative effect is a transition from a single accuracy number to a transparent, statistically grounded clinical decision-support pipeline, with final completion expected by September 2026.

**Keywords:** diabetic retinopathy, deep learning, conformal prediction, Monte Carlo Dropout, calibration, out-of-distribution detection, multi-class severity grading

\newpage
