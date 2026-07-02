# CHAPTER 2: BACKGROUND

This chapter establishes the clinical and technical background needed to motivate the methodology of the thesis. Section 2.1 reviews the pathophysiology and clinical grading of diabetic retinopathy and describes the APTOS 2019 dataset. Section 2.2 summarises the convolutional architectures used as backbones in this thesis. Sections 2.3 to 2.6 introduce the four pillars of the uncertainty-aware methodology: calibration, Bayesian approximations, conformal prediction, and out-of-distribution detection. Section 2.7 surveys related work that combines these ideas in DR or other medical imaging tasks.

## 2.1 Diabetic Retinopathy and the APTOS 2019 Dataset

### 2.1.1 Pathophysiology

Diabetic retinopathy is a microvascular complication of diabetes mellitus in which prolonged hyperglycaemia damages the small blood vessels supplying the retina. The disease progresses through a cascade of overlapping mechanisms: pericyte loss leading to vascular fragility, microaneurysm formation, increased vascular permeability and consequent macular oedema, capillary occlusion and resulting ischaemia, and finally vascular endothelial growth factor (VEGF)-driven neovascularisation. Each stage produces characteristic findings on fundus examination, summarised in the International Clinical DR Severity Scale used by the American Academy of Ophthalmology and adopted by the APTOS 2019 dataset:

- **Stage 0 — No DR**: no apparent retinal pathology.
- **Stage 1 — Mild non-proliferative DR**: presence of a small number of microaneurysms only.
- **Stage 2 — Moderate non-proliferative DR**: more than just microaneurysms but less than severe NPDR; may include hard exudates and dot/blot haemorrhages.
- **Stage 3 — Severe non-proliferative DR**: any of the "4-2-1" criteria (more than 20 intraretinal haemorrhages in each of four quadrants; or definite venous beading in two or more quadrants; or prominent intraretinal microvascular abnormalities in one or more quadrants).
- **Stage 4 — Proliferative DR (PDR)**: neovascularisation of the disc or elsewhere, with or without preretinal/vitreous haemorrhage.

Treatment escalates accordingly. Patients in Stage 0 or Stage 1 are typically scheduled for routine annual review. Stage 2 may merit a closer follow-up interval. Stages 3 and 4 require referral to an ophthalmologist, with anti-VEGF injections and panretinal photocoagulation as the cornerstones of treatment in Stage 4 to prevent retinal detachment and irreversible vision loss. The clinical implication for an automated screening system is therefore clear: a binary "DR / No DR" output is informationally lossy. A patient with Mild DR and a patient with PDR receive the same label but very different management.

### 2.1.2 APTOS 2019 Blindness Detection Dataset

The dataset used throughout this thesis is the APTOS 2019 Blindness Detection collection, hosted on Kaggle by the Asia Pacific Tele-Ophthalmology Society. It comprises 3,662 retinal fundus images, each cropped and resized to 224×224 pixels and labelled by an experienced clinician on the five-stage scale described above. The class distribution is heavily skewed toward Stage 0: approximately half of the images carry the No DR label, while the more clinically urgent Stages 3 and 4 together account for less than 14%. Any successful classifier on this dataset must therefore handle severe class imbalance, either through class-weighted loss, oversampling, or focal-style loss functions.

For the binary version of the task, Stages 1 to 4 are collapsed into a single "DR" class, yielding a roughly 50/50 split. This is the formulation used in the bachelor thesis. The present master work returns to the original 5-class formulation in Chapter 6 because of its higher clinical relevance.

## 2.2 Convolutional Architectures

The thesis evaluates six deep architectures and three classical classifiers. The deep architectures are summarised here; the classical methods are introduced in Section 3.6 of the methodology chapter.

### 2.2.1 ResNet50

ResNet50 [He et al., 2016] introduced residual learning with skip connections that bypass two or three convolutional layers. The architecture has 50 layers organised into five stages, with approximately 25.6 million parameters. Skip connections mitigate the vanishing-gradient problem and enable training of much deeper networks than the previous state of the art. In the present thesis, ResNet50 is used as a transfer-learning backbone with ImageNet pretrained weights and a 128-unit dense head.

### 2.2.2 DenseNet121

DenseNet121 [Huang et al., 2017] takes the idea of skip connections to an extreme: every layer in a dense block receives the concatenated outputs of all preceding layers. The 121-layer variant has only ~7 million parameters thanks to the parameter-sharing afforded by feature reuse. Dense connectivity also produces an implicit deep-supervision effect that encourages feature reuse and shortens gradient paths.

### 2.2.3 Xception

Xception [Chollet, 2017] replaces standard convolutions with depthwise separable convolutions that factorise spatial and channel-wise convolution. The architecture has 36 layers organised into Entry, Middle, and Exit flows, with ~21 million parameters. Despite its extreme decomposition of the standard convolution, Xception consistently performs at or above the level of much larger architectures on ImageNet.

### 2.2.4 VGG16

VGG16 [Simonyan and Zisserman, 2015] is a simpler, deeper architecture using only 3×3 convolutions and 2×2 max-pooling. Its uniformity makes it a popular pedagogical and transfer-learning choice. In the bachelor thesis, VGG16 was trained from scratch; in the present work it is evaluated as a transfer-learning backbone, which proves more accurate.

### 2.2.5 The 3-block Custom CNN

To anchor the comparison, the thesis also includes a from-scratch CNN with three convolution-pool blocks (32 → 64 → 128 filters) followed by a 128-unit dense layer and a sigmoid output. The model has ~11 million parameters and is trained on grayscale-replicated input. A second variant (CNN Tanh+ReLU) substitutes tanh for ReLU in the second convolutional block and the dense layer; the bachelor study reported that this combination performed competitively, but the master-level statistical analysis presented in Chapter 4 shows that it is in fact significantly worse than the all-ReLU variant.

## 2.3 Calibration of Deep Classifiers

A classifier is said to be well-calibrated if its predicted probabilities match observed frequencies: among the inputs for which the model outputs probability 0.8, approximately 80% should be correctly classified. Modern deep networks are typically *not* well-calibrated; Guo et al. [2017] showed that even high-accuracy networks tend to be over-confident, with the gap widening as architectures grow deeper and use stronger regularisation.

### 2.3.1 Expected and Maximum Calibration Error

The standard scalar measure of calibration is the **Expected Calibration Error (ECE)**:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where the predictions are partitioned into M equal-width confidence bins $B_m$, $\text{acc}(B_m)$ is the empirical accuracy in bin $m$, and $\text{conf}(B_m)$ is the average confidence in the bin. ECE = 0 indicates perfect calibration; in practice, values below 5% are considered acceptable, while values above 10% indicate clinically problematic over-confidence.

The **Maximum Calibration Error (MCE)** replaces the weighted sum with a maximum over bins; it captures the worst-case calibration gap and is sensitive to small badly-calibrated regions of the confidence range that ECE would average out.

### 2.3.2 Reliability Diagrams

Reliability diagrams plot per-bin accuracy against per-bin mean confidence. The diagonal corresponds to perfect calibration; bars below the diagonal indicate over-confidence and bars above indicate under-confidence. A useful augmentation is to mark the deviation from the diagonal with a coloured stem, scaled by bin size, making it easy to spot regions where the model is most miscalibrated.

### 2.3.3 Temperature Scaling

Temperature scaling [Guo et al., 2017] is a one-parameter post-hoc fix in which the pre-sigmoid logit $z$ is divided by a temperature $T$ before applying the sigmoid:

$$\hat p = \sigma(z / T)$$

Values of $T > 1$ soften the predictions (reducing peaked confidence), while $T < 1$ sharpens them. The temperature is fitted by minimising negative log-likelihood on a held-out validation set. Crucially, $T$ does not change the argmax prediction, so accuracy is preserved exactly; only the confidence is rescaled.

## 2.4 Bayesian Deep Learning and Monte Carlo Dropout

The Bayesian view of a neural network treats the weights as random variables with a posterior distribution over the training data. Predicting on a new input then requires marginalising over the posterior, which is intractable in closed form. A range of approximations have been proposed; the most computationally accessible is **Monte Carlo Dropout (MCD)**, introduced by Gal and Ghahramani [2016].

### 2.4.1 MC Dropout as Variational Inference

Gal and Ghahramani showed that a network trained with dropout regularisation can be interpreted as performing variational inference under a particular approximating family. At inference time, dropout is left active and the network is run $T$ times for the same input, producing a distribution of predictions. The mean of the $T$ samples is the point estimate, while the variance is an approximation to the model's epistemic (i.e., reducible-with-more-data) uncertainty.

### 2.4.2 Decomposition of Uncertainty

For binary classification with sigmoid output $p$, the predictive entropy of the mean prediction $H(\bar p)$ measures **total uncertainty**, while the mean per-sample entropy of the dropout samples $\mathbb{E}_t[H(p_t)]$ approximates the **aleatoric** (i.e., data-inherent) component. Their difference is the **mutual information** between the prediction and the model parameters, sometimes called the BALD score after the Bayesian Active Learning by Disagreement framework. High mutual information means the dropout samples agree on the input being hard to classify, but disagree on which class it belongs to — the canonical signature of *epistemic* uncertainty.

### 2.4.3 Deep Ensembles

A complementary, even simpler approach is to train several independent copies of the same architecture with different random initialisations [Lakshminarayanan et al., 2017]. The ensemble's mean prediction is typically more accurate and better-calibrated than any single member. The disagreement among members provides an uncertainty signal directly analogous to the MC Dropout variance.

The present thesis uses both MCD (Chapter 5) and a heterogeneous ensemble across the six binary architectures (Chapter 4). The latter is cheaper than a deep ensemble of identical members because the models already exist as a by-product of the bachelor-level architecture comparison.

## 2.5 Conformal Prediction

Conformal prediction is a framework, originally due to Vovk et al. [2005] and recently popularised by Angelopoulos and Bates [2021], for producing *prediction sets* with finite-sample, distribution-free coverage guarantees. Given a target miscoverage rate $\alpha$, conformal prediction produces, for each test input $x$, a set $C(x)$ such that

$$\Pr\left( y \in C(x) \right) \geq 1 - \alpha$$

under the assumption that the calibration and test data are exchangeable. Crucially, the guarantee holds *no matter what underlying classifier* is wrapped — even an uncalibrated, badly-trained model yields a statistically valid prediction set after conformal wrapping. The guarantee is a marginal one (averaged over all test inputs), although stratified variants exist.

### 2.5.1 Split (Inductive) Conformal Prediction

The version used in this thesis is **split conformal prediction**:

1. Hold out a calibration set $\{(x_i, y_i)\}_{i=1}^n$, disjoint from the training set.
2. For each calibration sample, compute a non-conformity score $s_i$ that quantifies how surprising the true label is under the trained classifier.
3. Compute the empirical $(1 - \alpha)$ quantile $\hat q$ of the scores, with a finite-sample correction $\lceil (n+1)(1 - \alpha) \rceil / n$.
4. For a test input $x$, include in $C(x)$ every candidate label whose non-conformity score is $\leq \hat q$.

### 2.5.2 Score Functions

Two non-conformity scores are evaluated in this thesis:

- **Least Ambiguous Classifier (LAC)**: $s(x, y) = 1 - \hat p_y(x)$. Simple and well-studied; gives marginal coverage.
- **Adaptive Prediction Sets (APS)** [Romano et al., 2020]: sort the per-class probabilities in descending order; the score for the true class is the cumulative probability up to and including it, with random tie-breaking. APS produces smaller average set sizes and tends to be more clinically useful because it can produce two-class "abstain" sets ({No DR, Mild}) for ambiguous inputs.

### 2.5.3 Set-Size Distribution and Conditional Coverage

Beyond the marginal coverage guarantee, two diagnostics are reported throughout the thesis: the **mean set size**, which captures how often the wrap is uncertain, and the **per-class conditional coverage**, which checks that the guarantee is honoured uniformly across true classes rather than concentrated on the easy majority class.

## 2.6 Out-of-Distribution Detection

A clinically deployed system will inevitably receive inputs outside its training distribution: chest X-rays mistakenly uploaded to a retina classifier, fundus images from a different camera with different colour calibration, or simply low-quality images. The model should refuse to predict on such inputs rather than confidently output a wrong answer.

### 2.6.1 Output-Space Methods

The simplest baseline is the **Maximum Softmax Probability (MSP)** [Hendrycks and Gimpel, 2017]: OOD inputs tend to produce lower confidence than ID inputs, so $\max_k \hat p_k$ can be thresholded. Despite its simplicity, MSP suffers when softmax saturates — a network can be forced into very high confidence even on noise.

The **Energy score** [Liu et al., 2020] sidesteps softmax saturation by computing the negative log-sum-exp of the logits. ID inputs produce lower energy than OOD inputs, and the score is unbounded above, which avoids the ceiling effect of MSP.

### 2.6.2 Feature-Space Methods

A more powerful family of methods compares the position of the test input in feature space to the distribution of training features. **Mahalanobis distance** [Lee et al., 2018] fits per-class Gaussian densities to the training features (with a shared covariance for stability) and scores test inputs by the negative log-likelihood under the closest class-conditional Gaussian. **Cosine distance to the in-distribution centroid** is a simpler variant. For inputs far from the training manifold (e.g., uniform noise), feature-space methods dramatically outperform output-space methods, often achieving perfect AUROC on synthetic OOD.

## 2.7 Related Work in DR with Uncertainty

A growing body of work applies uncertainty estimation to DR detection. Leibig et al. [2017] used MC Dropout on EyePACS images and showed that selective prediction substantially improves the precision of automated screening. Laves et al. [2020] reported well-calibrated regression uncertainty in medical imaging with a focus on retinal layer thickness.

However, very few studies have applied conformal prediction to DR, and even fewer have done so on the APTOS 2019 dataset. The present thesis closes this gap by combining (i) a thorough calibration analysis, (ii) two independent uncertainty mechanisms (MCD and ensemble disagreement), (iii) split conformal prediction with formal coverage at two confidence levels, and (iv) feature-space OOD detection — all on the same fixed test set so that the methods are directly comparable.

The methodological choices and their justification are described in detail in the next chapter.
