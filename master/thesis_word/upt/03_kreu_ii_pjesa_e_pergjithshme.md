# KREU II

# THEORETICAL BACKGROUND

This chapter establishes the clinical and technical background needed to motivate the methodology of the thesis. Section 2.1 reviews the pathophysiology and clinical grading of diabetic retinopathy, describes the APTOS 2019 dataset, and surveys the global epidemiology and existing automated screening systems that motivate the work. Section 2.2 summarises the convolutional architectures used as backbones in this thesis. Sections 2.3 to 2.6 introduce the four pillars of the uncertainty-aware methodology: calibration, Bayesian approximations, conformal prediction, and out-of-distribution detection. Section 2.7 surveys related work that combines these ideas in DR or other medical imaging tasks. Section 2.8 reviews the regulatory landscape (FDA SaMD, EU AI Act, WHO guidance) that increasingly mandates uncertainty disclosure for clinical AI.

## 2.1 Diabetic Retinopathy and the APTOS 2019 Dataset

### 2.1.1 Pathophysiology

Diabetic retinopathy is a microvascular complication of diabetes mellitus in which prolonged hyperglycaemia damages the small blood vessels supplying the retina. The disease progresses through a cascade of overlapping mechanisms: pericyte loss leading to vascular fragility, microaneurysm formation, increased vascular permeability and consequent macular oedema, capillary occlusion and resulting ischaemia, and finally vascular endothelial growth factor (VEGF)-driven neovascularisation. Each stage produces characteristic findings on fundus examination, summarised in the International Clinical DR Severity Scale used by the American Academy of Ophthalmology and adopted by the APTOS 2019 dataset:

- **Stage 0 — No DR**: no apparent retinal pathology.
- **Stage 1 — Mild non-proliferative DR**: presence of a small number of microaneurysms only.
- **Stage 2 — Moderate non-proliferative DR**: more than just microaneurysms but less than severe NPDR; may include hard exudates and dot/blot haemorrhages.
- **Stage 3 — Severe non-proliferative DR**: any of the "4-2-1" criteria (more than 20 intraretinal haemorrhages in each of four quadrants; or definite venous beading in two or more quadrants; or prominent intraretinal microvascular abnormalities in one or more quadrants).
- **Stage 4 — Proliferative DR (PDR)**: neovascularisation of the disc or elsewhere, with or without preretinal/vitreous haemorrhage.

Treatment escalates accordingly. Patients in Stage 0 or Stage 1 are typically scheduled for routine annual review. Stage 2 may merit a closer follow-up interval. Stages 3 and 4 require referral to an ophthalmologist, with anti-VEGF injections and panretinal photocoagulation as the cornerstones of treatment in Stage 4 to prevent retinal detachment and irreversible vision loss [3]. The clinical implication for an automated screening system is therefore clear: a binary "DR / No DR" output is informationally lossy. A patient with Mild DR and a patient with PDR receive the same label but very different management.

### 2.1.2 APTOS 2019 Blindness Detection Dataset

The dataset used throughout this thesis is the APTOS 2019 Blindness Detection collection [1], hosted on Kaggle by the Asia Pacific Tele-Ophthalmology Society. It comprises 3,662 retinal fundus images, each cropped and resized to 224 × 224 pixels and labelled by an experienced clinician on the five-stage scale described above. The class distribution is heavily skewed toward Stage 0: approximately half of the images carry the No DR label, while the more clinically urgent Stages 3 and 4 together account for less than 14%. Any successful classifier on this dataset must therefore handle severe class imbalance, either through class-weighted loss, oversampling, or focal-style loss functions.

For the binary version of the task, Stages 1 to 4 are collapsed into a single "DR" class, yielding a roughly 50/50 split. This formulation is convenient for benchmarking but is informationally lossy in the clinical sense; for that reason, this thesis evaluates both the binary collapse and the original 5-class severity scale in Kreu IV.

### 2.1.3 Epidemiology and the Screening Capacity Gap

Diabetic retinopathy is the leading cause of preventable blindness among working-age adults globally. According to the International Diabetes Federation, more than 537 million adults were living with diabetes in 2021, a figure projected to rise to 783 million by 2045 [4], and roughly one in three of those patients develops some form of retinopathy during their lifetime. Of these, an estimated 100 million live with vision-threatening DR — either macular oedema or proliferative disease — at any given time [5]. The disease is largely asymptomatic in its early stages, which means that a patient with treatable Mild or Moderate non-proliferative DR is unlikely to present spontaneously to an ophthalmologist; the only reliable mechanism for early detection is systematic retinal screening.

The clinical guideline in most countries is annual or biennial fundus examination for every diabetic patient, but the workforce required to deliver this at scale is simply not available. Sub-Saharan Africa, for instance, has roughly one ophthalmologist per million population, against a recommended ratio at least an order of magnitude higher. Even in high-income settings, large rural areas — including parts of Albania and the wider Western Balkans — face a comparable shortfall, because trained retinal specialists tend to concentrate in major urban hospitals. Tele-ophthalmology programmes that send fundus images electronically from primary-care clinics to regional reading centres are one well-established response to this constraint, but they still require a human grader at the centre of the workflow.

Automated screening is therefore not an academic curiosity but a structural answer to a workforce capacity gap. A system that can pre-screen images and confidently auto-resolve the obvious-No-DR cases — which constitute roughly half of every screening cohort — frees specialist time for the genuinely ambiguous and severe cases. The system developed in this thesis is explicitly designed for that triage role: a calibrated, conformal-set output that quantifies its own confidence and defers cleanly to the clinician when uncertain.

### 2.1.4 Existing Automated DR Systems

Three lines of work form the recent state of the art in automated DR screening. The first is the deep-learning baseline established by Gulshan et al. at Google, who in 2016 published the first JAMA paper showing that a CNN trained on ~128,000 fundus images could match ophthalmologist-level performance on referable DR detection [3]. That study triggered a wave of related work: Ting et al. [6] extended the approach to a multi-ethnic Singapore cohort and demonstrated transfer to other diabetic eye complications, and Sayres et al. [7] showed that augmenting human readers with a deep-learning second opinion improved both speed and consistency. None of these systems, however, exposed an uncertainty quantification layer to their downstream users.

The second line of work is the regulatory-grade autonomous-AI system. The first FDA-cleared autonomous DR screening device, IDx-DR (now LumineticsCore), was approved in 2018 on the strength of a 900-patient pivotal trial [8]: it returns one of two outputs — "more than mild DR detected, refer" or "negative for more than mild DR, retest in 12 months" — and explicitly refuses to grade ambiguous images. The refusal mechanism is precisely the kind of "I don't know" output that this thesis argues for, although IDx-DR's implementation is not publicly disclosed. EyeArt (Eyenuk) followed a similar regulatory path and has been deployed across a number of European screening programmes [9]. These commercial systems are closed-source and provide little methodological transparency.

The third line is research-stage work that explicitly combines uncertainty quantification with DR grading. Leibig et al. [10] demonstrated that MC Dropout improves the precision of automated screening on EyePACS images by deferring high-uncertainty cases. Mukhoti and Gal [11] argued that calibrated probabilities are necessary for selective prediction. More recently, Zhou et al. [12] released RETFound, a foundation model pretrained self-supervised on more than 1.6 million unlabelled retinal images, which substantially improves downstream DR grading when fine-tuned. The present thesis sits in this third line: it combines deep architectures, calibration, conformal sets, MCD, and OOD detection on a publicly available benchmark with a fully reproducible pipeline.

## 2.2 Convolutional Architectures

The thesis evaluates six deep architectures and three classical classifiers. The deep architectures are summarised here; the classical methods are introduced in Section 3.5 of the methodology chapter.

### 2.2.1 ResNet50

ResNet50 [13] introduced residual learning with skip connections that bypass two or three convolutional layers. The architecture has 50 layers organised into five stages, with approximately 25.6 million parameters. Skip connections mitigate the vanishing-gradient problem and enable training of much deeper networks than the previous state of the art. In the present thesis, ResNet50 is used as a transfer-learning backbone with ImageNet pretrained weights and a 128-unit dense head.

### 2.2.2 DenseNet121

DenseNet121 [14] takes the idea of skip connections to an extreme: every layer in a dense block receives the concatenated outputs of all preceding layers. The 121-layer variant has only ~7 million parameters thanks to the parameter-sharing afforded by feature reuse. Dense connectivity also produces an implicit deep-supervision effect that encourages feature reuse and shortens gradient paths.

### 2.2.3 Xception

Xception [15] replaces standard convolutions with depthwise separable convolutions that factorise spatial and channel-wise convolution. The architecture has 36 layers organised into Entry, Middle, and Exit flows, with ~21 million parameters. Despite its extreme decomposition of the standard convolution, Xception consistently performs at or above the level of much larger architectures on ImageNet.

### 2.2.4 VGG16

VGG16 [2] is a simpler, deeper architecture using only 3×3 convolutions and 2×2 max-pooling. Its uniformity makes it a popular pedagogical and transfer-learning choice. In this thesis, VGG16 is evaluated as a transfer-learning backbone with ImageNet pretrained weights and a 128-unit dense head, the same head design used for the other transfer backbones described above.

### 2.2.5 The 3-block Custom CNN

To anchor the comparison, the thesis also includes a from-scratch CNN with three convolution-pool blocks (32 → 64 → 128 filters) followed by a 128-unit dense layer and a sigmoid output. The model has ~11 million parameters and is trained on grayscale-replicated input. A second variant (CNN Tanh+ReLU) substitutes tanh for ReLU in the second convolutional block and the dense layer. The statistical analysis presented in Kreu IV shows that the all-ReLU variant clearly outperforms the mixed-activation variant on this task.

## 2.3 Calibration of Deep Classifiers

A classifier is said to be well-calibrated if its predicted probabilities match observed frequencies: among the inputs for which the model outputs probability 0.8, approximately 80% should be correctly classified. Modern deep networks are typically not well-calibrated; Guo et al. [16] showed that even high-accuracy networks tend to be over-confident, with the gap widening as architectures grow deeper and use stronger regularisation.

### 2.3.1 Expected and Maximum Calibration Error

The standard scalar measure of calibration is the Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where the predictions are partitioned into M equal-width confidence bins B_m, acc(B_m) is the empirical accuracy in bin m, and conf(B_m) is the average confidence in the bin. ECE = 0 indicates perfect calibration; in practice, values below 5% are considered acceptable, while values above 10% indicate clinically problematic over-confidence.

The Maximum Calibration Error (MCE) replaces the weighted sum with a maximum over bins; it captures the worst-case calibration gap and is sensitive to small badly-calibrated regions of the confidence range that ECE would average out [17].

### 2.3.2 Reliability Diagrams

Reliability diagrams plot per-bin accuracy against per-bin mean confidence. The diagonal corresponds to perfect calibration; bars below the diagonal indicate over-confidence and bars above indicate under-confidence. A useful augmentation is to mark the deviation from the diagonal with a coloured stem, scaled by bin size, making it easy to spot regions where the model is most miscalibrated.

### 2.3.3 Temperature Scaling

Among the families of post-hoc calibration methods — including Platt scaling [18] and isotonic regression [19] — temperature scaling [16] is a one-parameter post-hoc fix in which the pre-sigmoid logit z is divided by a temperature T before applying the sigmoid:

$$\hat{p} = \sigma(z / T)$$

Values of T > 1 soften the predictions (reducing peaked confidence), while T < 1 sharpens them. The temperature is fitted by minimising negative log-likelihood on a held-out validation set. Crucially, T does not change the argmax prediction, so accuracy is preserved exactly; only the confidence is rescaled.

## 2.4 Bayesian Deep Learning and Monte Carlo Dropout

The Bayesian view of a neural network treats the weights as random variables with a posterior distribution over the training data. Predicting on a new input then requires marginalising over the posterior, which is intractable in closed form. A range of approximations have been proposed; the most computationally accessible is Monte Carlo Dropout (MCD), introduced by Gal and Ghahramani [20].

### 2.4.1 MC Dropout as Variational Inference

Gal and Ghahramani [20] showed that a network trained with dropout regularisation can be interpreted as performing variational inference under a particular approximating family. At inference time, dropout is left active and the network is run T times for the same input, producing a distribution of predictions. The mean of the T samples is the point estimate, while the variance is an approximation to the model's epistemic (i.e., reducible-with-more-data) uncertainty.

### 2.4.2 Decomposition of Uncertainty

For binary classification with sigmoid output p, the predictive entropy of the mean prediction H(p̄) measures total uncertainty, while the mean per-sample entropy of the dropout samples E_t[H(p_t)] approximates the aleatoric (i.e., data-inherent) component, following the decomposition of Kendall and Gal [21]. Their difference is the mutual information between the prediction and the model parameters, sometimes called the BALD score after the Bayesian Active Learning by Disagreement framework. High mutual information means the dropout samples agree on the input being hard to classify, but disagree on which class it belongs to — the canonical signature of epistemic uncertainty.

### 2.4.3 Deep Ensembles

A complementary, even simpler approach is to train several independent copies of the same architecture with different random initialisations [22]. The ensemble's mean prediction is typically more accurate and better-calibrated than any single member. The disagreement among members provides an uncertainty signal directly analogous to the MC Dropout variance.

The present thesis uses both MCD and a heterogeneous ensemble across the six binary architectures. The latter is cheaper than a deep ensemble of identical members because the diverse architectures already cover a range of inductive biases that are otherwise expensive to obtain through retraining.

## 2.5 Conformal Prediction

Conformal prediction is a framework, originally due to Vovk et al. [23] and recently popularised by Angelopoulos and Bates [24], for producing prediction sets with finite-sample, distribution-free coverage guarantees. Given a target miscoverage rate α, conformal prediction produces, for each test input x, a set C(x) such that

$$\Pr\left( y \in C(x) \right) \geq 1 - \alpha$$

under the assumption that the calibration and test data are exchangeable. Crucially, the guarantee holds no matter what underlying classifier is wrapped — even an uncalibrated, badly-trained model yields a statistically valid prediction set after conformal wrapping. The guarantee is a marginal one (averaged over all test inputs), although stratified variants exist.

### 2.5.1 Split (Inductive) Conformal Prediction

The version used in this thesis is split conformal prediction:

1. Hold out a calibration set {(x_i, y_i)} disjoint from the training set.
2. For each calibration sample, compute a non-conformity score s_i that quantifies how surprising the true label is under the trained classifier.
3. Compute the empirical (1 − α) quantile q̂ of the scores, with a finite-sample correction ⌈(n+1)(1 − α)⌉ / n.
4. For a test input x, include in C(x) every candidate label whose non-conformity score is ≤ q̂.

### 2.5.2 Score Functions

Two non-conformity scores are evaluated in this thesis:

- **Least Ambiguous Classifier (LAC)**: s(x, y) = 1 − p̂_y(x). Simple and well-studied; gives marginal coverage.
- **Adaptive Prediction Sets (APS)** [25]: sort the per-class probabilities in descending order; the score for the true class is the cumulative probability up to and including it, with random tie-breaking. APS produces smaller average set sizes and tends to be more clinically useful because it can produce two-class "abstain" sets ({No DR, Mild}) for ambiguous inputs.

### 2.5.3 Set-Size Distribution and Conditional Coverage

Beyond the marginal coverage guarantee, two diagnostics are reported throughout the thesis: the mean set size, which captures how often the wrap is uncertain, and the per-class conditional coverage, which checks that the guarantee is honoured uniformly across true classes rather than concentrated on the easy majority class.

## 2.6 Out-of-Distribution Detection

A clinically deployed system will inevitably receive inputs outside its training distribution: chest X-rays mistakenly uploaded to a retina classifier, fundus images from a different camera with different colour calibration, or simply low-quality images. The model should refuse to predict on such inputs rather than confidently output a wrong answer.

### 2.6.1 Output-Space Methods

The simplest baseline is the Maximum Softmax Probability (MSP) [26]: OOD inputs tend to produce lower confidence than ID inputs, so max_k p̂_k can be thresholded. Despite its simplicity, MSP suffers when softmax saturates — a network can be forced into very high confidence even on noise.

The Energy score [27] sidesteps softmax saturation by computing the negative log-sum-exp of the logits. ID inputs produce lower energy than OOD inputs, and the score is unbounded above, which avoids the ceiling effect of MSP.

### 2.6.2 Feature-Space Methods

A more powerful family of methods compares the position of the test input in feature space to the distribution of training features. Mahalanobis distance [28] fits per-class Gaussian densities to the training features (with a shared covariance for stability) and scores test inputs by the negative log-likelihood under the closest class-conditional Gaussian. Cosine distance to the in-distribution centroid is a simpler variant. For inputs far from the training manifold (e.g., uniform noise), feature-space methods dramatically outperform output-space methods, often achieving perfect AUROC on synthetic OOD.

## 2.7 Related Work in DR with Uncertainty

A growing body of work applies uncertainty estimation to DR detection and to medical imaging more broadly. The earliest work in the DR-specific literature is that of Leibig et al. [10], who applied MC Dropout to a CNN trained on EyePACS images and showed that high predictive uncertainty correlates strongly with classification error: deferring the most uncertain ~20% of images doubled the precision of the auto-screened pool. Filos et al. [29] extended this analysis with a comparison of multiple uncertainty estimators (MCD, deep ensembles, deterministic) on the same DR cohort and found that the choice of uncertainty signal matters less than whether one is computed at all.

Laves et al. [30] turned the uncertainty question to retinal-layer-thickness regression and demonstrated that well-calibrated regression uncertainty produces useful selective inference on optical coherence tomography. Gulshan et al. [3] published the landmark JAMA paper showing that a deep CNN could match ophthalmologist-level performance on referable DR detection, but did not expose any uncertainty layer; subsequent regulatory-grade systems (IDx-DR [8], EyeArt [9]) added an internal "ungradeable" flag without disclosing the underlying mechanism. Sayres et al. [7] showed that pairing a deep classifier with an explicit confidence bar improved human-machine agreement on referable DR by reducing over-reliance on confident-but-wrong predictions.

The conformal-prediction literature has only recently turned to medical imaging. Lu et al. [31] applied split conformal prediction to chest X-ray classification and showed that the marginal-coverage guarantee transfers cleanly. Vazquez and Facelli [32] surveyed conformal prediction in medical applications and noted that adoption has been slowest in screening contexts because clinicians are unfamiliar with set-valued predictions; the present thesis addresses this by integrating the conformal output into a clear "auto-classify / refer" rule rather than presenting the set itself as the user-facing artefact.

A parallel literature on out-of-distribution detection in medical imaging has examined inputs as far afield as MRIs labelled as the wrong organ [33] and adversarial perturbations of dermatology images [34]. The consensus is that simple feature-space distance metrics often outperform output-space scores, particularly when the OOD inputs are visually distinct from the training distribution — a finding consistent with the synthetic-noise OOD result reported in Kreu IV.

Foundation-model approaches to retinal imaging have appeared in the last two years. Zhou et al. [12] released RETFound, a vision transformer pretrained self-supervised on 1.6 million unlabelled retinal images, which sets a new ceiling for downstream DR grading when fine-tuned. Caron et al. [35] introduced the DINO self-supervised framework that underpins much of this work. While the present thesis does not use these foundation models — both because of the CPU-only training budget and because the comparison would shift attention away from the uncertainty methods that are this thesis' core contribution — the uncertainty machinery developed here is architecture-agnostic and would transfer directly to any future foundation-model backbone.

Despite this rich landscape, very few studies have applied conformal prediction to DR specifically, and even fewer have done so on the APTOS 2019 benchmark. The closest precursor is Mukhoti and Gal [11], who applied calibration analysis to a different ophthalmic task and argued that calibrated probabilities are a precondition for selective prediction in clinical settings. The present thesis closes the conformal-prediction-on-APTOS gap by combining (i) a thorough calibration analysis with reliability diagrams and post-hoc temperature scaling for six architectures, (ii) two independent uncertainty mechanisms — MCD and heterogeneous ensemble disagreement — applied to the same fixed test set, (iii) split conformal prediction with both LAC and APS scores and formal coverage at α = 0.05 and α = 0.10, and (iv) feature-space OOD detection with Mahalanobis and cosine baselines.

## 2.8 Regulatory Landscape for Medical AI

The methodological choices in this thesis are not merely an academic preference for rigour: they are increasingly demanded by the regulatory frameworks under which a clinical screening tool would have to operate. Three frameworks are particularly relevant.

**U.S. Food and Drug Administration (FDA) — Software as a Medical Device (SaMD).** The FDA's evolving framework for SaMD [36] places automated DR screening tools squarely within the highest-risk Class III category when used as standalone diagnostic devices. The 2021 FDA Action Plan for AI/ML-based SaMD calls explicitly for "robustness assessment" and "real-world performance monitoring", both of which are easier to satisfy with a calibrated, conformal output than with a single argmax verdict. The FDA-cleared IDx-DR system [8] reportedly relies on an internal confidence threshold to refuse ambiguous cases, but the underlying mechanism is not publicly disclosed.

**European Union AI Act.** The EU AI Act, which entered into force in 2026, classifies medical AI systems as "high-risk" and imposes a set of mandatory requirements that map directly onto the methods used in this thesis: documented robustness testing (addressed by the K-fold CV and McNemar analysis), risk management through transparent uncertainty (addressed by calibration, conformal, and MCD), human oversight (addressed by the explicit refer-to-clinician rule), and post-market monitoring (enabled by the reliability-diagram and ECE infrastructure that can be re-run on production data). A model that emits only a point prediction would not satisfy these requirements.

**World Health Organisation (WHO) — Ethics and Governance of AI for Health.** The WHO guidance [37] adds equity considerations: a model trained predominantly on one population (in the case of APTOS 2019, primarily Indian) cannot be deployed elsewhere without explicit demographic-shift evaluation. This thesis acknowledges the limitation directly in Section 5.3, and the cross-dataset infrastructure in `master/run_cross_dataset.py` is the concrete first step towards addressing it.

The combined effect of these frameworks is that "clinical decision support" is no longer a label that can be applied to any classifier with high accuracy: it is a regulatory category with specific requirements on calibration, robustness, and uncertainty disclosure. The methodology of this thesis is designed to satisfy those requirements from the outset.

The methodological choices and their justification are described in detail in the next chapter.
