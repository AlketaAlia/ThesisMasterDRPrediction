# ABSTRAKTI

**KLASIFIKIM I RETINOPATISË DIABETIKE ME VETËDIJE NDAJ PASIGURISË DUKE PËRDORUR PARASHIKIMIN KONFORMAL DHE MËSIMIN E THELLË BAYESIAN**

Kjo tezë zhvillon një sistem të vetëdijshëm ndaj pasigurisë për klasifikimin e automatizuar të retinopatisë diabetike (DR) nga imazhet e fundus-it. Ndërkohë që rrjetat moderne konvolucionale arrijnë saktësi mbi 95% në datasetin Kaggle APTOS 2019, përdorimi klinik mbetet i kufizuar nga prirja e tyre për të prodhuar parashikime të gabuara me besueshmëri të lartë. Puna argumenton se një mjet screening-u duhet të bëjë më shumë sesa të maksimizojë saktësinë: duhet edhe të dijë e ta komunikojë kur është i pasigurt.

Metodologjia ndërton një protokoll vlerësimi statistikisht rigoroz mbi datasetin me 3,662 imazhe, me ndarje të stratifikuar 70/15/15, parapërpunim specifik për arkitekturë, peshim të balancuar të klasave, dhe një pipeline të unifikuar. Gjashtë arkitektura të thella vlerësohen si klasifikatorë binarë — ResNet50, Xception, DenseNet121, VGG16, një CNN nga zero, dhe një variant CNN tanh+ReLU — së bashku me tre klasifikatorë klasikë. Pas korrigjimit të preprocessing-ut, pesë modelet më të forta grupohen midis 95.27% dhe 96.00%, dhe testet McNemar tregojnë se janë statistikisht të padallueshme. Një ensemble heterogjen arrin 96.55%.

Përtej saktësisë, teza prezanton analizë kalibrimi (ECE, MCE, temperature scaling), Monte Carlo Dropout me T = 30 forward pass, dhe parashikim split conformal me garanci formale për coverage. Një 5-fold cross-validation i ResNet50 jep vlerësim prej 95.64 ± 0.18 pikë përqindje, dhe detektimi i imazheve jashtë-shpërndarjes me metoda në hapësirën e features arrin AUROC = 1.0.

Teza pastaj rireformulon detyrën në shkallën origjinale me 5 nivele. Një ResNet50 multi-class arrin 77.09% saktësi dhe quadratic-weighted kappa 0.847, konkurrues me leaderboard-in publik APTOS 2019. Conformal prediction në α = 0.10 prodhon coverage 89.45% me një politikë klinike të qartë: 71% e pacientëve auto-klasifikohen, 29% referohen tek mjeku.

Efekti kumulativ është një tranzicion nga një numër i vetëm saktësie tek një pipeline transparent për mbështetje vendimesh klinike.

**Fjalë kyçe:** retinopatia diabetike, mësimi i thellë, parashikim konformal, Monte Carlo Dropout, kalibrim, detektim jashtë-shpërndarjes

---

# ABSTRACT

**UNCERTAINTY-AWARE DIABETIC RETINOPATHY GRADING USING CONFORMAL PREDICTION AND BAYESIAN DEEP LEARNING**

This thesis develops an uncertainty-aware system for the automated grading of diabetic retinopathy (DR) from retinal fundus images. While modern convolutional neural networks routinely achieve accuracies above 95% on the Kaggle APTOS 2019 Blindness Detection benchmark, deployment in clinical screening remains hindered by their tendency to produce confidently incorrect predictions. This work argues that a screening tool intended for non-specialist clinicians must do more than maximise accuracy: it must also know, and communicate, when its predictions are unreliable.

The methodology re-establishes a statistically rigorous evaluation protocol over the 3,662-image APTOS 2019 dataset, with stratified 70/15/15 train/validation/test splits, architecture-specific input pre-processing, class-balanced loss weighting, and a unified training pipeline. Six deep architectures are first evaluated as binary classifiers — ResNet50, Xception, DenseNet121, VGG16, a from-scratch convolutional neural network, and a CNN variant combining tanh and ReLU activations — together with three classical classifiers (Decision Tree, Random Forest, SVM) trained on DenseNet121-extracted features. Once preprocessing and class weighting are corrected, the five strongest deep models cluster between 95.27% and 96.00% test accuracy, and pairwise McNemar tests show that they are statistically indistinguishable. A heterogeneous ensemble of all six members achieves 96.55%.

Beyond accuracy, the thesis introduces calibration analysis (ECE, MCE, reliability diagrams, post-hoc temperature scaling), Monte Carlo Dropout with T = 30 stochastic forward passes, and split conformal prediction with finite-sample, distribution-free coverage guarantees. A 5-fold cross-validation of ResNet50 yields 95.64 ± 0.18 percentage points. Out-of-distribution detection using feature-space methods attains AUROC = 1.0.

The thesis then re-formulates the task to the clinically meaningful 5-stage severity scale (No DR, Mild, Moderate, Severe, PDR). A multi-class ResNet50 reaches 77.09% accuracy and a quadratic-weighted kappa of 0.847, competitive with public APTOS 2019 leaderboards. Conformal prediction at α = 0.10 produces an empirical coverage of 89.45% and a clinical policy of 71% auto-classify / 29% refer-to-clinician.

The cumulative effect is a transition from a single accuracy number to a transparent, statistically grounded clinical decision-support pipeline.

**Keywords:** diabetic retinopathy, deep learning, conformal prediction, Monte Carlo Dropout, calibration, out-of-distribution detection, multi-class severity grading
