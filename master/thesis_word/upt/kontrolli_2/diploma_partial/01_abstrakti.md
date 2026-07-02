# ABSTRAKTI

**KLASIFIKIM I VETËDIJSHËM NDAJ PASIGURISË I RETINOPATISË DIABETIKE DUKE PËRDORUR PARASHIKIM KONFORMAL DHE MËSIM TË THELLË BAYESIAN**

*Statusi i draft-it: Kontrolli i Dytë — përafërsisht 50% i përfunduar*

Kjo tezë synon të zhvillojë një sistem të vetëdijshëm ndaj pasigurisë për klasifikimin e automatizuar të retinopatisë diabetike (DR) nga imazhet e fundus-it të retinës. Ndërkohë që rrjetat moderne neurale konvolucionale arrijnë rregullisht saktësi mbi 95% në datasetin Kaggle APTOS 2019 Blindness Detection, përdorimi në screening klinik mbetet i kufizuar nga prirja e këtyre modeleve për të prodhuar parashikime të gabuara me besueshmëri të lartë. Kjo punë argumenton se një mjet screening-u i destinuar për përdorim nga klinicistë jo-specialistë duhet të bëjë më shumë sesa të maksimizojë saktësinë: duhet edhe të dijë — dhe ta komunikojë — kur parashikimet e tij janë të pasigurta.

Metodologjia rivendos një protokoll vlerësimi statistikisht rigoroz mbi datasetin APTOS 2019 me 3,662 imazhe, me ndarje të stratifikuar 70/15/15 për train/validation/test, parapërpunim specifik për çdo arkitekturë, peshim të humbjes sipas balancimit të klasave, dhe një pipeline trajnimi të unifikuar. **Faza e parë e eksperimenteve është përfunduar**: gjashtë arkitektura të thella vlerësohen si klasifikatorë binarë — ResNet50, Xception, DenseNet121, VGG16, një CNN i ndërtuar nga zero, dhe një variant CNN që kombinon aktivizimet tanh dhe ReLU. Pasi preprocessing-u dhe class weighting korrigjohen, pesë modelet më të forta të thella grupohen midis **95.27% dhe 96.00% saktësi**, dhe testet McNemar tregojnë se ato janë **statistikisht të padallueshme** nga njëri-tjetri. Një ensemble heterogjen i të gjashtë anëtarëve arrin **96.55%**.

**Fazat e dyta dhe e treta janë në zhvillim**: implementimi i analizës së kalibrimit me temperature scaling, parashikim split conformal me garanci formale për coverage, Monte Carlo Dropout me T = 30 forward pass stokastikë, K-fold cross-validation, dhe detektim i imazheve jashtë-shpërndarjes janë në vazhdim. Faza e tretë — riformulimi i detyrës në shkallën origjinale klinike me 5 nivele — është planifikuar pas përfundimit të Fazës 2.

Efekti i parashikuar kumulativ është një tranzicion nga një numër i vetëm saktësie tek një pipeline transparent dhe statistikisht i themeluar për mbështetje vendimesh klinike, që pritet të finalizohet brenda Shtatorit 2026.

**Fjalë kyçe:** retinopatia diabetike, mësimi i thellë, parashikim konformal, Monte Carlo Dropout, kalibrim, detektim jashtë-shpërndarjes

---

# ABSTRACT

**UNCERTAINTY-AWARE DIABETIC RETINOPATHY GRADING USING CONFORMAL PREDICTION AND BAYESIAN DEEP LEARNING**

*Draft status: Second Control — approximately 50% complete*

This thesis develops an uncertainty-aware system for the automated grading of diabetic retinopathy (DR) from retinal fundus images. While modern convolutional neural networks routinely achieve accuracies above 95% on the Kaggle APTOS 2019 Blindness Detection benchmark, deployment in clinical screening remains hindered by their tendency to produce confidently incorrect predictions. This work argues that a screening tool intended for non-specialist clinicians must do more than maximise accuracy: it must also know, and communicate, when its predictions are unreliable.

The methodology establishes a statistically rigorous evaluation protocol over the 3,662-image APTOS 2019 dataset, with stratified 70/15/15 train/validation/test splits, architecture-specific input pre-processing, class-balanced loss weighting, and a unified training pipeline. **The first experimental phase has been completed**: six deep architectures are evaluated as binary classifiers — ResNet50, Xception, DenseNet121, VGG16, a from-scratch convolutional neural network, and a CNN variant combining tanh and ReLU activations. Once preprocessing and class weighting are corrected, the five strongest deep models cluster between **95.27% and 96.00% test accuracy**, and pairwise McNemar tests show that they are **statistically indistinguishable**. A heterogeneous ensemble of all six members achieves **96.55%**.

**Phases two and three are under active development**: implementation of calibration analysis with temperature scaling, split conformal prediction with finite-sample coverage guarantees, Monte Carlo Dropout with T = 30 stochastic forward passes, K-fold cross-validation, and out-of-distribution detection are in progress. The third phase — reformulating the task to the original 5-stage clinical scale with QWK as the headline metric — is planned for after Phase 2 completion.

The expected cumulative effect is a transition from a single accuracy number to a transparent, statistically grounded clinical decision-support pipeline, with final completion expected by September 2026.

**Keywords:** diabetic retinopathy, deep learning, conformal prediction, Monte Carlo Dropout, calibration, out-of-distribution detection

\newpage
