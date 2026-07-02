# KREU III

# METODOLOGJIA

Ky kapitull përshkruan protokollin eksperimental të unifikuar të përdorur përgjatë tri fazave të tezës. Seksioni 3.1 përshkruan dataset-in dhe ndarjet. Seksioni 3.2 mbulon preprocessing-un dhe augmentimin. Seksioni 3.3 përshkruan pipeline-in e trajnimit të përbashkët nga të gjitha arkitekturat. Seksioni 3.4 liston metrikat e vlerësimit. Seksioni 3.5 prezanton tre klasifikatorët klasikë të baseline. Seksionet 3.6 deri 3.9 përshkruajnë protokollet eksperimentale për fazat e mëtejshme (parashikim konformal, MC Dropout, OOD, K-fold) — për Kontrollin e Dytë, këto janë në fazë implementimi. Seksioni 3.10 detajon zgjerimin shumëklasor (i planifikuar për Fazën 3). Seksioni 3.11 përshkruan implementimin, organizimin e kodit, dhe reproduktueshmërinë.

> *Shënim për Kontrollin e Dytë: seksionet 3.1–3.5 dhe 3.11 janë stabilizuar plotësisht. Seksionet 3.6–3.10 përshkruajnë protokollet që janë në implementim aktiv.*

## 3.1 Dataset-i dhe ndarjet

### 3.1.1 APTOS 2019

Të gjitha eksperimentet përdorin dataset-in APTOS 2019 Blindness Detection, që përbëhet nga 3,662 imazhe fundus retinal me rezolucion 224×224 piksel, secila e etiketuar nga një klinicist në shkallën 0–4 të ashpërsisë. Dataset-i ngarkohet një herë në fillim të çdo eksperimenti dhe cache-ohet si një DataFrame i vetëm pandas në disk, për të garantuar që çdo eksperiment shikon të njëjtin rend imazhesh dhe vlera identike piksel-pas-pikseli. Shpërndarja e klasave është rëndë e zhvendosur, dominuar nga klasa Nuk ka DR.

### 3.1.2 Ndarja e stratifikuar 70/15/15

Përdoret një ndarje e stratifikuar 70/15/15 train/validation/test me `random_state = 123`:

- **Train (2,562 imazhe)**: përdoret për përditësimet e parametrave.
- **Validation (550 imazhe)**: përdoret për early stopping, planifikimin e learning rate-it, përshtatjen e temperature-scaling, dhe përshtatjen e konformal threshold. *Asnjëherë* nuk përdoret për vlerësimin përfundimtar.
- **Test (550 imazhe)**: i mbajtur i fiksuar përgjatë çdo eksperimenti në këtë tezë. Çdo metrikë test e raportuar llogaritet mbi të njëjtat 550 imazhe.

Stratifikimi ruan proporcionet e klasave përgjatë ndarjeve, gjë që ka rëndësi sepse dataset-i është i çekuilibruar. Pa stratifikim, validimi ose test set-i mund të nën- ose mbi-përfaqësonin klasat minoritare të ashpërsisë rastësisht, duke e bërë vlerësimin e test-it të pa besueshëm.

### 3.1.3 Pse ka rëndësi një test set i veçantë

Është metodologjikisht e rëndësishme të ndahen të dhënat e përdorura për përzgjedhjen e modelit nga të dhënat e përdorura për vlerësimin përfundimtar. Ri-përdorimi i set-it të mbajtur për përzgjedhjen e early stopping dhe për raportimin e saktësisë "validimit" është një formë e butë e data leakage: set-i i mbajtur përdoret implicit për përzgjedhjen e modelit nëpërmjet zgjedhjes së epokës early-stopping. Pipeline-i i përdorur këtu ndan pastër shqetësimet: validimi nxit të gjitha vendimet e përzgjedhjes, dhe test set-i kyçet derisa në fund të trajnimit. Test set-i përdoret gjithashtu i pandryshuar përgjatë të gjitha fazave.

## 3.2 Preprocessing i imazheve

### 3.2.1 Preprocessing arkitekturë-specifik

Një konsideratë metodologjike delikate por e rëndësishme është përdorimi i preprocessing-ut input arkitekturë-specifik. Çdo backbone ImageNet-pretrained në Keras ka funksionin e tij `preprocess_input`:

- **ResNet50**: BGR channel order, mean subtraction me `[103.939, 116.779, 123.68]` (ImageNet means në BGR).
- **DenseNet121**: i njëjti si ResNet50 (stil caffe).
- **Xception**: shkallëzim i thjeshtë në `[-1, 1]`.
- **VGG16**: BGR mean subtraction (stil caffe).
- **CNN (nga zero)**: konvertim grayscale + pjesëtim me 255.

Një rishkallëzim uniform `1/255` i aplikuar në çdo arkitekturë është një mospërputhje serioze për backbone-et ImageNet, peshat e parainstaluara të të cilave presin input të mean-subtracted; përdorimi i `1/255` në vend mund të ulë saktësinë test me 10 pikë përqindjeje ose më shumë. Çdo backbone ImageNet në këtë tezë merr kështu funksionin e tij kanonik preprocessing.

### 3.2.2 Augmentimi i të dhënave

Gjeneratori i të dhënave të trajnimit aplikon një set të moderuar augmentimesh gjeometrike për të luftuar overfitting-un:

- Rangu i rrotullimit: ±20°
- Zhvendosje gjerësie dhe lartësie: ±10%
- Rangu shear: 10%
- Rangu zoom: 10%
- Flip horizontal: i aktivizuar

Këto aplikohen **vetëm në set-in e trajnimit**. Gjeneratorët e validimit dhe test-it aplikojnë të njëjtin preprocessing arkitekturë-specifik por pa augmentim, kështu që të gjitha metrikat llogariten në imazhe të papërpunuara.

## 3.3 Pipeline i trajnimit

### 3.3.1 Komponentët e përbashkët

Çdo model në këtë tezë trajnohet me të njëjtin pipeline bërthamë:

- **Optimizer**: Adam me learning rate fillestar $1 \times 10^{-3}$.
- **Loss**: binary cross-entropy për klasifikuesit binarë; categorical cross-entropy për modelet 5-klasore.
- **Batch size**: 32.
- **Class weights**: të llogaritura me `sklearn.utils.class_weight.compute_class_weight(class_weight='balanced')` dhe të kaluara në `model.fit(class_weight=...)`. Kjo është kritike për problemin rëndë të çekuilibruar 5-klasor.
- **Max epokat**: 100, me early stopping i aktivizuar nëse humbja e validimit nuk përmirësohet për 10 epoka të njëpasnjëshme.

### 3.3.2 Callbacks

Tre callbacks janë standarde:

1. **EarlyStopping** (`monitor='val_loss', patience=10, restore_best_weights=True`): ndalon trajnimin sapo humbja e validimit plateaus dhe rikthen peshat më të mira.
2. **ReduceLROnPlateau** (`factor=0.5, patience=5, min_lr=1e-6`): përgjysmon learning rate-in nëse humbja e validimit stagnohet për 5 epoka.
3. **ModelCheckpoint** (`monitor='val_accuracy', save_best_only=True`): ruan modelin më të mirë të validimit në disk.

### 3.3.3 Kokat e arkitekturave

Për modelet transfer-learning, baza ImageNet ngarkohet me `include_top=False`, ngrihet, dhe topohet me një shtresë Global Average Pooling, një shtresë Dense me 128 njësi me aktivizim ReLU, dhe një shtresë përfundimtare klasifikimi (sigmoid për binary, softmax për multi-class). CNN-ja nga zero përdor tre blloqe konvolucion-pooling (32 → 64 → 128 filtra me kernels 3×3 dhe max-pooling 2×2) të ndjekur nga një shtresë Flatten, një Dense ReLU me 128 njësi, dhe shtresa e klasifikimit.

## 3.4 Metrikat e vlerësimit

Metrika të shumta janë llogaritur për çdo model, të organizuara si vijon.

### 3.4.1 Metrikat e diskriminimit

- **Saktësia**: pjesë e parashikimeve të sakta në test set.
- **AUROC**: për detyrat binare, mat diskriminimin në të gjitha threshold-et.
- **Macro dhe weighted F1**: për multi-class, kapin balancimin per-class precision-recall.

### 3.4.2 Metrikat e kalibrimit

- **Expected Calibration Error (ECE)** me 15 shporta të barabarta të konfidencës.
- **Maximum Calibration Error (MCE)**: hendeku më i keq i kalibrimit në nivel shporte.
- **Reliability diagram**: saktësia per-shportë kundër konfidencës.

### 3.4.3 Metrikat ordinale (vetëm multi-class)

Për gradimin 5-fazor, dy metrika shtesë kapin strukturën ordinale të etiketave:

- **Quadratic-Weighted Kappa (QWK)**: penalizon gabimet sipas distancës së tyre të katrore në shkallën e ashpërsisë.
- **Mean Ordinal Distance**: $\mathbb{E}[|y - \hat y|]$.

### 3.4.4 Inferenca statistikore

- **Bootstrap 95% confidence intervals** mbi saktësinë test, të llogaritura me 1,000 resamples dhe seed 42.
- **Pairwise McNemar tests** midis çdo çifti modeli, me një korrigjim vazhdimi.
- **Cohen's kappa** midis çifteve të klasifikatorëve, si masë marrëveshje e korrigjuar nga rastësia.

## 3.5 Klasifikuesit klasikë baseline

Për të ankoruar krahasimin e arkitekturave të thella ndaj metodave më të thjeshta, tre klasifikues klasikë të mësimit të makinës janë vlerësuar gjithashtu: një Decision Tree, një Random Forest, dhe një Support Vector Machine. Në vend që t'i trajnojmë në 224×224×3 piksele raw — që do prodhonte vektorë veçorish 150,528-dimensionalë — modelet klasike trajnohen mbi **veçoritë e thella të nxjerra nga një backbone DenseNet121 i ngrirë me peshat ImageNet pretrained**. Rrjeti bazë ngarkohet me `include_top=False, pooling='avg'`, duke prodhuar një vektor veçorish 1024-dimensional për çdo imazh input.

Tre klasifikuesit, të gjithë nga `scikit-learn`:

- **Decision Tree**: kriter i parazgjedhur i ndarjes Gini, `class_weight='balanced'`, `random_state=123`.
- **Random Forest**: 300 pemë, `class_weight='balanced'`, `random_state=123`.
- **Support Vector Machine**: RBF kernel, `probability=True`, `class_weight='balanced'`.

## 3.6 Protokolli i parashikimit konformal *(në implementim)*

Set-i i validimit (550 imazhe) do të përdoret si calibration set konformal. Të dy score-t LAC dhe APS do të vlerësohen. Threshold-i i moskonformitetit $\hat q$ do të llogaritet si kuantili empirik $\lceil (n+1)(1-\alpha) \rceil / n$, duke dhënë korrigjimin finite-sample të nevojshëm për garancinë e coverage marginal.

Dy nivele miscoverage do të vlerësohen: $\alpha = 0.10$ (90% coverage target) dhe $\alpha = 0.05$ (95% coverage target).

## 3.7 Protokolli Monte Carlo Dropout *(në implementim)*

Dy variante MC-Dropout do të prezantohen për Fazën 2:

- **`cnn_mcd`**: CNN-ja 3-bllokore nga zero me `SpatialDropout2D(rate=0.3)` pas çdo çifti konvolucion-pooling.
- **`resnet50_mcd`**: ResNet50 transfer learning me `Dropout(rate=0.3)` para shtresës dense përfundimtare.

Në kohën e inferencës, modeli do thirret me `training=True` për të mbajtur dropout-in aktiv. T = 30 forward passes stokastike do të kryhen për çdo input test.

## 3.8 Protokolli i detektimit jashtë-shpërndarjes *(në implementim)*

Për Fazën 2, input-et ID do të jenë 550 imazhet fundus të APTOS test set. Input-et OOD do të jenë 300 imazhe sintetike të zhurmës uniforme. DenseNet121 me peshat ImageNet do prodhojë veçori 1024-dimensionale për input-et e të dy llojeve. Set-i i validimit do të përdoret për të përshtatur means për-klasë dhe një kovariancë të përbashkët për Mahalanobis.

Katër score do vlerësohen: MSP, Energy, Mahalanobis, dhe Cosine.

## 3.9 K-Fold Cross-Validation *(planifikuar)*

Faza 2 gjithashtu do të kryejë një 5-fold cross-validation të ResNet50. Pool-i train+validation (3,112 imazhe) do të ndahet në 5 fold të stratifikuara; test set-i (550 imazhe) do mbahet i fiksuar përgjatë fold-eve. Metrikat përfundimtare të test-it do të jenë mean dhe deviacioni standard përgjatë fold-eve.

## 3.10 Zgjerimi shumëklasor *(Faza 3 — planifikuar)*

Faza 3 do të riformulojë detyrën duke përdorur etiketat origjinale 5-klasore të ashpërsisë (0 deri 4) në vend të kolapsit binar. E njëjta ndarje 70/15/15 do të përdoret, por e stratifikuar tani mbi etiketën 5-klasore.

Dy arkitektura do të vlerësohen në cilësimin multi-class: `cnn_5class` dhe `resnet50_5class`.

## 3.11 Implementimi dhe reproduktueshmëria

### 3.11.1 Mjedisi softuerik

- Python 3.9.6
- TensorFlow 2.15.0, Keras 2.15
- scikit-learn 1.4.2, pandas 2.2.2, numpy 1.26.4
- streamlit 1.33.0

Mjedisi i plotë i fiksuar është kapur në `requirements.txt`.

### 3.11.2 Hardware

Të gjitha eksperimentet ekzekutohen në një MacBook Air Apple Silicon (M-series) me 8 GB RAM, vetëm CPU. Trajnimi i një klasifikatori binar transfer-learning merr afërsisht 15 deri 30 minuta; CNN-ja nga zero merr afërsisht 30 deri 60 minuta.

### 3.11.3 Organizimi i kodit

- `lib/`: Streamlit user interface dhe modulet e inferencës.
- `scripts/`: driver-i i unifikuar i trajnimit (`train.py --arch <name>`).
- `master/uncertainty/`: modulet e kalibrimit, konformal, ensemble, MC Dropout, dhe OOD.
- `master/run_*.py`: drivers analiza që prodhuan çdo CSV, JSON, dhe figurë të referuar në këtë tezë.
- `master/results/`: të gjitha output-et eksperimentale.

### 3.11.4 Konvencionet e reproduktueshmërisë

- **Random seeds**: ndarja e stratifikuar përdor `random_state=123`; bootstrap CIs përdorin seed 42.
- **Modelet e ruajtura**: çdo model i trajnuar ruhet si një skedar `.keras` me history pickle korrespondues.

\newpage
