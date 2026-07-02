# UNIVERSITETI POLITEKNIK I TIRANËS

# FAKULTETI I TEKNOLOGJISË SË INFORMACIONIT

# DEPARTAMENTI I INXHINIERISË INFORMATIKE

---

# RAPORT PROGRESI — KONTROLLI I DYTË I DIPLOMËS

**Titulli i temës:** Klasifikim i Vetëdijshëm ndaj Pasigurisë i Retinopatisë Diabetike duke Përdorur Parashikim Konformal dhe Mësim të Thellë Bayesian

**Kandidate:** Alketa Alia

**Udhëheqës shkencor:** Prof. Dr. [Emri Mbiemri]

**Data e dorëzimit:** [Plotëso datën]

**Statusi i përgjithshëm:** **Përafërsisht 50% i përfunduar**

---

## 1. Përmbledhje e statusit

Punimi i diplomës po zhvillohet sipas planit fillestar të aprovuar gjatë Kontrollit të Parë. Faza eksperimentale e parë (klasifikimi binar) është përfunduar me rezultate konkrete; faza e dytë (mjetet e avancuara të pasigurisë) është në zhvillim aktiv; pjesa teorike (Kreu II) është pothuajse e plotësuar, dhe metodologjia (Kreu III) është draft i parë i stabilizuar.

Tabela më poshtë përmbledh statusin sipas kapitujve dhe komponentëve eksperimentale.

| Komponenti | Status | Përqindje |
|------------|--------|----------:|
| Kreu I — Hyrja | Drafti i parë i përfunduar | 90% |
| Kreu II — Sfondi teorik | Drafti i parë, në rishikim | 80% |
| Kreu III — Metodologjia | Draft pune, po stabilizohet | 70% |
| Kreu IV — Rezultate Faza 1 (binary) | Eksperimentet të mbaruara, tabelat draft | 65% |
| Kreu IV — Rezultate Faza 2 (UQ) | Implementim në vazhdim | 30% |
| Kreu IV — Rezultate Faza 3 (multi-class) | E planifikuar | 0% |
| Kreu V — Përfundimet | E planifikuar | 0% |
| Aplikacioni Streamlit | Prototip funksional | 40% |
| Bibliografia | 20 referenca aktuale, target 40+ | 50% |
| **Totali i përgjithshëm** | — | **~50%** |

---

## 2. Çfarë është realizuar deri tani

### 2.1 Pjesa teorike (Kreu I dhe II)

- Hyrja e diplomës është shkruar e plotë, me përshkrimin e problemit klinik të retinopatisë diabetike, motivimin për përdorimin e pasigurisë në AI mjekësor, dhe parashtrimin e **katër pyetjeve kërkimore** që udhëheqin punën.
- Kreu II — Sfondi teorik — është shkruar në draft të parë, përfshirë:
  - Patofiziologjia dhe shkalla klinike e DR-së (5 faza)
  - Dataset-i APTOS 2019 (3,662 imazhe)
  - Përshkrimi i 6 arkitekturave të vlerësuara
  - Konceptet e kalibrimit, Monte Carlo Dropout, parashikim konformal, dhe OOD detection
  - Korniza rregullatore (FDA SaMD, EU AI Act)

### 2.2 Metodologjia (Kreu III)

- Është krijuar **pipeline-i i unifikuar i trajnimit** (`scripts/train.py`)
- Implementuar **ndarja stratified 70/15/15** me random_state të fiksuar (`scripts/helpers.py`)
- Konfiguruar **preprocessing arkitekturë-specifik** për çdo backbone
- Vendosur **callbacks standard**: EarlyStopping (patience=10), ReduceLROnPlateau, ModelCheckpoint
- Dokumentuar protokollet eksperimentale për kalibrim, conformal, MCD, OOD

### 2.3 Eksperimentet — Faza 1 (klasifikim binar)

E përfunduar plotësisht. Janë trajnuar **6 modele binary** dhe rezultatet preliminare janë:

| Modeli | Test acc | 95% CI |
|--------|---------:|:------:|
| ResNet50 | 95.27% | [93.45, 97.09] |
| Xception | 95.45% | [93.82, 97.09] |
| DenseNet121 | 95.64% | [94.00, 97.27] |
| VGG16 | 95.64% | [94.00, 97.27] |
| CNN (nga zero) | 96.00% | [94.36, 97.45] |
| CNN (Tanh+ReLU) | 92.91% | [90.72, 94.91] |

Gjetjet preliminare:
- Pesë modelet e forta janë statistikisht të padallueshme (McNemar p > 0.5)
- CNN(Tanh+ReLU) është dukshëm më e dobët (p < 0.05)
- Pipeline-i tregon stabilitet të lartë në metrikat e raportuara

### 2.4 Aplikacioni Streamlit — prototip

Prototip funksional është krijuar (`app.py`), që mbështet:
- Ngarkim imazhi i vetëm
- Inference me modelin e zgjedhur
- Shfaqje probabilitetesh raw
- UI dygjuhësh (anglisht / shqip)

---

## 3. Çfarë është aktualisht në progres

### 3.1 Faza 2 — Mjete të avancuara pasigurie

- **Parashikim konformal** (`master/uncertainty/conformal.py`): Implementimi i LAC është i përfunduar; po finalizoj implementimin e APS dhe diagnostikat per-klasë. Pritet të mbarojë brenda 2 javësh.
- **Monte Carlo Dropout**: Janë projektuar dy variantet (`cnn_mcd`, `resnet50_mcd`); trajnimi në vazhdim. Pas trajnimit, do bëhet analiza me T = 30 forward passes.
- **K-Fold Cross-Validation**: Script-i është gati (`master/run_kfold_cv.py`); planifikohet ekzekutimi për ResNet50 (~2 orë në CPU).
- **OOD Detection**: Implementimi i 4 metodave (MSP, Energy, Mahalanobis, Cosine) është në vazhdim.

### 3.2 Analiza statistikore

- Llogaritja e Bootstrap CI është e gatshme dhe e ekzekutuar për Fazën 1.
- Testet McNemar pairwise janë ekzekutuar.
- Analiza e kalibrimit (ECE, MCE, reliability diagrams) është gjysmë e mbaruar.

### 3.3 Zgjerim i Streamlit app

Po punohet për integrimin e:
- Probabiliteteve të kalibruara (pas T-scaling)
- Konformal sets në kohë reale
- Heuristikave të cilësisë së imazhit
- Eksportit PDF

---

## 4. Çfarë mbetet për t'u realizuar (~50% e mbetur)

### 4.1 Eksperimentet që mbeten

- **Faza 2 e plotë**: konformal evaluation, MC Dropout analysis, K-Fold CV, OOD detection
- **Faza 3 — Multi-class 5-stage grading**:
  - Trajnimi i `cnn_5class` dhe `resnet50_5class`
  - Llogaritja e Quadratic Weighted Kappa, ordinal distance
  - Multi-class conformal me coverage per-klasë
  - Multi-class calibration
- **Klasifikuesit klasikë** (Decision Tree, Random Forest, SVM) mbi DenseNet features

### 4.2 Pjesa e shkruar që mbetet

- Kreu IV — seksionet për Fazën 2 dhe Fazën 3
- Kreu IV — sinteza e gjetjeve dhe analiza e ndjeshmërisë
- Kreu V — Përfundimet, kufizimet, dhe punët në të ardhmen
- Përmbledhja (Abstrakti) në shqip dhe anglisht
- Diskutimi i implikimeve klinike

### 4.3 Komponentët mbështetës

- Finalizim i aplikacionit Streamlit me të gjitha veçoritë e UQ
- Zgjerim i bibliografisë në 40+ referenca (Vancouver style)
- Krijim i të gjitha figurave dhe tabelave përfundimtare
- Formatim sipas standardit UPT (Times New Roman 12pt, margjinat 2.54/3.8 cm, etj.)
- Tabela e Përmbajtjes, Lista e Figurave, Lista e Tabelave

---

## 5. Plani kohor për përfundimin

| Periudha | Aktivitetet kryesore |
|----------|---------------------|
| Maj 2026 (java 2-3) | Finalizimi i Fazës 2: conformal evaluation, MC Dropout, OOD |
| Maj 2026 (java 4) | K-Fold cross-validation për ResNet50 |
| Qershor 2026 (java 1-2) | Trajnimi dhe analiza e Fazës 3 (5-class grading) |
| Qershor 2026 (java 3-4) | Shkrimi i Kreut IV (rezultatet e plota) |
| Korrik 2026 (java 1-2) | Shkrimi i Kreut V (përfundimet) + sinteza e Kreut IV |
| Korrik 2026 (java 3-4) | Finalizim i Streamlit app me të gjitha veçoritë UQ |
| Gusht 2026 (java 1-2) | Zgjerim i bibliografisë, rishikim i Kreut II |
| Gusht 2026 (java 3-4) | Formatim final UPT, krijim TOC + lista figurash/tabelash |
| Shtator 2026 (java 1) | Rishikim i plotë me udhëheqësin |
| Shtator 2026 (java 2) | Korrigjime përfundimtare dhe printim për mbrojtje |
| Shtator 2026 (java 3) | **Dorëzimi përfundimtar dhe mbrojtja** |

---

## 6. Sfida të identifikuara dhe zgjidhjet

### 6.1 Trajnim CPU-only

**Sfida**: Mjedisi i zhvillimit është vetëm me CPU (MacBook Air M-series, 8GB RAM). Kjo kufizon arkitekturat dhe regjimet e trajnimit që janë realisht të ekzekutueshme.

**Zgjidhja**: Kjo është konvertuar në një pikë metodologjike — sistemi është dizajnuar të jetë i deployueshëm në mjedise me burime të kufizuara (klinika primare). Modelet e mëdha si Vision Transformers janë lënë për punët në të ardhmen.

### 6.2 Imbalanca në dataset

**Sfida**: APTOS ka shpërndarje shumë të çekuilibruar (50% No DR, vetëm 13% për fazat 3+4 së bashku).

**Zgjidhja**: Përdorim i `class_weight='balanced'` në trajnim; analiza per-klasë në Fazën 3 për të zbuluar performancën në klasat minoritare; conformal prediction me coverage per-klasë.

### 6.3 OOD validim me të dhëna reale

**Sfida**: Asnjë dataset i dytë i disponueshëm aktualisht për cross-dataset evaluation.

**Zgjidhja për tani**: Përdorim i zhurmës sintetike si OOD; infrastruktura për cross-dataset (`master/run_cross_dataset.py`) është e gatshme për kur Messidor-2 ose IDRiD të disponohen.

---

## 7. Statistika përmbledhëse

| Treguesi | Vlera aktuale | Targeti final |
|----------|--------------:|--------------:|
| Faqe të shkruara | ~25 | ~60 |
| Numri i fjalëve | ~8,500 | ≥17,000 |
| Numri i kapitujve të draft-uar | 3 (Kreu I, II, III) | 5 |
| Numri i tabelave të rezultatit | 3 | ~12 |
| Numri i figurave | 6 | ~24 |
| Modelet e trajnuara | 6 binary | 10 (6 binary + 2 MCD + 2 multi-class) |
| Referencat e bibliografisë | 20 | 40+ |

---

## 8. Përfundim

Punimi po zhvillohet sipas planit fillestar. Faza eksperimentale 1 është e përfunduar me rezultate konkurruese (ensemble ~96.5%), dhe infrastruktura e kodit për fazat e mëtejshme është e ngritur dhe e dokumentuar. Plani kohor i propozuar e bën të realizueshme dorëzimin përfundimtar në **Shtator 2026**, duke lënë kohë të mjaftueshme për rishikim me udhëheqësin shkencor dhe korrigjime përfundimtare.

Faleminderit për vëmendjen dhe drejtimin tuaj.

---

**Alketa Alia**

Kandidate për Master, Inxhinieri Informatike

[Email: aivalanche.2023@gmail.com]

Tiranë, [Data e dorëzimit]
