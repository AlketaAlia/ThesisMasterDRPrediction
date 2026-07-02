# KREU II

# SFONDI TEORIK

Ky kapitull vendos sfondin klinik dhe teknik të nevojshëm për të motivuar metodologjinë e tezës. Seksioni 2.1 rishikon patofiziologjinë dhe gradimin klinik të retinopatisë diabetike dhe përshkruan dataset-in APTOS 2019. Seksioni 2.2 përmbledh arkitekturat konvolucionale të përdorura si backbones në këtë tezë. Seksionet 2.3 deri 2.6 prezantojnë katër shtyllat e metodologjisë së vetëdijshme ndaj pasigurisë: kalibrimi, përafrimet Bayesian, parashikimi konformal, dhe detektimi jashtë-shpërndarjes. Seksioni 2.7 anketon punët e lidhura që kombinojnë këto ide në DR ose në detyra të tjera të imazherisë mjekësore.

> *Shënim për Kontrollin e Dytë: ky kapitull është draft i parë i përfunduar. Pjesë të literaturës dhe diskutimit shtesë janë në rishikim, përfshirë: zgjerim të lit review-it për foundation models, zgjerim të diskutimit rregullator, dhe shtim të punës më të fundit në UQ në mjekësi.*

## 2.1 Retinopatia diabetike dhe Dataset-i APTOS 2019

### 2.1.1 Patofiziologjia

Retinopatia diabetike është një ndërlikim mikrovaskular i diabetit mellitus, në të cilin hiperglicemia e zgjatur dëmton enët e vogla të gjakut që ushqejnë retinën. Sëmundja progreson nëpërmjet një kaskade mekanizmash që mbivendosen: humbja e periciteve që çon në fragilitet vaskular, formimi i mikroaneurizmave, rritja e përshkueshmërisë vaskulare dhe oedema makulare konseguente, okluzioni kapilar dhe ishemia që rezulton, dhe së fundi neovaskularizimi i drejtuar nga vascular endothelial growth factor (VEGF). Çdo fazë prodhon gjetje karakteristike në ekzaminimin e fundus-it, të përmbledhura në Shkallën Klinike Ndërkombëtare të Ashpërsisë së DR të përdorur nga Akademia Amerikane e Oftalmologjisë dhe e adoptuar nga dataset-i APTOS 2019:

- **Faza 0 — Nuk ka DR**: pa patologji të dukshme retinale.
- **Faza 1 — DR non-proliferative i butë**: prezenca e një numri të vogël mikroaneurizmash.
- **Faza 2 — DR non-proliferative moderate**: më shumë sesa mikroaneurizma por më pak se NPDR i ashpër; mund të përfshijë ekzudate të forta dhe hemorragji dot/blot.
- **Faza 3 — DR non-proliferative i ashpër**: cilido nga kriteret "4-2-1" (më shumë se 20 hemorragji intraretinale në çdo nga katër kuadrantet; ose beading definit venoz në dy ose më shumë kuadrante; ose anomalitë mikrovaskulare intraretinale të dukshme në një ose më shumë kuadrante).
- **Faza 4 — DR Proliferative (PDR)**: neovaskularizim i diskut ose në vende të tjera, me ose pa hemorragji preretinale/vitreale.

Trajtimi shkallëzohet në përputhje. Pacientët në Fazën 0 ose Fazën 1 zakonisht janë të planifikuar për rishikim vjetor rutinor. Faza 2 mund të meritojë një interval më të afërt të follow-up. Fazat 3 dhe 4 kërkojnë referim tek një oftalmolog, me injeksione anti-VEGF dhe panretinal photocoagulation si bazat e trajtimit në Fazën 4 për të parandaluar shkëputjen e retinës dhe humbjen e pakthyeshme të shikimit. Implikimi klinik për një sistem screening i automatizuar është kështu i qartë: një output binar "DR / Nuk ka DR" është informacionalisht me humbje. Një pacient me Mild DR dhe një pacient me PDR marrin të njëjtin etiketim por menaxhime shumë të ndryshme.

### 2.1.2 Dataset-i APTOS 2019 Blindness Detection

Dataset-i i përdorur përgjatë kësaj teze është koleksioni APTOS 2019 Blindness Detection [1], i hostuar në Kaggle nga Asia Pacific Tele-Ophthalmology Society. Përbëhet nga 3,662 imazhe fundus retinal, secila e prerë dhe e ridimensionuar në 224 × 224 piksel dhe e etiketuar nga një klinicist me përvojë në shkallën pesë-fazore të përshkruar më sipër. Shpërndarja e klasave është rëndë e zhvendosur drejt Fazës 0: përafërsisht gjysma e imazheve mbartin etiketën Nuk ka DR, ndërsa Fazat 3 dhe 4 klinikisht më urgjente së bashku përbëjnë më pak se 14%. Çdo klasifikues i suksesshëm në këtë dataset duhet kështu të trajtojë çekuilibrim të rëndë të klasave, ose nëpërmjet humbjes me peshime klasash, oversampling, ose funksione humbjeje në stil focal.

Për versionin binar të detyrës, Fazat 1 deri 4 kolapsohen në një klasë të vetme "DR", duke prodhuar një ndarje afërsisht 50/50. Ky formulim është i përshtatshëm për benchmarking por është informacionalisht me humbje në kuptimin klinik; për këtë arsye, kjo tezë vlerëson si kolapsin binar ashtu edhe shkallën origjinale 5-klasore të ashpërsisë në Kreun IV (Faza 3, e planifikuar).

## 2.2 Arkitekturat konvolucionale

Teza vlerëson gjashtë arkitektura të thella dhe tre klasifikues klasikë. Arkitekturat e thella janë përmbledhur këtu; metodat klasike janë prezantuar në Seksionin 3.5 të kapitullit të metodologjisë.

### 2.2.1 ResNet50

ResNet50 [2] prezantoi mësimin rezidual me skip connections që anashkalojnë dy ose tre shtresa konvolucionale. Arkitektura ka 50 shtresa të organizuara në pesë faza, me përafërsisht 25.6 milionë parametra. Skip connections zbusin problemin e gradient-zhdukës dhe mundësojnë trajnimin e rrjeteve shumë më të thella se ana paraprake e teknologjisë. Në tezën aktuale, ResNet50 përdoret si një backbone transfer-learning me peshat e paratrajnuara ImageNet dhe një kokë dense me 128 njësi.

### 2.2.2 DenseNet121

DenseNet121 [3] e merr idenë e skip connections në ekstrem: çdo shtresë në një bllok dense merr output-et e bashkangjitura të të gjitha shtresave paraprake. Varianti me 121 shtresa ka vetëm ~7 milionë parametra falë ndarjes së parametrave të mundësuar nga ripërdorimi i veçorive. Lidhshmëria dense gjithashtu prodhon një efekt implicit deep-supervision që inkurajon ripërdorimin e veçorive dhe shkurton shtigjet e gradient-it.

### 2.2.3 Xception

Xception [4] zëvendëson konvolucionet standarde me konvolucione të ndashme depthwise që faktorizojnë konvolucionin spacial dhe channel-wise. Arkitektura ka 36 shtresa të organizuara në rrjedhat Entry, Middle, dhe Exit, me ~21 milionë parametra.

### 2.2.4 VGG16

VGG16 [5] është një arkitekturë më e thjeshtë, më e thellë që përdor vetëm konvolucione 3×3 dhe max-pooling 2×2. Uniformiteti i saj e bën një zgjedhje të popullarizuar pedagogjike dhe transfer-learning.

### 2.2.5 CNN-ja Custom me 3-blloqe

Për të ankoruar krahasimin, teza përfshin gjithashtu një CNN nga zero me tre blloqe konvolucion-pooling (32 → 64 → 128 filtra) të ndjekur nga një shtresë dense me 128 njësi dhe një output sigmoid. Modeli ka ~11 milionë parametra dhe trajnohet mbi input grayscale të replikuar. Një variant i dytë (CNN Tanh+ReLU) zëvendëson tanh për ReLU në bllokun e dytë konvolucional dhe në kokën dense. Analiza statistikore e paraqitur në Kreun IV tregon se varianti all-ReLU e tejkalon qartë variantin me aktivizim të përzier në këtë detyrë.

## 2.3 Kalibrimi i klasifikatorëve të thellë

Një klasifikator thuhet se është i kalibruar mirë nëse probabilitetet e tij të parashikuara përputhen me frekuencat e vërejtura: midis input-eve për të cilat modeli prodhon probabilitet 0.8, përafërsisht 80% duhet të jenë të klasifikuara saktë. Rrjetat e thella moderne zakonisht NUK janë të kalibruara mirë; Guo et al. [6] treguan se edhe rrjetat me saktësi të lartë priren të jenë mbi-konfidentë, me hendekun që zgjerohet ndërsa arkitekturat rriten më të thella dhe përdorin rregullarizim më të fortë.

### 2.3.1 Gabimi i pritur dhe maksimal i kalibrimit

Masa standarde skalare e kalibrimit është Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

ku parashikimet janë të ndara në M shporta të barabarta të konfidencës $B_m$, $\text{acc}(B_m)$ është saktësia empirike në shportën $m$, dhe $\text{conf}(B_m)$ është konfidenca mesatare në shportë. ECE = 0 tregon kalibrim perfekt; në praktikë, vlerat nën 5% konsiderohen të pranueshme, ndërsa vlerat mbi 10% tregojnë mbi-konfidencë klinikisht problematike.

Maximum Calibration Error (MCE) zëvendëson shumën e peshuar me një maksimum mbi shportat; kap hendekun më të keq të kalibrimit dhe është i ndjeshëm ndaj rajoneve të vogla të keq-kalibruara të rangut të konfidencës që ECE do t'i mesatarizonte [7].

### 2.3.2 Diagramet e besueshmërisë

Diagramet e besueshmërisë plotojnë saktësinë për shportë kundër konfidencës mesatare për shportë. Diagonalja korrespondon me kalibrim perfekt; barat nën diagonale tregojnë mbi-konfidencë dhe barat mbi diagonalen tregojnë nën-konfidencë.

### 2.3.3 Temperature scaling

Temperature scaling [6] është një rregullim post-hoc një-parametër në të cilin logit pre-sigmoid $z$ pjesëtohet me një temperaturë $T$ përpara aplikimit të sigmoid:

$$\hat{p} = \sigma(z / T)$$

Vlerat e $T > 1$ butësojnë parashikimet (duke reduktuar konfidencën e theksuar), ndërsa $T < 1$ i mpreh ato. Temperatura përshtatet duke minimizuar log-likelihood-in negativ në një validation set të mbajtur. Kritike, $T$ nuk e ndryshon parashikimin argmax, kështu që saktësia ruhet ekzakt; vetëm konfidenca rishkallëzohet.

## 2.4 Mësimi i thellë Bayesian dhe Monte Carlo Dropout

Pamja Bayesian e një rrjeti neural i trajton peshat si ndryshore të rastësishme me një shpërndarje posteriore mbi të dhënat e trajnimit. Parashikimi në një input të ri kërkon kështu marginalizimin mbi posteriorin, që është i pa kapshëm në formë të mbyllur. Janë propozuar një gamë përafrimesh; ai më i aksesueshëm llogaritëse është Monte Carlo Dropout (MCD), i prezantuar nga Gal dhe Ghahramani [8].

### 2.4.1 MC Dropout si inferencë variacionale

Gal dhe Ghahramani [8] treguan se një rrjet i trajnuar me rregullarizim dropout mund të interpretohet si duke kryer inferencë variacionale nën një familje të caktuar përafruese. Në kohën e inferencës, dropout-i lihet aktiv dhe rrjeti ekzekutohet T herë për të njëjtin input, duke prodhuar një shpërndarje parashikimesh. Mesatarja e T mostrave është vlerësimi pikë, ndërsa varianca është një përafrim me pasigurinë epistemike të modelit.

### 2.4.2 Dekompozimi i pasigurisë

Për klasifikim binar me output sigmoid p, entropia parashikuese e parashikimit mesatar $H(\bar p)$ mat pasigurinë totale, ndërsa entropia mesatare për mostër e mostrave dropout $\mathbb{E}_t[H(p_t)]$ përafron komponentin aleatorik (i ngulitur në të dhëna). Diferenca e tyre është informacioni i ndërsjellë midis parashikimit dhe parametrave të modelit, ndonjëherë i quajtur BALD score sipas kornizës Bayesian Active Learning by Disagreement.

### 2.4.3 Deep Ensembles

Një qasje plotësuese, edhe më e thjeshtë, është të trajnohen disa kopje të pavarura të të njëjtës arkitekturë me iniciativa të ndryshme të rastësishme [9]. Parashikimi mesatar i ensemble-it është zakonisht më i saktë dhe më i kalibruar se cilido anëtar i vetëm. Mosmarrëveshja midis anëtarëve siguron një sinjal pasigurie direkt analog me variancën MC Dropout.

Teza aktuale do përdorë si MCD ashtu edhe një ensemble heterogjen përgjatë gjashtë arkitekturave binare.

## 2.5 Parashikim Konformal

Parashikimi konformal është një kornizë, fillimisht për shkak të Vovk et al. [10] dhe e popullarizuar së fundi nga Angelopoulos dhe Bates [11], për prodhimin e seteve të parashikimit me garanci coverage finite-sample, të pavarura nga shpërndarja. Duke pasur parasysh një shkallë miscoverage të synuar α, parashikimi konformal prodhon, për çdo input test $x$, një set $C(x)$ i tillë që

$$\Pr\left( y \in C(x) \right) \geq 1 - \alpha$$

nën supozimin se të dhënat e kalibrimit dhe test-i janë të shkëmbyeshme.

### 2.5.1 Parashikim Split (Induktiv) Konformal

Versioni i përdorur në këtë tezë është parashikimi split konformal:

1. Mbaj një set kalibrimi $\{(x_i, y_i)\}$ të ndarë nga seti i trajnimit.
2. Për çdo mostër kalibrimi, llogarit një score moskonformiteti $s_i$ që kuantifikon sa surprizuese është etiketa e vërtetë nën klasifikuesin e trajnuar.
3. Llogarit kuantilin empirik $(1 − α)$ $\hat q$ të scores.
4. Për një input test $x$, përfshi në $C(x)$ çdo etiketë kandidate, score moskonformiteti i së cilës është $\leq \hat q$.

### 2.5.2 Funksionet e score-it

Dy score moskonformiteti vlerësohen në këtë tezë: **Least Ambiguous Classifier (LAC)** dhe **Adaptive Prediction Sets (APS)** [12]. APS prodhon madhësi mesatare seti më të vogla dhe priret të jetë klinikisht më i dobishëm sepse mund të prodhojë sete dy-klasore "abstain" për input-e të paqarta.

## 2.6 Detektim jashtë-shpërndarjes

Një sistem klinikisht i deployuar do të marrë në mënyrë të pashmangshme input-e jashtë shpërndarjes së trajnimit. Modeli duhet të refuzojë të parashikojë mbi input-e të tilla në vend që të prodhojë me siguri një përgjigje të gabuar.

### 2.6.1 Metodat e hapësirës-output

Baza më e thjeshtë është Maximum Softmax Probability (MSP) [13]: input-et OOD priren të prodhojnë konfidencë më të ulët se input-et ID. MSP vuan kur softmax saturohet — një rrjet mund të detyrohet në konfidencë shumë të lartë edhe në zhurmë. Energy score [14] anashkalon saturim e softmax duke llogaritur log-sum-exp negative të logits.

### 2.6.2 Metodat e hapësirës-veçori

Një familje më e fuqishme metodash krahason pozicionin e input-it test në hapësirën e veçorive me shpërndarjen e veçorive të trajnimit. Distanca Mahalanobis [15] përshtat densitete Gaussian për klasë në veçoritë e trajnimit dhe shkallëzon input-et test me log-likelihood-in negativ nën Gaussian-in e klasës më të afërt të kushtëzuar. Distanca cosine ndaj qendrës in-distribution është një variant më i thjeshtë.

## 2.7 Punët e lidhura në DR me pasiguri

Një trupë në rritje e punës aplikon vlerësimin e pasigurisë në detektimin e DR. Leibig et al. [16] përdorën MC Dropout në imazhet EyePACS dhe treguan se parashikimi selektiv përmirëson dukshëm precizionin e screening-ut të automatizuar. Laves et al. [18] raportuan pasigurinë e regresionit të kalibruar mirë në imazherinë mjekësore me një fokus mbi trashësinë e shtresës retinale. Gulshan et al. [17] publikuan punimin landmark JAMA që demonstron se një CNN i thellë mund të përputhej me performancën e nivelit oftalmolog në detektimin e DR referueshëm, por nuk adresoi pasigurinë.

Megjithatë, shumë pak studime kanë aplikuar parashikim konformal në DR, dhe edhe më pak e kanë bërë këtë në dataset-in APTOS 2019. Teza aktuale synon të mbyllë këtë boshllëk duke kombinuar (i) një analizë të plotë kalibrimi, (ii) dy mekanizma të pavarur pasigurie (MCD dhe mosmarrëveshje ensemble), (iii) parashikim split konformal me coverage formal në dy nivele besueshmërie, dhe (iv) detektim feature-space OOD — të gjitha në të njëjtin set test të fiksuar.

Zgjedhjet metodologjike dhe justifikimi i tyre janë përshkruar në detaje në kapitullin tjetër.

\newpage
