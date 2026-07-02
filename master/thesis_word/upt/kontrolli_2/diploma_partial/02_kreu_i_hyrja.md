# KREU I

# HYRJA

## 1.1 Përshkrimi i Problemit

Retinopatia diabetike (DR) është një ndërlikim mikrovaskular i diabetit mellitus dhe mbetet një nga shkaqet kryesore të verbimit të parandalueshëm tek të rriturit në moshë pune në mbarë botën. Sëmundja shfaqet kur hiperglicemia e zgjatur dëmton enët e vogla të gjakut që ushqejnë retinën; cikli i rrjedhshmërisë vaskulare, ischemisë, dhe neovaskularizimit aberrant, nëse nuk zbulohet, përfundon në shkëputjen e retinës dhe humbje të pakthyeshme të shikimit [1]. Sepse fazat e hershme janë kryesisht pa simptoma, screening-u i rregullt i fundus-it është thelbësor — por prevalenca globale e diabetit (vlerësuar mbi 537 milionë njerëz në 2024) tejkalon dukshëm kapacitetin e oftalmologëve të trajnuar, sidomos në vendet me të ardhura të ulëta dhe në zonat rurale. Mjetet e automatizuara të screening-ut që mund të triagojnë pacientët para rishikimit nga specialisti kanë bërë kështu fushë aktive të kërkimit.

Sistemi klinik i fazimit i përdorur nga oftalmologët dallon pesë nivele ashpërsie: Nuk ka DR (Faza 0), Mild non-proliferative (Faza 1), Moderate non-proliferative (Faza 2), Severe non-proliferative (Faza 3), dhe DR Proliferative (Faza 4). Çdo hap ka implikime të ndryshme menaxhimi, duke filluar nga vëzhgimi në fazën e butë deri tek ndërhyrja urgjente vitreoretinale në fazën proliferative. Një mjet screening-u klinikisht i dobishëm duhet kështu jo vetëm të dallojë "DR është prezent" nga "DR mungon" por gjithashtu të tregojë ashpërsinë në një mënyrë që harton mbi rrjedhën ekzistuese të referimit dhe trajtimit.

Një pipeline kuptimplotë screening-u duhet të adresojë tre pyetje më të thella që çdo deployment i një sistemi të tillë do duhej t'iu përgjigjej:

1. A janë dallimet ndërmjet modeleve kandidate statistikisht kuptimplote, apo janë zhurmë mostrimi?
2. Kur një model është gabim, a e di? A është konfidenca e tij e kalibruar, dhe a jep një sinjal pasigurie që mund të nxisë vendime me njeri-në-cikël?
3. Çfarë bën sistemi kur input-i nuk është një imazh fundus, ose kur pacienti ndodhet midis dy fazave të ashpërsisë?

Kjo tezë trajton këto tre pyetje si agjendën e saj kryesore të kërkimit.

## 1.2 Motivimi: Pse pasiguria ka rëndësi në AI mjekësor

Rrjetat moderne konvolucionale janë notoricisht mbi-konfidente. Një klasifikues binar që u jep "DR" me probabilitet 0.99 mund të jetë gabim shumë më shpesh se 1% të kohës, sidomos nën zhvendosjen e shpërndarjes ose për input-e atipike. Në fusha të padëmshme (reklamim, rekomandim), miskalibrim i tillë është mesatarisht i papërshtatshëm. Në mjekësinë e screening-ut, është i rrezikshëm: një "Nuk ka DR" i sigurt gabimisht mund të vonojë ndërhyrjen shpëtuese të shikimit, ndërsa një "DR" i sigurt gabimisht mund të nxisë referime të panevojshme që ngarkojnë sistemin shëndetësor.

Zhvillimet e fundit rregullatore kanë filluar të bëjnë kuantifikimin e pasigurisë një kërkesë në vend të një pasurie kërkimore. Korniza e Administratës së Ushqimit dhe Ilaçeve (FDA) e SHBA-së për Softuerin si Pajisje Mjekësore, Akti i Inteligjencës Artificiale i Bashkimit Europian (në fuqi nga 2026), dhe udhëzimi i Organizatës Botërore të Shëndetësisë mbi AI për shëndetin parashtrojnë të gjithë se AI mjekësor me rrezik të lartë duhet të ofrojë dëshmi të kalibrimit, qëndrueshmërisë ndaj input-eve jashtë-shpërndarjes, dhe vlerësime besueshmërie të interpretueshme. Një pipeline screening retinal që emeton vetëm një shifër "saktësi = 96%" nuk është i deployueshëm nën këto korniza; një që ekspozon probabilitete të kalibruara, sete parashikimi konformale, dhe shkallë novelty feature-space është.

Një argument i dytë, po aq i rëndësishëm, është klinik. Oftalmologët rishikojnë rregullisht raste të paqarta për opinione të dyta, dhe e gjithë struktura e screening-ut retinal telehealth është ndërtuar rreth triage — rastet e sigurta zgjidhen automatikisht ndërsa ato të pasigurta përshkallëzohen. Një sistem që prodhon vetëm një verdikt argmax pa output të qartë "nuk e di" nuk mund të integrohet pastër në këtë rrjedhë pune. Sistemi i zhvilluar në këtë tezë do prodhojë eksplicitisht tre shtresa output:

1. Një parashikim pikë me një prag të fortë vendimi.
2. Një probabilitet i kalibruar për çdo klasë, i validuar nga diagrame besueshmërie dhe temperature scaling post-hoc.
3. Një set parashikimi konformal që, me garanci coverage të provueshme, përmban etiketën e vërtetë të paktën në 90% (ose 95%) të kohës.

Të kombinuara, këto lejojnë një workflow klinik downstream të filtrojë rastet sipas besueshmërisë, duke përshkallëzuar tek një specialist vetëm ato realisht të paqarta.

## 1.3 Pyetjet kërkimore dhe kontributet

Kjo tezë organizohet rreth katër pyetjeve kërkimore:

**RQ1.** A ndryshojnë arkitekturat alternative të thella në performancë në detyrën APTOS 2019 pasi preprocessing dhe trajnimi janë konfiguruar drejtëzisht, dhe cili është roli i model ensembling?

**RQ2.** Sa të kalibruar janë modelet e trajnuara, dhe deri në çfarë mase temperature scaling post-hoc redukton gabimin e kalibrimit?

**RQ3.** Cili është sinjali më i dobishëm i pasigurisë për parashikim selektiv dhe detektim jashtë-shpërndarjes, dhe a mund të ofrojë një mbështjellës parashikimi konformal një garanci coverage finite-sample?

**RQ4.** A përgjithësohet e njëjta makineri pasigurie nga klasifikimi binar tek detyra klinikisht kuptimplote me 5 faza grading, dhe cili është workflow-i klinik që del prej saj?

Kontributet kumulative të planifikuara të kësaj pune janë përmbledhur më poshtë:

1. Një pipeline vlerësimi reproduktive, statistikisht rigoroz për datasetin APTOS 2019, me ndarje të stratifikuara, preprocessing arkitekturë-specifik, peshim të balancuar të klasave, dhe callbacks trajnimi të unifikuara. *(Përfunduar.)*

2. Një analizë kalibrimi për gjashtë klasifikues DR, që ekspozon mbi-konfidencë të paraporrtuar në VGG16 dhe demonstron një reduktim relativ ECE prej rreth 40% nëpërmjet temperature scaling. *(Në vazhdim.)*

3. Aplikimi i parë, sa di unë, i parashikimit split conformal me garanci formale të coverage për multi-stage DR grading në APTOS 2019. *(Në vazhdim.)*

4. Një krahasim krah-më-krah i Monte Carlo Dropout, mosmarrëveshjes së deep-ensemble, dhe detektimit feature-space OOD në një benchmark të vetëm. *(Në vazhdim.)*

5. Një model grading me 5 faza që pritet të arrijë QWK ≈ 0.85, konkurrues me leaderboard-et publike APTOS 2019, me sete konformale që përkthehen në një ndarje klinike auto-klasifiko/referoj rreth 70%/30%. *(Planifikuar — Faza 3.)*

6. Një aplikacion Streamlit open-source që ekspozon probabilitetet e kalibruara, mosmarrëveshjen e ensemble-it, dhe setet e parashikimit konformal për përdoruesit përfundimtarë në kohë reale. *(Prototip funksional.)*

## 1.4 Përmbledhje e metodologjisë

Dataset-i është koleksioni Kaggle APTOS 2019 Blindness Detection [1]: 3,662 imazhe fundus retinal me rezolucion 224 × 224 piksel, secila e etiketuar nga një klinicist me përvojë në shkallën e ashpërsisë 0-4. Të gjitha eksperimentet përdorin një ndarje të vetme, të fiksuar të stratifikuar 70/15/15 train/validation/test me random seed 123, kështu që çdo metrikë e raportuar në tezë llogaritet mbi të njëjtat 550 imazhe test të mbajtura.

Gjithë trajnimi kryhet në një laptop konsumatori vetëm-CPU (Apple Silicon M-series, 8 GB RAM), me TensorFlow 2.15. Asnjë GPU nuk u përdor. Kjo është një zgjedhje e qëllimshme: kufizon eksperimentet në arkitektura dhe regjime trajnimi që janë realisht të deployueshme në mjedise me burime të kufizuara, përfshirë llojin e kompjuterit të përdorur në klinikat e kujdesit primar.

Kodi është organizuar në katër pako kryesore Python: `lib/` për inferencën dhe ndërfaqen e përdoruesit Streamlit, `scripts/` për driver-in e unifikuar të trajnimit, `master/uncertainty/` për kalibrimin, konformalin, ensemble, OOD dhe modulet Monte Carlo Dropout, dhe `master/run_*.py` për skriptet e analizës që prodhuan çdo CSV, JSON dhe figurë në këtë dokument. I gjithë kodi është open-source.

## 1.5 Skema e tezës

Pjesa e mbetur e tezës është organizuar si vijon:

**Kreu II** rishikon sfondin klinik dhe teknik të nevojshëm për të motivuar metodologjinë: patofiziologjia dhe fazimi i DR-së, arkitekturat konvolucionale të përdorura si backbones, kalibrimi i klasifikatorëve të thellë, përafrime Bayesian nëpërmjet Monte Carlo Dropout, korniza e parashikimit konformal, dhe detektimi jashtë-shpërndarjes.

**Kreu III** përshkruan protokollin e unifikuar eksperimental — dataset, ndarje, augmentim, callbacks trajnimi, metrika vlerësimi — dhe implementimin e çdo moduli pasigurie.

**Kreu IV** raporton dhe diskuton rezultatet eksperimentale. Në këtë draft Kontrolli i Dytë, përfshihet Faza 1 (klasifikimi binar dhe analiza preliminare e kalibrimit); Faza 2 (parashikim konformal, MC Dropout, K-fold, dhe OOD detection) dhe Faza 3 (grading shumëklasor) janë në zhvillim aktiv dhe do të përfshihen në dorëzimin përfundimtar.

**Kreu V** (i planifikuar) do të mbyllë me një përmbledhje të kontributeve, implikimeve klinike, kufizimeve, dhe drejtimeve për punët në të ardhmen.

\newpage
