# LISTA E SHKURTESAVE

| Shkurtesa | Shpjegimi |
|-----------|-----------|
| AI | Artificial Intelligence (Inteligjenca Artificiale) |
| APS | Adaptive Prediction Sets |
| APTOS | Asia Pacific Tele-Ophthalmology Society |
| AUC | Area Under the Curve |
| AUROC | Area Under the Receiver Operating Characteristic Curve |
| BALD | Bayesian Active Learning by Disagreement |
| CI | Confidence Interval (Interval Besueshmërie) |
| CNN | Convolutional Neural Network (Rrjet Neural Konvolucional) |
| CV | Cross-Validation |
| DR | Diabetic Retinopathy (Retinopatia Diabetike) |
| DT | Decision Tree (Pemë Vendimi) |
| ECE | Expected Calibration Error |
| EHR | Electronic Health Records |
| EU | European Union (Bashkimi Europian) |
| FDA | Food and Drug Administration |
| FPR | False Positive Rate |
| GAP | Global Average Pooling |
| HbA1c | Glycated Haemoglobin |
| ID | In-Distribution |
| IDRiD | Indian Diabetic Retinopathy Image Dataset |
| LAC | Least Ambiguous Classifier |
| MCD | Monte Carlo Dropout |
| MCE | Maximum Calibration Error |
| MSP | Maximum Softmax Probability |
| NLL | Negative Log-Likelihood |
| NPDR | Non-Proliferative Diabetic Retinopathy |
| OOD | Out-of-Distribution (Jashtë-Shpërndarjes) |
| PDR | Proliferative Diabetic Retinopathy |
| pp | Percentage Points (Pikë Përqindjeje) |
| QWK | Quadratic-Weighted Kappa |
| ReLU | Rectified Linear Unit |
| ResNet | Residual Network |
| RF | Random Forest (Pyll i Rastësishëm) |
| RQ | Research Question (Pyetje Kërkimore) |
| SaMD | Software as a Medical Device |
| SVM | Support Vector Machine |
| TPR | True Positive Rate |
| TS | Temperature Scaling |
| VEGF | Vascular Endothelial Growth Factor |
| VGG | Visual Geometry Group |
| ViT | Vision Transformer |
| WHO | World Health Organization (Organizata Botërore e Shëndetësisë) |

\pagebreak

# LISTA E TABELAVE

Tabela 4.1. Performanca për model me intervale besueshmërie bootstrap 95% në test set (N = 550)

Tabela 4.2. Rezultatet e testit pairwise McNemar midis gjashtë klasifikatorëve

Tabela 4.3. Saktësia selektive e ensemble në coverage të ndryshme

Tabela 4.4. Klasifikatorët klasikë baseline mbi veçoritë e nxjerra nga DenseNet121

Tabela 4.5. Parashikim konformal në α = 0.10 (target 90% coverage)

Tabela 4.6. Rezultatet e MC Dropout në test set me 550 imazhe

Tabela 4.7. Metrikat e detektimit OOD (AUROC dhe FPR @ TPR=95%)

Tabela 4.8. 5-fold cross-validation i ResNet50

Tabela 4.9. Rezultatet e grading-ut 5-fazor në test set (N = 550)

Tabela 4.10. Precision, recall dhe F1 për klasë për resnet50_5class

Tabela 4.11. Parashikim konformal multi-class për resnet50_5class

Tabela 4.12. Metrikat kryesore përgjatë tri fazave eksperimentale

\pagebreak

# LISTA E FIGURAVE

Figura 3.1. Shpërndarja e klasave të datasetit APTOS 2019 në pesë nivelet e ashpërsisë

Figura 4.1. Saktësia test për model me intervale besueshmërie bootstrap 95%

Figura 4.2. Heatmap i p-vlerave pairwise McNemar përgjatë gjashtë klasifikatorëve

Figura 4.3. Diagramet e besueshmërisë për VGG16 (raw dhe temperature-scaled)

Figura 4.4. Diagrami i besueshmërisë i ensemble heterogjen me 6 modele

Figura 4.5. Kurbat risk-coverage për ensemble binar nën katër sinjale pasigurie

Figura 4.6. Histogrami i mosmarrëveshjes së ensemble në test set binar

Figura 4.7. Dinamika e trajnimit të ResNet50 (transfer learning)

Figura 4.8. Dinamika e trajnimit të Xception (transfer learning)

Figura 4.9. Dinamika e trajnimit të DenseNet121 (transfer learning)

Figura 4.10. Dinamika e trajnimit të VGG16 (transfer learning)

Figura 4.11. Dinamika e trajnimit të CNN-së nga zero

Figura 4.12. Dinamika e trajnimit të variantit CNN(Tanh+ReLU)

Figura 4.13. Matricat e konfuzionit për tre klasifikatorët klasikë

Figura 4.14. Kurba risk-coverage për resnet50_mcd me MC Dropout

Figura 4.15. Histogrami i devijimit standard σ të MC Dropout (correct vs wrong)

Figura 4.16. AUROC për diskriminim ID-vs-OOD përgjatë katër metodave

Figura 4.17. Saktësia test e 5-fold cross-validation të ResNet50

Figura 4.18. Matrica e konfuzionit e resnet50_5class në test set

Figura 4.19. Diagrami i besueshmërisë multi-class i resnet50_5class

Figura 4.20. F1 për klasë për resnet50_5class në test set multi-class

Figura 4.21. Kurba risk-coverage për ensemble multi-class në Fazën 3

Figura 4.22. Dinamika e trajnimit të CNN-së nga zero (kokë 5-klasore)

Figura 4.23. Dinamika e trajnimit të ResNet50 (kokë 5-klasore)

Figura A.1–A.9. Diagrame besueshmërie dhe kurba ROC plotësuese (Shtojca C–E)

Figura A.10. Pamja e parashikimit binar në aplikacionin Streamlit (Shtojca G)

Figura A.11. Pamja e krahasimit shumë-model në aplikacionin Streamlit (Shtojca G)

Figura A.12. Pamja e grading-ut 5-fazor me conformal në aplikacionin Streamlit (Shtojca G)

\pagebreak
