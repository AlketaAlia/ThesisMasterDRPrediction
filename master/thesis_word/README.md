# Master Thesis — Word-Ready Markdown Files

Drafts for each chapter of the master thesis, written in Markdown so that they paste cleanly into Word (the headers, lists, tables, and bold/italic styling are preserved).

## How to use

1. Open your existing Word thesis file.
2. For each `.md` file below, open it in any text editor (VS Code, Sublime, even TextEdit), select all (Cmd+A), copy (Cmd+C), and paste into Word at the appropriate location.
3. Word will preserve headings (Heading 1, Heading 2, …), tables, bullets, and bold/italic formatting.
4. Adjust styling (font, margins, line spacing) to match your university's template.
5. Replace placeholder text where indicated (e.g., `[TO FILL]`, supervisor name).

## Files (paste in this order)

| File | Maps to | Approx. pages |
|------|---------|--------------|
| `00_abstract.md` | ABSTRACT + ABSTRAKTI | 2-3 |
| `01_introduction.md` | CHAPTER 1: INTRODUCTION | 5-7 |
| `02_background.md` | CHAPTER 2: BACKGROUND | 10-14 |
| `03_methodology.md` | CHAPTER 3: METHODOLOGY | 12-16 |
| `04_phase1_baselines_calibration.md` | CHAPTER 4: PHASE 1 RESULTS | 10-14 |
| `05_phase2_conformal_ood_mcd.md` | CHAPTER 5: PHASE 2 RESULTS | 12-16 |
| `06_phase3_multiclass.md` | CHAPTER 6: PHASE 3 MULTI-CLASS | 10-14 |
| `07_discussion_conclusion.md` | CHAPTER 7: DISCUSSION & CONCLUSION | 6-9 |

**Total: approximately 70-90 pages**, depending on font size, line spacing, and figure inclusion.

## What still needs your work

1. **Title page**: replace bachelor-era text with your master programme name, supervisor, year (2026).
2. **Acknowledgements & Dedication**: rewrite or update from the bachelor version.
3. **Figures**: insert from `master/thesis/figures/` where the prose says "Figure 4.1", etc.:
   - `fig_phase1_accuracy_ci.pdf` — Phase 1 per-model accuracy with CI
   - `fig_phase1_risk_coverage.pdf` — Phase 1 risk-coverage curves
   - `fig_phase2_kfold.pdf` — Phase 2 5-fold CV bar chart
   - `fig_phase2_ood.pdf` — Phase 2 OOD AUROC + FPR
   - `fig_phase3_per_class.pdf` — Phase 3 per-class F1
   - And the existing PNG plots from `master/results/` (reliability diagrams, confusion matrices, etc.)
4. **References / bibliography**: copy from the bachelor's references list; new references to add are listed in `master/thesis/references.bib`. Key new ones:
   - Guo et al. 2017 (calibration)
   - Gal and Ghahramani 2016 (MC Dropout)
   - Lakshminarayanan et al. 2017 (Deep Ensembles)
   - Vovk et al. 2005 / Angelopoulos & Bates 2021 (conformal prediction)
   - Romano et al. 2020 (APS)
   - Hendrycks & Gimpel 2017 (MSP baseline)
   - Liu et al. 2020 (Energy score)
   - Lee et al. 2018 (Mahalanobis OOD)
   - Leibig et al. 2017 (uncertainty in DR)
5. **Appendices**: optionally include code listings from the `lib/`, `scripts/`, and `master/` packages.

## Tips for Word

- After pasting, select-all and apply your university's body-text style. Headings should pick up Heading 1 / 2 / 3 styles automatically.
- Tables will paste as Markdown text; in Word, select the rows, then Insert → Convert Text to Table → Separate text at: Other: `|`. Word will create a real table.
- For inline math like `$\hat p$`, Word's equation editor handles the LaTeX-style notation if you copy the `$...$` content into an equation field. Otherwise, replace with plain text (e.g., "p-hat").
- Save often.

## Supporting files (auto-generated, do not edit)

The following auto-generated artifacts are referenced in the prose and are kept up to date by re-running the analysis scripts:

- `master/results/per_model_summary.csv` — Phase 1 per-model metrics
- `master/results/mcnemar_pairwise.csv` — Phase 1 McNemar tests
- `master/results/phase2/conformal_results.csv` — Phase 2 conformal results
- `master/results/phase2/ood_metrics.json` — Phase 2 OOD metrics
- `master/results/mc_dropout/mc_dropout_summary.csv` — Phase 2 MC Dropout
- `master/results/kfold/resnet50_fold_summary.csv` — Phase 2 K-fold CV
- `master/results/multiclass/summary.csv` — Phase 3 multi-class metrics
- `master/results/multiclass/resnet50_5class_per_class.csv` — Phase 3 per-class F1
- `master/results/multiclass/resnet50_5class_conformal.csv` — Phase 3 multi-class conformal

To regenerate everything from scratch:

```bash
python -m master.run_uncertainty_analysis      # Phase 1
python -m master.run_phase2_analysis           # Phase 2 conformal + OOD
python -m master.run_kfold_cv --arch resnet50  # Phase 2 K-fold (slow, ~2 hours)
python -m master.run_mc_dropout_analysis       # Phase 2 MC Dropout
python -m master.run_multiclass_analysis       # Phase 3 multi-class
python -m master.generate_thesis_figures       # PDFs for Word
python -m master.generate_thesis_tables        # LaTeX tables (if you switch to LaTeX)
```
