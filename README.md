# Sequential Prediction of Professional Boxing Round Outcomes

This repository contains the replication code, summary data, figures, and LaTeX manuscript for the paper:

> **"Sequential Prediction of Professional Boxing Round Outcomes: Evaluating Scorecard State and Recurrent Memory Beyond Punch Statistics"**  
> *Targeted for IEEE Conference Class (IEEEtran, 6 pages).*

---

## Repository Structure

```
boxing-judging-paper/
├── .gitignore                      # Ignores raw Excel, panel.csv, and LaTeX build artifacts
├── README.md                       # This document
├── requirements.txt                # Python package dependencies
├── figures/                        # Generated PDF (vector) and PNG (300 DPI) figures
│   ├── fig1_attention_by_round.pdf & .png
│   └── fig2_heldout_metrics.pdf & .png
├── paper/                          # IEEEtran LaTeX manuscript
│   ├── IEEEtran.cls                # IEEE conference template class
│   ├── main.tex                    # 6-page IEEE paper source
│   ├── main.pdf                    # Compiled publication-ready PDF
│   └── references.bib              # Complete BibTeX bibliography
├── results/                        # Slim summary metrics & econometric coefficients
│   ├── stage2_coefficients.csv     # Econometric parameter estimates & SEs (M0, M1, M2)
│   ├── stage2_summary.json         # Log-likelihoods, AIC, BIC, and likelihood-ratio tests
│   ├── stage2_xgboost.json         # 5-fold CV logs and pooled XGBoost metrics
│   ├── stage3_attention_by_round.csv # Attention weights across rounds 2–12
│   ├── stage3_cv_metrics.json      # 5-fold CV metrics & memory-ablation statistics
│   └── stage3_summary.json         # In-sample CBSN fit and paired attention tests
├── scripts/
│   └── make_figures.py             # Rebuilds Fig 1 & Fig 2 from results/ into figures/
└── src/                            # Pipeline scripts (with configurable DATA_ROOT)
    ├── build_panel.py
    ├── cv_groups.py
    ├── jabbr_io.py
    ├── stage1_p_hat.py
    ├── stage2_fe_logit.py
    ├── stage2_xgboost.py
    └── stage3_cbsn.py
```

---

## 1. Installation

Install Python dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

---

## 2. Rebuilding Figures

All empirical metrics from econometric estimation and 5-fold cross-validation are pre-computed in `results/`. You can reproduce the publication figures in `figures/` at any time with:

```bash
python scripts/make_figures.py
```

This generates:
- `figures/fig1_attention_by_round.pdf` & `.png` (Attention dynamics across rounds 2–12)
- `figures/fig2_heldout_metrics.pdf` & `.png` (Held-out Log Loss and Accuracy across 5 folds for XGBoost vs CBSN)

---

## 3. Compiling the LaTeX Paper

The manuscript is written in standard IEEE conference format using `IEEEtran.cls` (bundled locally in `paper/`).

To compile the paper to PDF:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting `main.pdf` is strictly 6 pages with all tables, figures, formulas, and references resolved.

> **Authorship:** The manuscript is configured with Hritij Rana (Netaji Subhas University of Technology, Delhi, India; hritij.rana.pg26@nsut.ac.in).

---

## 4. Jabbr Data Location & Optional Full Refits

The raw Jabbr dataset (DeepStrike computer-vision Excel files and `pairedData.csv`) is external and proprietary. Per regulatory and licensing rules, raw data and reconstructed `panel.csv` are gitignored and not redistributed in this bundle.

**Pre-computed results in `results/` are already sufficient to regenerate all figures and compile the paper without retraining any model.**

If you have local access to the Jabbr data folder (`Jabbr-AI-Boxing-Judging`), you can run full refits:

1. **Set data path:**
   By default, scripts look for `Jabbr-AI-Boxing-Judging` adjacent to this repository (`../Jabbr-AI-Boxing-Judging`). You can also point to a custom directory via the environment variable:
   ```bash
   export DATA_ROOT="/path/to/Jabbr-AI-Boxing-Judging"
   # or on Windows PowerShell:
   $env:DATA_ROOT="D:\path\to\Jabbr-AI-Boxing-Judging"
   ```

2. **Run pipeline stages:**
   ```bash
   python src/build_panel.py      # Build panel.csv from paired scorecards (13,776 rounds)
   python src/stage1_p_hat.py     # Fit L1 punch baseline (\hat{P})
   python src/stage2_fe_logit.py  # Fit M0, M1, M2 econometric fixed-effects models
   python src/stage2_xgboost.py   # Run 5-fold bout-grouped CV with XGBoost
   python src/stage3_cbsn.py      # Train causal CBSN (LSTM) with GroupKFold
   ```

---

## Key Empirical Findings

1. **Compensatory Officiating ($H_1$ Supported):**
   Ringside judges exhibit significant compensatory catch-up bias ($\beta = -0.3016$, $z = -12.63$, $p = 1.40 \times 10^{-36}$). Holding objective punch output constant, each point a fighter leads on a judge's card reduces their odds of winning the next round by 26%.
2. **Late-Round Attenuation ($H_2$ Refuted):**
   The interaction with championship rounds 11–12 is positive ($\beta = +0.0990$, $p = 1.18 \times 10^{-4}$), demonstrating that catch-up bias weakens under the imminent finality of late rounds.
3. **Machine Learning Model Comparison:**
   - **XGBoost (30 punches + $\Delta_{\text{score}}$ + Late):** Held-out Log Loss **0.468**, Accuracy **76.7%**.
   - **CBSN (Causal LSTM):** Held-out Log Loss **0.483**, Accuracy **76.1%**.
   - Causal sequence models do not outperform gradient boosted decision trees on short, noisy combat sports timelines.
