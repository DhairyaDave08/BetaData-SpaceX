# 🚀 Space Mission Risk Analytics

**MSoC 2026 Hackathon — Data Science & ML Track — Problem Statement 03**
*Microsoft Student Technical Club, DAU*

Predicting satellite/rocket launch success probability from historical mission conditions — vehicle reliability, launch site, payload class, and weather — with **calibrated risk scores**, **per-mission SHAP explanations**, and an **interactive analytics dashboard**.

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Setup & Reproduction](#setup--reproduction)
- [Known Limitations & Data Quality Issues](#known-limitations--data-quality-issues)
- [Team](#team)

---

## Overview

Space launch is one of the most capital-intensive, risk-laden activities in engineering — a single failure can destroy years of satellite work and hundreds of millions of dollars. This project builds a **binary classifier with calibrated probability outputs** that estimates mission success probability, paired with an analytics layer that helps mission planners understand *why* a launch is flagged as risky — not just that it is.

**Core principle guiding every design decision in this repo: calibration and honesty over inflated accuracy.** A model that's 90% "accurate" by always predicting success is useless. We optimized instead for calibrated probabilities, PR-AUC on the failure class, and transparent documentation of every limitation.

### What this project delivers (mapped to hackathon objectives)

| # | Objective | Status |
|---|-----------|--------|
| 1 | Binary classifier with calibrated probabilities | ✅ Calibrated Random Forest (isotonic) |
| 2 | Features from rocket type, payload, site, weather, reliability | ✅ See [Methodology](#methodology) |
| 3 | High-cardinality categorical handling | ✅ Frequency-based grouping (min_count=10) |
| 4 | Calibrated probability outputs | ✅ Isotonic regression, Brier score reported |
| 5 | Per-mission SHAP attribution | ✅ Permutation explainer, plain-language output |
| 6 | Analytics dashboard (vehicle/site/decade/payload slices) | ✅ Streamlit `dashboard/app.py` |
| 7 | What-if simulator | ✅ Live prediction + SHAP in dashboard |
| 8 | Documented data quality issues | ✅ See [Known Limitations](#known-limitations--data-quality-issues) |

---

## Live Demo

- **Demo video:** [YouTube link — unlisted] *(add link here)*
- **Dashboard:** https://betadata-spacex-yfphzp4z7rfsq2wregcdha.streamlit.app/ 🚀
- **Dashboard:** run locally via `streamlit run dashboard/app.py` (see [Setup](#setup--reproduction))
- **Presentation deck:** [`presentation/BetaData-SpaceX.pdf`](presentation/BetaData-SpaceX.pdf)

---

## Dataset

**Primary source:** [All Space Missions from 1957](https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957) (Kaggle) — 4,324 launches, 1957–2020.

**Weather enrichment:** [Open-Meteo Historical Weather API](https://open-meteo.com/) (ERA5 reanalysis, free, no API key) — real historical wind speed, temperature, and precipitation for the **top 10 launch facilities by volume**, covering **759 of 4,198 launches (18.1%)** after cleaning. See [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) for full citation and licensing details.

We deliberately scoped weather enrichment to high-volume sites rather than attempting full coverage — see [Known Limitations](#known-limitations--data-quality-issues) for why.

## Repository Structure
```text
space-mission-risk-analytics/
│
├── README.md                       # Project overview, setup instructions, usage
├── RESULTS.md                      # Final held-out test metrics
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files and directories ignored by Git
│
├── data/
│   ├── raw/
│   │   ├── Space_Corrected.csv     # Original Kaggle dataset         
│   ├── processed/
│   │   └── features.csv            # Final engineered feature set
│   └── DATA_SOURCES.md             # Dataset citations, licenses, preprocessing notes
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Leak-free feature engineering & weather enrichment
│   ├── 03_modeling_baseline.ipynb  # Logistic Regression baseline
│   ├── 04_modeling_final.ipynb     # Calibrated Random Forest (final model)
│   └── 05_shap_analysis.ipynb      # SHAP explainability analysis
│
├── src/
│   ├── data_loader.py              # Load and clean raw dataset
│   ├── feature_engineering.py      # Leak-free rolling reliability, payload proxy, grouping
│   ├── weather_enrichment.py       # Open-Meteo integration with decade-chunked caching
│   ├── train.py                    # Era-stratified train/test split and model pipelines
│   ├── calibrate.py                # Isotonic/Platt calibration utilities
│   ├── explain.py                  # SHAP explainer (encoded categorical workaround)
│   └── predict.py                  # Inference layer used by the dashboard
│
├── models/
│   ├── model.pkl                   # Trained and calibrated Random Forest model
│   └── model_metadata.json         # Metrics snapshot and operating threshold
│
├── dashboard/
│   ├── app.py                      # Streamlit application entry point
│   └── components/
│       ├── historical_view.py      # Historical success rate visualizations
│       ├── whatif_simulator.py     # Interactive mission risk predictor
│       └── shap_display.py         # Human-readable SHAP explanations
│
├── tests/
│   └── test_no_leakage.py          # Ensures rolling features use only past data
│
└── presentation/
    ├── slides.pptx
    ├── calibration_curve.png
    ├── pr_curve_failure.png
    └── shap_global_importance.png
```

## Methodology

### 1. Data Cleaning
Raw launch records (rocket + payload strings, free-text locations, 4-class mission status) are parsed into structured fields: rocket family, launch site, country, launch date. The original `Status Mission` (Success / Failure / Partial Failure / Prelaunch Failure) is collapsed to a binary target — **Partial Failures are treated as failures**, a simplification documented in [Known Limitations](#known-limitations--data-quality-issues).

### 2. Feature Engineering (`src/feature_engineering.py`)
- **Leak-free rolling reliability**: for every launch, we compute the rocket family's, launch site's, and country's historical success rate using **only strictly prior launches** — implemented via `.shift(1).expanding().mean()` and verified with a dedicated unit test (`tests/test_no_leakage.py`).
- **Vehicle maturity**: days since a rocket family's first-ever launch — young vehicles are known to fail more often, a real, well-documented aerospace pattern.
- **Payload capacity proxy**: the dataset has no payload mass column, so we use a documented static lookup of publicly known approximate LEO payload capacities per rocket family, flagged via `payload_capacity_known` so the model (and any reader) can distinguish real matches from fallback values.
- **High-cardinality grouping**: 339 raw rocket families were reduced to 92 groups (`min_count=10` threshold, chosen after explicitly comparing coverage tradeoffs at multiple thresholds — see `notebooks/02_feature_engineering.ipynb`), preventing overfitting on a ~4,200-row dataset while retaining 80.5% of category-level signal.
- **Weather enrichment**: real historical weather (`wind_speed_max_kmh`, `temp_max_c`, `precipitation_mm`) fetched via Open-Meteo for the top 10 launch facilities, decade-chunked and cached to avoid redundant API calls. Launches outside this scope are explicitly flagged `weather_available=False` rather than imputed.

### 3. Modeling (`src/train.py`, `notebooks/04_modeling_final.ipynb`)
- **Split**: chronological (era-stratified), not random — the most recent ~20% of launches form the test set, respecting the "no future leakage" constraint and mirroring real deployment (predicting upcoming launches from historical data only).
- **Baseline**: Logistic Regression with `class_weight="balanced"`, ROC-AUC 0.72–0.73.
- **Final model**: Random Forest (100 estimators, max_depth=6, `class_weight="balanced"`) — chosen over XGBoost after XGBoost's calibration step repeatedly collapsed discrimination on this small, imbalanced dataset (see [Results](#results) for the full diagnostic story).

### 4. Calibration (`src/calibrate.py`)
Isotonic regression, applied via `CalibratedClassifierCV`. We explicitly verify calibration does **not** destroy discrimination by comparing uncalibrated vs. calibrated ROC-AUC at every step — a check most naive calibration pipelines skip, and one that caught a real bug during development (see [RESULTS.md](RESULTS.md)).

### 5. Explainability (`src/explain.py`)
Model-agnostic SHAP `Permutation` explainer (the final model is a `CalibratedClassifierCV`-wrapped pipeline, not a raw tree SHAP can introspect directly). Categorical features are numerically encoded before SHAP computation (a known SHAP limitation with mixed-type inputs) and decoded back to real category names in all outputs, so explanations remain fully human-readable.

### 6. Threshold Selection
Rather than the default 0.5 cutoff — which, on a 90%-success dataset, trivially predicts "success" for everything — we selected an **F1-optimal threshold on the failure class** from the full precision-recall curve, explicitly trading some precision for meaningfully higher failure recall. This choice is documented and justified in `models/model_metadata.json` and `RESULTS.md`.

---

## Results

See [`RESULTS.md`](RESULTS.md) for full held-out test metrics.

**Headline numbers:**
- **ROC-AUC: 0.749** (calibrated and uncalibrated — identical, confirming no discrimination loss from calibration)
- **Brier score: 0.047** (down from 0.165 uncalibrated)
- **PR-AUC (failure class): 0.33** — roughly **5–6x better than random** given the test set's ~5.7% failure base rate
- **At chosen operating threshold: 48% precision, 33% recall on failure detection**

**Why these numbers, not higher:** rocket failures are driven substantially by engineering and manufacturing factors (specific component failures, material defects) that no publicly available historical metadata captures. This model predicts *systemic* risk (vehicle maturity, site experience, era) — not the specific technical fault of any individual launch. A PR-AUC of ~0.31–0.33 for this class of problem, using only metadata, represents genuine, honest signal rather than a claim of high-confidence individual-launch prediction. See `RESULTS.md` for the full reasoning and validation checks (including a baseline comparison and a documented calibration bug we found and fixed mid-development).

---

## Setup & Reproduction

### Requirements
```bash
pip install -r requirements.txt
```

### Option A — Run the full pipeline from scratch (Google Colab recommended)
```python
!git clone https://github.com/DhairyaDave08/BetaData-SpaceX.git
%cd BetaData-SpaceX
!pip install -q -r requirements.txt
```
Then run notebooks in order: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_modeling_baseline.ipynb` → `04_modeling_final.ipynb` → `05_shap_analysis.ipynb`.

Each notebook is self-contained and regenerates its required inputs from `data/raw/Space_Corrected.csv` and cached weather JSON — no manual intermediate file management needed.

### Option B — Run the dashboard only (uses pre-trained model)
```bash
git clone https://github.com/DhairyaDave08/BetaData-SpaceX.git
cd BetaData-SpaceX
pip install -r requirements.txt
streamlit run dashboard/app.py
```
Requires `models/model.pkl`, `models/model_metadata.json`, and `data/processed/features.csv` to be present in the repo (all committed).

### Running tests
```bash
python tests/test_no_leakage.py
```

---

## Known Limitations & Data Quality Issues

Documented transparently per Objective #8:

- **No payload mass column** in the source dataset — addressed via a documented public-knowledge lookup table (`ROCKET_PAYLOAD_CAPACITY_KG` in `src/feature_engineering.py`), not fabricated per-launch data. Rockets not in the lookup use a median fallback, flagged explicitly.
- **Cost data (`Rocket` column) is ~78% missing** — not used as a model feature; documented rather than imputed to avoid introducing fabricated signal.
- **4-class mission status collapsed to binary** — Partial Failures are treated as failures, a simplification: a wrong-orbit partial failure is not operationally identical to a total loss, but the dataset doesn't support finer-grained modeling with reasonable sample sizes per class.
- **Weather coverage is 18.1% of launches** (top 10 facilities only) — a deliberate scoping decision documented in `data/DATA_SOURCES.md`, not a hidden gap. Remaining launches are explicitly flagged `weather_available=False`.
- **Small dataset overall** (~4,200 launches after cleaning, ~230 failures) — this is a hard constraint of the domain (there have only been so many orbital launch attempts in history), addressed via conservative model regularization, high-cardinality category grouping, and an honest reporting of PR-AUC relative to the base rate rather than inflated accuracy claims.
- **Historical data is biased toward older, Soviet-era vehicles** by launch volume (e.g., Cosmos-3M alone accounts for 422 launches) — success-rate patterns for high-volume historical vehicles may not generalize cleanly to modern commercial launch providers with very different engineering practices.

---

## Team

Built for **MSoC 2026 Hackathon** by **Team BetaData**.

### Team Members

- **Dhairya Dave**
- **Rudra Bhatt**


## License & Acknowledgments

- Dataset: [Kaggle — All Space Missions from 1957](https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957)
- Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0)
- Built with `scikit-learn`, `shap`, `streamlit`, `xgboost`, `pandas`
