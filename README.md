# Aedes Early Warning

Code accompanying "Constrained Data Synthesis with Spectral Validation for Vector Surveillance Forecasting: Application to *Aedes aegypti* Early Warning" (submitted to INFORMS Journal on Data Science).

This repository implements the paper's methodological contributions: SARIMAX gap reconstruction with literature-derived trend amplification, GMM-based breeding-intensity classification, a constrained random-walk synthetic data generator, Discrete Fourier Transform (DFT) spectral validation, and a two-stage LSTM+XGBoost hybrid forecasting model with out-of-fold residual generation and four-fold cross-validation. Raw C&oacute;rdoba ovitrap records are not included; see **Data availability** below.

## Core contributions implemented here

- **`src/reconstruct_sarimax_gap.py`** — Fits a SARIMAX model (no exogenous regressors) to the spatially-averaged weekly mean egg count for 2009-2013, forecasts the missing 2014-2022 years, then applies the literature-derived compounding trend-amplification rate (~0.09%/year) clipped to the empirical 90% confidence interval, and outputs the temporal template and empirical week-to-week differences used by the synthesis step.
- **`src/classify_breeding_intensity.py`** — Fits the two Gaussian Mixture Models (weekly counts and seasonal totals) to pooled 2009-2013 records and classifies each week into a mild/moderate/severe breeding-intensity phase.
- **`src/synthesize_egg_count_series.py`** — Generates 2,000 synthetic 28-week ovitrap series via the constrained random walk bounded by the temporal trend template, volatility constraint, and magnitude constraint described in the paper.
- **`src/validate_spectral_fidelity.py`** — Computes and compares the Fourier amplitude spectra of real and synthetic series to verify seasonal periodicity is preserved.
- **`src/forecast_hybrid_model.py`** — The two-stage Bidirectional LSTM + XGBoost residual-correction forecaster, with out-of-fold LSTM residual generation for the XGBoost training target and the four-fold cross-validation routine over the real breeding seasons.
- **`src/tune_residual_models.py`** — Randomized hyperparameter search tuning XGBoost, Random Forest, and Gradient Boosting as residual correctors, using an identical feature set across all three.
- **`src/evaluate_naive_baselines.py`** — Computes the persistence and climatology baselines on the held-out 2023-2024 test set, using the pooled-MSE convention described below.
- **`src/extract_landsat_indices.py`** — Computes NDVI, NDBI, and NDWI at sampled trap coordinates from LANDSAT-7 surface reflectance bands.

## Pipeline

| Order | Script | Reads | Writes |
|---|---|---|---|
| 1 | `src/extract_landsat_indices.py` | `trap_coordinates.csv`; LANDSAT-7 band GeoTIFFs (B2, B3, B4, B5) | `coordinates_with_indices.csv` (adds NDVI, NDBI, NDWI per coordinate) |
| 2 | `src/reconstruct_sarimax_gap.py` | pooled weekly egg counts for 2009-2013 (`weekly_egg_counts_2009_2013.csv`) | `sarimax_gap_reconstruction.csv` (the amplified 2014-2022 weekly reconstruction) and `sarimax_template_parameters.csv` (the 2022 temporal template and the empirical week-to-week differences, both consumed by `synthesis_parameters.xlsx`) |
| 3 | `src/classify_breeding_intensity.py` | pooled 2009-2013 weekly egg counts (CSV) and per-trap seasonal totals (Excel) | fitted GMM objects and a phase label (mild/moderate/severe) per week, held in memory |
| 4 | `src/synthesize_egg_count_series.py` | `synthesis_parameters.xlsx` (see **Upstream artifact contracts**) and the GMM parameters from step 3 | `synthetic_series.csv` (2,000 synthetic 28-week series) and `synthetic_vs_empirical.png` |
| 5 | `src/validate_spectral_fidelity.py` | `real_vs_synthetic_series.xlsx` (see **Upstream artifact contracts**) | per-year Fourier coefficients and the amplitude discrepancy (Error_k) at the dominant frequencies, held in memory only (not written to disk) |
| 6 | `src/forecast_hybrid_model.py` | the assembled feature panel (see **Upstream artifact contracts**) for the synthetic series, the real 2009-2013 seasons, and the held-out 2023-2024 test set | a trained LSTM, a trained XGBoost residual corrector, and four-fold cross-validation metrics |
| 7 | `src/tune_residual_models.py` | the same feature panel as step 6, plus LSTM predictions from an already-trained model | tuned Random Forest, Gradient Boosting, and XGBoost residual correctors and their comparison metrics |
| 8 | `src/evaluate_naive_baselines.py` | `weekly_egg_counts_2009_2013.csv` (climatology) and `weekly_egg_counts_2023_2024.csv` (test trap-weeks) | `naive_baseline_metrics.csv` (persistence and climatology error metrics) |

## Upstream artifact contracts

Three of the scripts above (`src/synthesize_egg_count_series.py`, `src/validate_spectral_fidelity.py`, and `src/forecast_hybrid_model.py`) consume intermediate artifacts rather than raw data directly, so their exact expected shape is documented here:

- **`synthesis_parameters.xlsx`** — the SARIMAX-reconstructed weekly template (28 weeks) with the literature-derived trend amplification already applied, plus its 90% empirical confidence bounds per week. Assembled from the outputs of `src/reconstruct_sarimax_gap.py` (`sarimax_template_parameters.csv`, containing the amplified 2022 weekly template and the empirical week-to-week differences) together with the weekly-count GMM intensity anchors from `src/classify_breeding_intensity.py`.
- **`real_vs_synthetic_series.xlsx`** — paired real/synthetic weekly series, one `...actual`/`...synth` row pair per year, used for the DFT comparison.
- **The 8-week, 23-raw-feature panel** consumed by `src/forecast_hybrid_model.py` and `src/tune_residual_models.py` — one row per trap-week, columns for the 23 predictors in Table 2 of the paper (environmental indices, climatic variables, temporal features, and autoregressive lags including Count_lag52), assembled per trap location via the KNN environmental-similarity match described in Section 2.1 for synthetic traps.
- **Persistence baseline** — not a learned model; a naive forecast computed independently per trap. For trap *i* at week *t*, the forecast is the trap's own observed count at week *t*-1: `y_hat[i,t] = y[i,t-1]`. The lag is computed within each trap (no cross-trap carryover), and the week-1 forecast uses that trap's last real observation before the evaluation window, consistent with the within-trap protocol described in Section 3.2 of the paper. Evaluated over the same held-out 2023-2024 trap-weeks as the hybrid model, using the same pooled-MSE convention (errors pooled across all test trap-weeks rather than averaged per trap first).

The construction of the feature panel, lag features and sequence tensors from the raw environmental and surveillance sources is specified in Section 2.1 and Section 2.4.4 of the manuscript and in its feature-definition appendix. See [`REPRODUCE.md`](REPRODUCE.md) for the script-to-section mapping.

## Intermediate and model artifacts

Running the pipeline also produces intermediate CSVs (`coordinates_with_indices.csv`, `synthetic_series.csv`) and trained model artifacts (fitted GMM objects, LSTM weights, the XGBoost residual corrector). These aren't checked into this repository since they depend on the restricted raw surveillance data described in [`DATA_README.md`](DATA_README.md) and are regenerated locally when the pipeline is run.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.9 or later.

## Data

None of the raw input data is included in this repository. Raw input formats, intermediate artifact contracts, the 2009-2013/2023-2024 data split, and data-availability terms are documented in [`DATA_README.md`](DATA_README.md).

## Citation

If you use this code, please cite the paper (citation to be added upon acceptance).
