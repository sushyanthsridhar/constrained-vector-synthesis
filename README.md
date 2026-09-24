# Aedes Early Warning

Code accompanying "Constrained Data Synthesis with Spectral Validation for Vector Surveillance Forecasting: Application to *Aedes aegypti* Early Warning" (submitted to INFORMS Journal on Data Science).

This repository implements the paper's methodological contributions: SARIMAX gap reconstruction with literature-derived trend amplification, GMM-based breeding-intensity classification, a constrained random-walk synthetic data generator, Discrete Fourier Transform (DFT) spectral validation, KNN environmental matching and feature-panel assembly for synthetic traps, the pretrained environmental context autoencoder, and a two-stage LSTM+XGBoost hybrid forecasting model with out-of-fold residual generation and four development-stage validation trials. Raw C&oacute;rdoba ovitrap records are not included; see **Data availability** below.

## Core contributions implemented here

- **`src/reconstruct_sarimax_gap.py`** — Fits a SARIMAX model (no exogenous regressors) to the spatially-averaged weekly mean egg count for 2009-2013, forecasts the missing 2014-2022 years, then applies the literature-derived compounding trend-amplification rate (~0.09%/year) clipped to the empirical 90% confidence interval, and outputs the temporal template and empirical week-to-week differences used by the synthesis step.
- **`src/classify_breeding_intensity.py`** — Fits the two Gaussian Mixture Models (weekly counts and seasonal totals) to pooled 2009-2013 records and classifies each week into a mild/moderate/severe breeding-intensity phase.
- **`src/build_synthesis_parameters.py`** — Combines the SARIMAX template with both fitted GMMs into the single parameter file `synthesize_egg_count_series.py` consumes.
- **`src/synthesize_egg_count_series.py`** — Generates 2,000 synthetic 28-week ovitrap series via the constrained random walk bounded by the temporal trend template, volatility constraint, and magnitude constraint described in the paper.
- **`src/validate_spectral_fidelity.py`** — Computes and compares the Fourier amplitude spectra of real and synthetic series to verify seasonal periodicity is preserved.
- **`src/extract_landsat_indices.py`** — Computes NDVI, NDBI, and NDWI at trap coordinates from LANDSAT-7 surface reflectance bands, for both the real and synthetic trap networks.
- **`src/match_synthetic_traps_knn.py`** — Matches each synthetic trap coordinate to its nearest real 2009-2013 trap in normalized NDVI/NDBI/NDWI space.
- **`src/build_feature_panel.py`** — Assembles the 23-raw-feature trap-week panel, its lag features (including the zero-filled Count_lag52 for the 31 test traps), and the (8, 23) sequence tensors.
- **`src/train_environmental_autoencoder.py`** — Trains the frozen autoencoder that produces the 4-D environmental context embedding concatenated onto the feature panel.
- **`src/forecast_hybrid_model.py`** — The two-stage Bidirectional LSTM + XGBoost residual-correction forecaster, with out-of-fold LSTM residual generation for the XGBoost training target and the four development-stage validation trials over the real breeding seasons.
- **`src/tune_residual_models.py`** — Randomized hyperparameter search tuning XGBoost, Random Forest, and Gradient Boosting as residual correctors, using an identical feature set across all three and the same out-of-fold residual generation as the main forecaster.
- **`src/evaluate_naive_baselines.py`** — Computes the persistence and climatology baselines on the held-out 2023-2024 test set, using the pooled-MSE convention described below.

## Pipeline

| Order | Script | Reads | Writes |
|---|---|---|---|
| 1 | `src/extract_landsat_indices.py` | trap coordinate CSVs (real and synthetic); LANDSAT-7 band GeoTIFFs (B2, B3, B4, B5) | `real_trap_coordinates_with_indices.csv` and `coordinates_with_indices.csv` (NDVI, NDBI, NDWI per coordinate) |
| 2 | `src/reconstruct_sarimax_gap.py` | pooled weekly egg counts for 2009-2013 (`weekly_egg_counts_2009_2013.csv`) | `sarimax_gap_reconstruction.csv` (the amplified 2014-2022 weekly reconstruction) and `sarimax_template_parameters.csv` (the 2022 temporal template and the empirical week-to-week differences) |
| 3 | `src/classify_breeding_intensity.py` | pooled 2009-2013 weekly egg counts (CSV) and per-trap seasonal totals (Excel) | fitted GMM objects and a phase label (mild/moderate/severe) per week, held in memory |
| 4 | `src/build_synthesis_parameters.py` | `sarimax_template_parameters.csv` from step 2; refits the GMMs from step 3 | `synthesis_parameters.xlsx` |
| 5 | `src/synthesize_egg_count_series.py` | `synthesis_parameters.xlsx` | `synthetic_series.csv` (2,000 synthetic 28-week series) and `synthetic_vs_empirical.png` |
| 6 | `src/validate_spectral_fidelity.py` | `real_vs_synthetic_series.xlsx` (see **Upstream artifact contracts**) | per-year Fourier coefficients and the amplitude discrepancy (Error_k) at the dominant frequencies, held in memory only (not written to disk) |
| 7 | `src/match_synthetic_traps_knn.py` | `real_trap_coordinates_with_indices.csv` and `coordinates_with_indices.csv` from step 1 | `synthetic_trap_environmental_matches.csv` |
| 8 | `src/build_feature_panel.py` | weekly egg counts (real and synthetic), NASA POWER climatic variables, the environmental indices from step 1, and the KNN matches from step 7 | `feature_panel.csv` (the 23-raw-feature trap-week panel) |
| 9 | `src/train_environmental_autoencoder.py` | `feature_panel.csv` | `environmental_context_embedding.csv` (the 4-D embedding per trap-week) |
| 10 | `src/forecast_hybrid_model.py` | `feature_panel.csv` and `environmental_context_embedding.csv`, assembled into (32, 8, 27) sequence tensors for the synthetic series, the real 2009-2013 seasons, and the held-out 2023-2024 test set | a trained LSTM, a trained XGBoost residual corrector, and four development-stage validation trial metrics |
| 11 | `src/tune_residual_models.py` | the same feature panel as step 10, plus LSTM predictions from an already-trained model | tuned Random Forest, Gradient Boosting, and XGBoost residual correctors and their comparison metrics |
| 12 | `src/evaluate_naive_baselines.py` | `weekly_egg_counts_2009_2013.csv` (climatology) and `weekly_egg_counts_2023_2024.csv` (test trap-weeks) | `naive_baseline_metrics.csv` (persistence and climatology error metrics) |

## Upstream artifact contracts

One script still consumes an intermediate artifact whose shape is worth documenting directly:

- **`real_vs_synthetic_series.xlsx`** — paired real/synthetic weekly series, one `...actual`/`...synth` row pair per year, used for the DFT comparison in `src/validate_spectral_fidelity.py`.
- **Persistence baseline** — not a learned model; a naive forecast computed independently per trap. For trap *i* at week *t*, the forecast is the trap's own observed count at week *t*-1: `y_hat[i,t] = y[i,t-1]`. The lag is computed within each trap (no cross-trap carryover), and the week-1 forecast uses that trap's last real observation before the evaluation window, consistent with the within-trap protocol described in Section 3.2 of the paper. Evaluated over the same held-out 2023-2024 trap-weeks as the hybrid model, using the same pooled-MSE convention (errors pooled across all test trap-weeks rather than averaged per trap first).

The construction of the feature panel, lag features, and sequence tensors is specified in Section 2.1 and Section 2.4.4 of the manuscript and in its feature-definition appendix, and implemented in `src/build_feature_panel.py`, `src/match_synthetic_traps_knn.py`, and `src/train_environmental_autoencoder.py`. See [`REPRODUCE.md`](REPRODUCE.md) for the full script-to-section mapping.

## Intermediate and model artifacts

Running the pipeline also produces intermediate CSVs (`coordinates_with_indices.csv`, `synthetic_series.csv`, `feature_panel.csv`, `environmental_context_embedding.csv`) and trained model artifacts (fitted GMM objects, the autoencoder, LSTM weights, the XGBoost residual corrector). These aren't checked into this repository since they depend on the restricted raw surveillance data described in [`DATA_README.md`](DATA_README.md) and are regenerated locally when the pipeline is run.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.9 or later.

## Data

None of the raw input data is included in this repository. Raw input formats, intermediate artifact contracts, the 2009-2013/2023-2024 data split, and data-availability terms are documented in [`DATA_README.md`](DATA_README.md).

## Citation

If you use this code, please cite the paper (citation to be added upon acceptance).
