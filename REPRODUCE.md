# Reproducing the analysis

Companion repository to "Constrained Data Synthesis with Spectral Validation for Vector Surveillance Forecasting".

## What this repository contains

Scripts in `src/` implementing the following stages of the pipeline. Each script has a matching `.txt` description file.

| Stage | Script | Manuscript section |
|---|---|---|
| Gap reconstruction of 2014-2022 (SARIMAX, no exogenous regressors) | `reconstruct_sarimax_gap.py` | Sec. 2.4.1 (`sec:gap_reconstruction`) |
| Literature trend amplification, GMM fitting, constrained random-walk synthesis | `synthesize_egg_count_series.py` | Sec. 2.4.2-2.4.3 (`sec:trend_amplification`, `sec:gmm`) |
| DFT spectral validation (k = 1..4, magnitude only) | `validate_spectral_fidelity.py` | Sec. 2.4.5 (`sec:fourier_validation`) |
| Land-cover indices from Landsat-7 ETM+ | `extract_landsat_indices.py` | Sec. 2.1 (`sec:data_description`) |
| Breeding-intensity classification | `classify_breeding_intensity.py` | Sec. 2.1 |
| Two-stage BiLSTM + XGBoost forecaster | `forecast_hybrid_model.py` | Sec. 2.6 (`sec:hybrid_model`) |
| Residual-model hyperparameter search | `tune_residual_models.py` | Sec. 2.6.3 (`sec:tuning`) |
| Persistence and climatology baselines | `evaluate_naive_baselines.py` | Sec. 3.3 (`sec:forecasting_performance`), Appendix B |

## Suggested execution order

1. `reconstruct_sarimax_gap.py`
2. `synthesize_egg_count_series.py`
3. `validate_spectral_fidelity.py` (results are held in memory only)
4. `extract_landsat_indices.py`, `classify_breeding_intensity.py`
5. `forecast_hybrid_model.py`
6. `tune_residual_models.py`
7. `evaluate_naive_baselines.py`

Install dependencies with `pip install -r requirements.txt`.

`tune_residual_models.py` is the residual-tuning procedure rather than a standalone entry point. It operates on an already-trained LSTM and an assembled feature panel, which it expects to be defined in the calling session. `forecast_hybrid_model.py` provides the model, training and cross-validation functions, which are called on the assembled feature panel.

Seeding is partial. `classify_breeding_intensity.py`, `reconstruct_sarimax_gap.py` and the XGBoost and inner-fold splits in `forecast_hybrid_model.py` are seeded. `synthesize_egg_count_series.py` is not seeded, so each run draws a different set of 2,000 series; the distributional constraints in Algorithm 1 of the manuscript, not a fixed seed, are what make the batch reproducible in the sense the paper claims. LSTM training is additionally subject to TensorFlow non-determinism.

## Feature panel, lag features and sequence tensors

The construction of the per-trap feature panel (23 raw predictors, lag features including the zero-filled `Count_lag52`, the 4-dimensional autoencoder spatial embedding, and the (32, 8, 27) sequence tensors) is specified in the manuscript, Section 2.1 (`sec:data_description`) and Section 2.4.4 (`sec:spatial_embedding`), and Appendix "Feature Definitions" (`sec:app_features`). `forecast_hybrid_model.py` consumes a panel built to that specification.

## Tables and figures

Tables and figures in the manuscript are compiled from the outputs of the stages above together with the manuscript's own description of each analysis. The `tables/` and `figures/` folders hold the exported results that appear in the paper; see the README in each folder.

## Data

The raw Córdoba ovitrap records are provided under a data-sharing agreement and cannot be redistributed. See `DATA_README.md`. NASA POWER variables and Landsat-7 ETM+ imagery are public.

## Scope note

This repository reflects the state of the code at the time of submission. A more detailed written description of the panel, lag and sequence construction is available from the corresponding author on request.
