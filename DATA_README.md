# Data

This file documents the raw and intermediate data the pipeline expects. No raw data is checked into this repository (see **Data availability** below); this is a specification of what to supply, not a description of files present here.

## Raw inputs

To run the full pipeline you will need to supply, in the formats each script expects:

- Pooled weekly ovitrap egg counts and per-trap seasonal totals for 2009-2013.
- LANDSAT-7 ETM+ scene GeoTIFFs covering Cordoba, and a CSV of candidate trap coordinates (`trap_coordinates.csv`). Confirm the specific scene(s) used before running `src/extract_landsat_indices.py`.
- NASA POWER climatic variables (temperature, humidity, precipitation) for the same period.

## Intermediate artifacts

Two of the pipeline scripts (`src/synthesize_egg_count_series.py`, `src/validate_spectral_fidelity.py`) and the forecasting scripts (`src/forecast_hybrid_model.py`, `src/tune_residual_models.py`) consume intermediate artifacts assembled from the raw inputs above, rather than raw data directly. Their exact expected shapes:

- **`synthesis_parameters.xlsx`** - the SARIMAX-reconstructed weekly template (28 weeks) with the literature-derived trend amplification already applied, plus its 90% empirical confidence bounds per week. Produced from the SARIMAX gap-reconstruction step described in Section 2.3.1-2.3.2 of the paper.
- **`real_vs_synthetic_series.xlsx`** - paired real/synthetic weekly series, one `...actual`/`...synth` row pair per year, used for the DFT comparison.
- **The 8-week, 23-raw-feature panel** consumed by `src/forecast_hybrid_model.py` and `src/tune_residual_models.py` - one row per trap-week, columns for the 23 predictors in Table 2 of the paper (environmental indices, climatic variables, temporal features, and autoregressive lags including Count_lag52), assembled per trap location via the KNN environmental-similarity match described in Section 2.1 for synthetic traps.

Reconstructing these artifacts from the raw environmental/climatic sources and the GMM/SARIMAX outputs is a data-assembly step specific to the restricted raw surveillance data; the finalized scripts for this step will be included in the tagged release accompanying the accepted manuscript, consistent with the Data and Code Availability statement in the paper.

## Outputs not checked in

Running the pipeline also produces intermediate CSVs (`coordinates_with_indices.csv`, `synthetic_series.csv`) and trained model artifacts (fitted GMM objects, LSTM weights, the XGBoost residual corrector). These aren't checked into this repository since they depend on the restricted raw surveillance data described above, and are regenerated locally when the pipeline is run.

## Data split

Real ovitrap records span 2009-2013 and 2023-2024. The 2009-2013 seasons are used for GMM fitting, synthetic data generation, and the four-fold cross-validation training/validation rotation. The 2023-2024 season is held out entirely - it is not used in augmentation, GMM fitting, or any training or validation fold - and serves only as the final test set against which every reported metric in the paper is computed.

## Data availability

The raw Cordoba ovitrap surveillance records are subject to a data-sharing agreement with the original public health authorities and cannot be redistributed here. Researchers seeking access should contact the corresponding author.
