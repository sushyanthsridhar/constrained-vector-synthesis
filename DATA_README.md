# Data

This file documents the raw and intermediate data the pipeline expects. No raw data is checked into this repository (see **Data availability** below); this is a specification of what to supply, not a description of files present here.

## Raw inputs

To run the full pipeline you will need to supply, in the formats each script expects:

- Pooled weekly ovitrap egg counts and per-trap seasonal totals for 2009-2013.
- LANDSAT-7 ETM+ scene GeoTIFFs covering Cordoba, and CSVs of candidate trap coordinates for both the real trap network and the 2,000 synthetic coordinates (`trap_coordinates.csv`). Confirm the specific scene(s) used before running `src/extract_landsat_indices.py`.
- NASA POWER climatic variables (temperature, humidity, precipitation) for the trap-weeks in the forecasting feature panel. The SARIMAX gap reconstruction in `src/reconstruct_sarimax_gap.py` uses no exogenous regressors.
- Weekly egg counts for the 2023-2024 test traps (`weekly_egg_counts_2023_2024.csv`), used by `src/evaluate_naive_baselines.py`, and a list of those 31 test trap IDs, used by `src/build_feature_panel.py` to zero-fill the 52-week lag features.

## Intermediate artifacts

Most intermediate artifacts are now produced by a script rather than assembled by hand; the pipeline table in [`README.md`](README.md) lists what each script reads and writes. One artifact is still worth documenting directly, since only one script consumes it:

- **`real_vs_synthetic_series.xlsx`** - paired real/synthetic weekly series, one `...actual`/`...synth` row pair per year, used for the DFT comparison. `src/validate_spectral_fidelity.py` keeps its results in memory only.

The 23-raw-feature trap-week panel (Table 2 of the paper), the KNN environmental-similarity match anchoring each synthetic trap to a real 2009-2013 trap, and the 4-D environmental context embedding are specified in Section 2.1 and Section 2.4.4 of the manuscript and implemented in `src/build_feature_panel.py`, `src/match_synthetic_traps_knn.py`, and `src/train_environmental_autoencoder.py` respectively.

## Outputs not checked in

Running the pipeline also produces intermediate CSVs (`coordinates_with_indices.csv`, `synthetic_series.csv`, `feature_panel.csv`, `environmental_context_embedding.csv`) and trained model artifacts (fitted GMM objects, the autoencoder, LSTM weights, the XGBoost residual corrector). These aren't checked into this repository since they depend on the restricted raw surveillance data described above, and are regenerated locally when the pipeline is run.

## Data split

Real ovitrap records span 2009-2013 and 2023-2024. The 2009-2013 seasons are used for GMM fitting, synthetic data generation, and the four-fold cross-validation training/validation rotation. The 2023-2024 season is held out entirely - it is not used in augmentation, GMM fitting, or any training or validation fold - and serves only as the held-out test set against which every reported metric in the paper is computed.

## Data availability

The raw Cordoba ovitrap surveillance records are subject to a data-sharing agreement with the original public health authorities and cannot be redistributed here. Researchers seeking access should contact the corresponding author.
