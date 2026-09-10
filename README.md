# Aedes Early Warning

Code accompanying "Constrained Data Synthesis with Spectral Validation for Vector Surveillance Forecasting: Application to *Aedes aegypti* Early Warning" (submitted to INFORMS Journal on Data Science).

This repository implements GMM-based breeding-intensity classification, a constrained random-walk synthetic data generator, Discrete Fourier Transform (DFT) spectral validation, and a two-stage LSTM+XGBoost hybrid forecasting model with four-fold cross-validation, as described in the paper. Raw C&oacute;rdoba ovitrap records are not included; see **Data availability** below.

## Pipeline

| Order | Script | Reads | Writes |
|---|---|---|---|
| 1 | `extract_landsat_indices.py` | `trap_coordinates.csv`; LANDSAT-7 band GeoTIFFs (B2, B3, B4, B5) | `coordinates_with_indices.csv` (adds NDVI, NDBI, NDWI per coordinate) |
| 2 | `classify_breeding_intensity.py` | pooled 2009-2013 weekly egg counts (CSV) and per-trap seasonal totals (Excel) | fitted GMM objects and a phase label (mild/moderate/severe) per week, held in memory |
| 3 | `synthesize_egg_count_series.py` | `synthesis_parameters.xlsx` — the SARIMAX-reconstructed weekly template, its empirical confidence bounds, and the GMM parameters from step 2 | `synthetic_series.csv` (2,000 synthetic 28-week series) and `synthetic_vs_empirical.png` |
| 4 | `validate_spectral_fidelity.py` | `real_vs_synthetic_series.xlsx` — paired real/synthetic weekly series, one `...actual`/`...synth` row pair per year | per-year Fourier coefficients and the amplitude discrepancy (Error_k) at the dominant frequencies, held in memory |
| 5 | `forecast_hybrid_model.py` | the assembled feature panel (23 raw features per week, 8-week windows) for the synthetic series, the real 2009-2013 seasons, and the held-out 2023-2024 test set | a trained LSTM, a trained XGBoost residual corrector, and four-fold cross-validation metrics |
| 6 | `tune_residual_models.py` | the same feature panel as step 5, plus LSTM predictions from an already-trained model | tuned Random Forest, Gradient Boosting, and XGBoost residual correctors and their comparison metrics |

## Data split

Real ovitrap records span 2009-2013 and 2023-2024. The 2009-2013 seasons are used for GMM fitting, synthetic data generation, and the four-fold cross-validation training/validation rotation. The 2023-2024 season is held out entirely — it is not used in augmentation, GMM fitting, or any training or validation fold — and serves only as the final test set against which every reported metric in the paper is computed.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.9 or later.

## Expected input data

None of the raw input data is included in this repository (see **Data availability**). To run these scripts you will need to supply, in the formats each script expects:

- Pooled weekly ovitrap egg counts and per-trap seasonal totals for 2009-2013.
- LANDSAT-7 ETM+ scene GeoTIFFs covering C&oacute;rdoba, and a CSV of candidate trap coordinates (`trap_coordinates.csv`).
- NASA POWER climatic variables (temperature, humidity, precipitation) for the same period.
- The SARIMAX-reconstructed weekly template and its confidence bounds, assembled into `synthesis_parameters.xlsx` (this file is produced by the SARIMAX reconstruction step, which is not part of this release — see note below).

## Files

- **`extract_landsat_indices.py`** — Computes NDVI, NDBI, and NDWI at sampled trap coordinates from LANDSAT-7 surface reflectance bands.
- **`classify_breeding_intensity.py`** — Fits two Gaussian Mixture Models to pooled 2009-2013 records (weekly counts and seasonal totals) and classifies each week into a mild/moderate/severe breeding-intensity phase.
- **`synthesize_egg_count_series.py`** — Generates 2,000 synthetic 28-week ovitrap series via a constrained random walk bounded by a temporal trend template, a volatility constraint, and a magnitude constraint.
- **`validate_spectral_fidelity.py`** — Compares the Fourier amplitude spectra of real and synthetic series to confirm seasonal periodicity is preserved.
- **`forecast_hybrid_model.py`** — The two-stage Bidirectional LSTM + XGBoost residual-correction forecaster, including the four-fold cross-validation routine over the real breeding seasons.
- **`tune_residual_models.py`** — Randomized hyperparameter search tuning XGBoost, Random Forest, and Gradient Boosting as residual correctors, using an identical feature set across all three.

Each script has a matching `.txt` file with a short, paper-accurate description.

## Data availability

The raw C&oacute;rdoba ovitrap surveillance records are subject to a data-sharing agreement with the original public health authorities and cannot be redistributed here. Researchers seeking access should contact the corresponding author.

## Honest reproducibility note

This code reproduces the methods described in the paper, but running it end-to-end as a pipeline requires filling gaps this release does not cover:

- **The SARIMAX reconstruction and literature-derived trend amplification step is not included in this release.** `synthesize_egg_count_series.py` consumes its output (the weekly template and confidence bounds in `synthesis_parameters.xlsx`) but does not produce it.
- **`tune_residual_models.py` expects an already-trained LSTM model and its predictions in scope** (`model`, `X_train`, `y_train`, `X_test`, `y_test`, `features`); it is meant to be run after `forecast_hybrid_model.py` in the same session or notebook, not standalone.
- **`extract_landsat_indices.py` LANDSAT scene filenames are placeholders** pointing at an October 2023 scene; confirm and replace with the scene(s) actually used before running.
- The K-Nearest-Neighbors environmental-similarity matching step, which the paper describes as anchoring each synthetic trap location to a real 2009-2013 record by NDVI/NDBI/NDWI similarity, is not included in this release.
- **The feature-panel assembly step is not included.** `forecast_hybrid_model.py` and `tune_residual_models.py` both expect an already-built 8-week, 23-raw-feature panel as input; the script that assembles NDVI/NDBI/NDWI, NASA POWER climatic variables, and the synthetic/real egg counts into that panel is not part of this release.

## Citation

If you use this code, please cite the paper (citation to be added upon acceptance).
