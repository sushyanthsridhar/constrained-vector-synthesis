# outputs/

Non-figure, non-table intermediate artifacts produced by running the pipeline, for example:

- `coordinates_with_indices.csv` — from `src/extract_landsat_indices.py`
- `synthetic_series.csv` — the 2,000 synthetic 28-week series from `src/synthesize_egg_count_series.py`
- fitted GMM objects and trained model weights (LSTM, XGBoost) from `src/classify_breeding_intensity.py`, `src/forecast_hybrid_model.py`, and `src/tune_residual_models.py`

Nothing is checked into this folder — these files depend on the restricted raw surveillance data (see the top-level `DATA_README`) and are regenerated locally when the pipeline is run.
