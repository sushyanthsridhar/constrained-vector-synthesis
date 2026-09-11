# tables/

Numerical results tables corresponding to the paper, for example:

- Table 5 (final-test performance comparison) and Table 6 (per-trial hybrid model results) — from `src/forecast_hybrid_model.py`
- Table 7 (hybrid model vs. baseline comparison) — from `src/forecast_hybrid_model.py` and `src/tune_residual_models.py`
- Table 8 (DFT amplitude discrepancy, Error_k, by frequency) — from `src/validate_spectral_fidelity.py`

Nothing is checked into this folder — these are regenerated locally when the pipeline is run against the raw surveillance data (see the top-level `DATA_README`).
