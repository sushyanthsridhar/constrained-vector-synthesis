# figures/

Static reference copies of the manuscript's figures that exist as external image files. Numbering follows the compiled manuscript.

| File | Paper figure | Caption |
|---|---|---|
| `figure1_data_timeline.png` | Figure 1 | Temporal data partitioning and evaluation protocol across the 2009-2013 and 2023-2024 periods |
| none | Figure 2 | Two-stage LSTM + XGBoost architecture. Drawn directly in the LaTeX source as a `tikzpicture`, so there is no image file in this folder |
| `figure3_ovitrap_comparison.png` | Figure 3 | Comparison of ovitrap distributions across Cordoba, real training traps and synthetic spatially uniform distribution |
| `figure4_synthetic_sampling.png` | Figure 4 | Representative synthetic series within the 90% empirical confidence interval |
| `figure5_fourier_plots.png` | Figure 5 | DFT amplitude spectra of weekly egg counts, empirical benchmark versus synthetic |
| `figure6_severity_classification.png` | Figure 6 | Consolidated frequency of mild, moderate and severe breeding-intensity classifications by week |
| `figure7_prediction_comparison.png` | Figure 7 | Actual egg counts, standalone LSTM predictions and hybrid predictions on the test set |
| `figure8_residual_distribution.jpg` | Figure 8 | Distribution of final prediction residuals for the hybrid model |
| `figure9_correlation_heatmap.png` | Figure 9 (Appendix) | Temporal correlation matrix, environmental indices versus weekly egg counts |
| `figure10_ndwi_scatter.png` | Figure 10 (Appendix) | Relationship between surface water (NDWI) and egg abundance |
| `figure11_trap_deployments.png` | Figure 11 (Appendix) | The two ovitrap deployments, 127 traps in 2009-2013 and 31 traps in 2023-2024 |

**Note on provenance:** these are static reference copies of the manuscript's figures. Figure 4 is written directly by `src/synthesize_egg_count_series.py` as `synthetic_vs_empirical.png`. The remaining figures were produced from the analysis outputs and are not written by the scripts in this form. See `REPRODUCE.md` for the script-to-section mapping.
