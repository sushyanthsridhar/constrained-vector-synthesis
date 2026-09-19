# tables/

All 14 tables from the manuscript (main text and appendices) as static reference CSVs. Numbering follows the compiled manuscript.

| File | Paper table | Contents |
|---|---|---|
| `table1_literature_comparison.csv` | Table 1 | Literature comparison across methodological characteristics |
| `table2_feature_set.csv` | Table 2 | Feature set for the hybrid forecasting model |
| `table3_lstm_config.csv` | Table 3 | LSTM architecture and training configuration |
| `table4_xgb_input_features.csv` | Table 4 | Input features for the XGBoost residual model |
| `table5_performance_comparison.csv` | Table 5 | Held-out test performance of the LSTM and hybrid model, with and without synthetic augmentation |
| `table6_fold_level_results.csv` | Table 6 | Per-trial results for the standalone LSTM and hybrid model |
| `table7_hybrid_comparison_optimized.csv` | Table 7 | Hybrid model versus baseline comparison |
| `table8_sensitivity_analysis.csv` | Table 8 | Augmentation design parameter sensitivity analysis |
| `table9_notation.csv` | Table 9 (Appendix A) | Notation |
| `table10_performance_metrics.csv` | Table 10 (Appendix B) | Forecast performance metrics and skill scores |
| `table11_performance_comparison_full.csv` | Table 11 (Appendix B) | Full comparison including Random Forest, Gradient Boosting, persistence and climatology |
| `table12_lstm_features.csv` | Table 12 (Appendix C) | The 23 LSTM predictor definitions |
| `table13_rf_gb_config.csv` | Table 13 (Appendix E) | Randomized-search spaces for the RF and GB comparison models (20 configurations each) |
| `table14_avg_correlation.csv` | Table 14 (Appendix F) | Average seasonal correlation between environmental indices and egg counts |

**Note on provenance:** these are static reference copies of the manuscript's tables. Where the scripts in `src/` compute the underlying numbers, they print or hold them in memory and do not write them in this format. See `REPRODUCE.md` for the script-to-section mapping.
