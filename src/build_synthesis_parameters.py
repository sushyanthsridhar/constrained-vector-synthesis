import numpy as np
import pandas as pd

from classify_breeding_intensity import (
    fit_weekly_count_gmm,
    fit_seasonal_total_gmm,
    load_pooled_weekly_counts,
    load_pooled_seasonal_totals,
)

WEEKLY_COUNTS_FILE = "weekly_egg_counts_2009_2013.csv"
SEASONAL_TOTALS_FILE = "per_trap_seasonal_totals_2009_2013.xlsx"
SARIMAX_PARAMS_FILE = "sarimax_template_parameters.csv"
OUTPUT_FILE = "synthesis_parameters.xlsx"
OUTPUT_SHEET = "SynthesisParams_2009_2013"


def load_sarimax_template(filename):
    df = pd.read_csv(filename, index_col=0)
    weekly_differences = df.loc["weekly_differences"].dropna().values.astype(float)
    template_egg_trap_week = df.loc["template_egg_trap_week"].dropna().values.astype(float)
    return weekly_differences, template_egg_trap_week


def build_parameter_rows(weekly_gmm, seasonal_gmm, weekly_differences, template_egg_trap_week):
    weekly_means = weekly_gmm.means_.flatten()
    weekly_variances = weekly_gmm.covariances_.flatten()
    weekly_weights = weekly_gmm.weights_.flatten()

    total_eggs_means = seasonal_gmm.means_.flatten()
    total_eggs_variances = seasonal_gmm.covariances_.flatten()
    total_eggs_weights = seasonal_gmm.weights_.flatten()

    rows = [
        weekly_means,
        weekly_variances,
        weekly_weights,
        total_eggs_means,
        total_eggs_variances,
        total_eggs_weights,
        weekly_differences,
        template_egg_trap_week,
    ]
    max_len = max(len(row) for row in rows)
    padded = [np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in rows]
    return pd.DataFrame(padded)


if __name__ == "__main__":
    weekly_counts = load_pooled_weekly_counts(WEEKLY_COUNTS_FILE)
    seasonal_totals = load_pooled_seasonal_totals(SEASONAL_TOTALS_FILE)

    weekly_gmm, _ = fit_weekly_count_gmm(weekly_counts)
    seasonal_gmm = fit_seasonal_total_gmm(seasonal_totals)

    weekly_differences, template_egg_trap_week = load_sarimax_template(SARIMAX_PARAMS_FILE)

    df_params = build_parameter_rows(
        weekly_gmm, seasonal_gmm, weekly_differences, template_egg_trap_week
    )

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        df_params.to_excel(writer, sheet_name=OUTPUT_SHEET, header=False, index=False)
