import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

WEEKLY_COUNTS_FILE = "weekly_egg_counts_2009_2013.csv"
OUTPUT_GAP_YEARS_FILE = "sarimax_gap_reconstruction.csv"
OUTPUT_PARAMS_FILE = "sarimax_template_parameters.csv"

YEARS_REAL = ('2009', '2010', '2011', '2012', '2013')
N_WEEKS = 28
N_GAP_YEARS = 9

CENTURY_PCT_INCREASE = 0.095
ANNUAL_RATE = (1 + CENTURY_PCT_INCREASE) ** (1 / 100) - 1

CONFIDENCE_LEVEL = 0.90

ORDER_GRID = range(0, 3)
SEASONAL_ORDER_GRID = range(0, 2)
SEASONAL_PERIOD = N_WEEKS


def load_pooled_weekly_means(filename, years=YEARS_REAL, n_weeks=N_WEEKS):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df[['Year', 'Week', 'Egg Count']]
    df['Year'] = df['Year'].astype(str)
    df['Week'] = df['Week'].astype(int)
    df = df[df['Year'].isin(years)]
    pivot = df.groupby(['Year', 'Week'])['Egg Count'].mean().unstack('Week')
    pivot = pivot.reindex(columns=range(1, n_weeks + 1))
    pivot = pivot.reindex(sorted(years, key=str))
    return pivot


def empirical_weekly_stats(pivot):
    weekly_mean = pivot.mean(axis=0).values.astype(float)
    weekly_std = pivot.std(axis=0, ddof=1).values.astype(float)
    return weekly_mean, weekly_std


def empirical_ci_band(weekly_mean, weekly_std, confidence_level=CONFIDENCE_LEVEL):
    z = stats.norm.ppf((1 + confidence_level) / 2)
    lower = np.maximum(0, weekly_mean - z * weekly_std)
    upper = weekly_mean + z * weekly_std
    return lower, upper


def select_sarimax_order(series, seasonal_period=SEASONAL_PERIOD,
                          order_grid=ORDER_GRID, seasonal_grid=SEASONAL_ORDER_GRID):
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_fit = None
    for p, q in itertools.product(order_grid, order_grid):
        for P, Q in itertools.product(seasonal_grid, seasonal_grid):
            order = (p, 1, q)
            seasonal_order = (P, 1, Q, seasonal_period)
            try:
                fit = SARIMAX(
                    series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            except Exception:
                continue
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
                best_seasonal_order = seasonal_order
                best_fit = fit
    return best_fit, best_order, best_seasonal_order


def fit_sarimax_gap_model(pivot, seasonal_period=SEASONAL_PERIOD):
    series = pivot.values.flatten().astype(float)
    series = series[~np.isnan(series)]
    return select_sarimax_order(series, seasonal_period)


def forecast_gap_years(fit, n_gap_years=N_GAP_YEARS, n_weeks=N_WEEKS):
    horizon = n_gap_years * n_weeks
    forecast = np.asarray(fit.get_forecast(steps=horizon).predicted_mean)
    forecast = np.maximum(0, forecast)
    return forecast.reshape(n_gap_years, n_weeks)


def apply_trend_amplification(gap_years, weekly_mean, weekly_std,
                               annual_rate=ANNUAL_RATE, confidence_level=CONFIDENCE_LEVEL):
    lower, upper = empirical_ci_band(weekly_mean, weekly_std, confidence_level)
    amplified = np.zeros_like(gap_years)
    for i in range(gap_years.shape[0]):
        year_index = i + 1
        factor = (1 + annual_rate) ** year_index
        candidate = gap_years[i] * factor
        ceiling = np.maximum(upper, lower)
        amplified[i] = np.clip(candidate, lower, ceiling)
    return amplified


def build_synthesis_template(amplified_gap_years):
    return amplified_gap_years[-1]


def build_synthesis_parameters(weekly_mean, amplified_gap_years):
    weekly_differences = np.diff(weekly_mean)
    template_egg_trap_week = build_synthesis_template(amplified_gap_years)
    return weekly_differences, template_egg_trap_week


if __name__ == "__main__":
    pivot = load_pooled_weekly_means(WEEKLY_COUNTS_FILE)
    weekly_mean, weekly_std = empirical_weekly_stats(pivot)

    fit, order, seasonal_order = fit_sarimax_gap_model(pivot)
    gap_years = forecast_gap_years(fit)
    amplified_gap_years = apply_trend_amplification(gap_years, weekly_mean, weekly_std)
    weekly_differences, template_egg_trap_week = build_synthesis_parameters(
        weekly_mean, amplified_gap_years
    )

    pd.DataFrame(
        amplified_gap_years,
        index=[str(2013 + i) for i in range(1, N_GAP_YEARS + 1)],
        columns=[f"Week {w}" for w in range(1, N_WEEKS + 1)],
    ).to_csv(OUTPUT_GAP_YEARS_FILE)

    pd.DataFrame(
        [weekly_differences, template_egg_trap_week],
        index=["weekly_differences", "template_egg_trap_week"],
    ).to_csv(OUTPUT_PARAMS_FILE)
