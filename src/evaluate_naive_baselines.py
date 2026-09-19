import numpy as np
import pandas as pd

TRAIN_WEEKLY_COUNTS_FILE = "weekly_egg_counts_2009_2013.csv"
TEST_WEEKLY_COUNTS_FILE = "weekly_egg_counts_2023_2024.csv"
OUTPUT_METRICS_FILE = "naive_baseline_metrics.csv"

TRAIN_YEARS = ('2009', '2010', '2011', '2012', '2013')
N_WEEKS = 28


def load_test_trap_weeks(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df['Week'] = df['Week'].astype(int)
    df = df.sort_values(['Trap', 'Week']).reset_index(drop=True)
    return df


def compute_climatology_weekly_mean(filename, years=TRAIN_YEARS, n_weeks=N_WEEKS):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df['Year'] = df['Year'].astype(str)
    df['Week'] = df['Week'].astype(int)
    df = df[df['Year'].isin(years)]
    weekly_mean = df.groupby('Week')['Egg Count'].mean()
    weekly_mean = weekly_mean.reindex(range(1, n_weeks + 1))
    return weekly_mean


def persistence_errors(test_df, n_weeks=N_WEEKS):
    errors = []
    for _, trap_group in test_df.groupby('Trap'):
        trap_group = trap_group.sort_values('Week')
        counts = trap_group['Egg Count'].values
        weeks = trap_group['Week'].values
        if len(counts) < 2:
            continue
        forecasts = counts[:-1]
        actuals = counts[1:]
        scored = (weeks[1:] >= 1) & (weeks[1:] <= n_weeks)
        errors.append((actuals - forecasts)[scored])
    return np.concatenate(errors)


def climatology_errors(test_df, weekly_mean, n_weeks=N_WEEKS):
    scored_df = test_df[(test_df['Week'] >= 1) & (test_df['Week'] <= n_weeks)]
    predicted = scored_df['Week'].map(weekly_mean)
    return scored_df['Egg Count'].values - predicted.values


def pooled_mse(errors):
    return float(np.mean(np.square(errors)))


def compute_skill_score(model_mse, baseline_mse):
    return 1 - (model_mse / baseline_mse)


if __name__ == "__main__":
    test_df = load_test_trap_weeks(TEST_WEEKLY_COUNTS_FILE)
    weekly_mean = compute_climatology_weekly_mean(TRAIN_WEEKLY_COUNTS_FILE)

    pers_errors = persistence_errors(test_df)
    clim_errors = climatology_errors(test_df, weekly_mean)

    persistence_mse = pooled_mse(pers_errors)
    climatology_mse = pooled_mse(clim_errors)

    pd.DataFrame(
        {
            "baseline": ["persistence", "climatology"],
            "mse": [persistence_mse, climatology_mse],
            "n_trap_weeks": [len(pers_errors), len(clim_errors)],
        }
    ).to_csv(OUTPUT_METRICS_FILE, index=False)
