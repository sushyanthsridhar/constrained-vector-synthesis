import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def select_gmm_by_bic(data, min_components=2, max_components=5, random_state=42):
    data = data.reshape(-1, 1)
    bic_scores = []
    models = []
    for n in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
        models.append(gmm)
    best_idx = int(np.argmin(bic_scores))
    return models[best_idx], bic_scores


def fit_weekly_count_gmm(weekly_egg_counts, min_components=2, max_components=5, random_state=42):
    weekly_egg_counts = weekly_egg_counts[~np.isnan(weekly_egg_counts)]
    gmm, bic_scores = select_gmm_by_bic(weekly_egg_counts, min_components, max_components, random_state)
    return gmm, bic_scores


def fit_seasonal_total_gmm(seasonal_totals, n_components=2, random_state=42):
    seasonal_totals = seasonal_totals[~np.isnan(seasonal_totals)]
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(seasonal_totals.reshape(-1, 1))
    return gmm


def assign_breeding_phase(weekly_gmm, weekly_egg_counts):
    valid = ~np.isnan(weekly_egg_counts)
    responsibilities = np.full((len(weekly_egg_counts), weekly_gmm.n_components), np.nan)
    responsibilities[valid] = weekly_gmm.predict_proba(weekly_egg_counts[valid].reshape(-1, 1))

    order = np.argsort(weekly_gmm.means_.flatten())
    phase_names = ['mild', 'moderate', 'severe'][:weekly_gmm.n_components]
    phase_labels = {order[i]: phase_names[i] for i in range(weekly_gmm.n_components)}

    phase_assignment = np.full(len(weekly_egg_counts), None, dtype=object)
    component_assignment = np.argmax(responsibilities[valid], axis=1)
    phase_assignment[valid] = [phase_labels[c] for c in component_assignment]
    return phase_assignment


def load_pooled_weekly_counts(filename, years=('2009', '2010', '2011', '2012', '2013')):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df[['Year', 'Week', 'Egg Count']]
    df['Year'] = df['Year'].astype(str)
    df['Week'] = df['Week'].astype(int)
    df = df[df['Year'].isin(years)]
    return df['Egg Count'].values.astype(float)


def load_pooled_seasonal_totals(filename, years=('2009', '2010', '2011', '2012', '2013')):
    df = pd.read_excel(filename, sheet_name='PerTrap')
    year_cols = [c for c in df.columns if str(c) in years]
    return df[year_cols].values.flatten().astype(float)


if __name__ == "__main__":
    weekly_counts = load_pooled_weekly_counts("weekly_egg_counts_2009_2013.csv")
    seasonal_totals = load_pooled_seasonal_totals("per_trap_seasonal_totals_2009_2013.xlsx")

    weekly_gmm, bic_scores = fit_weekly_count_gmm(weekly_counts)
    seasonal_gmm = fit_seasonal_total_gmm(seasonal_totals)

    phase_by_week = assign_breeding_phase(weekly_gmm, weekly_counts)
