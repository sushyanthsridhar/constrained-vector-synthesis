import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

PARAMS_FILE = "synthesis_parameters.xlsx"
PARAMS_SHEET = "SynthesisParams_2009_2013"
OUTPUT_FILE = "synthetic_series.csv"

N_TRAPS = 2000
N_WEEKS = 28
CONFIDENCE_LEVEL = 0.90


def load_synthesis_parameters(params_file=PARAMS_FILE, params_sheet=PARAMS_SHEET, n_weeks=N_WEEKS):
    df_params = pd.read_excel(params_file, sheet_name=params_sheet, header=None)

    weekly_means = df_params.iloc[0, :3].dropna().values.astype(float)
    weekly_variances = df_params.iloc[1, :3].dropna().values.astype(float)
    weekly_weights = df_params.iloc[2, :3].dropna().values.astype(float)

    total_eggs_means = df_params.iloc[3, :2].dropna().values.astype(float)
    total_eggs_variances = df_params.iloc[4, :2].dropna().values.astype(float)
    total_eggs_weights = df_params.iloc[5, :2].dropna().values.astype(float)

    weekly_differences = df_params.iloc[6, :n_weeks - 1].dropna().values.astype(float)
    template_egg_trap_week = df_params.iloc[7, :n_weeks].dropna().values.astype(float)

    return (
        weekly_means, weekly_variances, weekly_weights,
        total_eggs_means, total_eggs_variances, total_eggs_weights,
        weekly_differences, template_egg_trap_week,
    )


def compute_volatility_band(weekly_differences, z_score):
    """Volatility constraint: 90% CI on the empirical week-to-week differences."""
    std_diff = np.std(weekly_differences, ddof=1)
    ci_lower_diff = weekly_differences - z_score * std_diff
    ci_upper_diff = weekly_differences + z_score * std_diff
    weekly_diff_ci = np.vstack((ci_lower_diff, ci_upper_diff)).T
    return weekly_diff_ci, std_diff


def compute_template_band(template_egg_trap_week, z_score):
    """90% CI band around the amplified SARIMAX template, floored at zero."""
    std_egg_week = np.std(template_egg_trap_week, ddof=1)
    ci_lower_egg_week = np.maximum(0, template_egg_trap_week - z_score * std_egg_week)
    ci_upper_egg_week = template_egg_trap_week + z_score * std_egg_week
    return ci_lower_egg_week, ci_upper_egg_week, std_egg_week


def compute_global_anchor(weekly_means, weekly_variances, z_score):
    """Magnitude constraint: the global floor and ceiling from the lowest and
    highest weekly-count GMM component, sorted by mean. One global range, not
    a week-specific anchor."""
    intensity_anchors = np.sort(weekly_means)
    anchor_sds = np.sqrt(weekly_variances[np.argsort(weekly_means)])
    anchor_floor = max(0.0, intensity_anchors[0] - z_score * anchor_sds[0])
    anchor_ceiling = intensity_anchors[-1] + z_score * anchor_sds[-1]
    return anchor_floor, anchor_ceiling


def draw_seasonal_totals(total_eggs_means, total_eggs_variances, total_eggs_weights, n_traps=N_TRAPS):
    """Draws each trap's total seasonal egg count S from the seasonal-total GMM,
    by first choosing a component according to its weight and then drawing from
    that component's normal distribution, floored at zero."""
    trap_categories = np.random.choice(len(total_eggs_means), size=n_traps, p=total_eggs_weights)
    total_eggs_per_trap = np.random.normal(
        total_eggs_means[trap_categories], np.sqrt(total_eggs_variances[trap_categories])
    )
    return np.maximum(total_eggs_per_trap, 0)


def compute_trap_scales(total_eggs_per_trap, template_egg_trap_week):
    """Per-series scale factor c, the drawn total S divided by the template's
    own 28-week total."""
    template_total = np.sum(template_egg_trap_week)
    return total_eggs_per_trap / template_total


def generate_one_series(scale, template_egg_trap_week, ci_lower_egg_week, ci_upper_egg_week,
                         anchor_floor, anchor_ceiling, weekly_diff_ci, std_egg_week, std_diff,
                         n_weeks=N_WEEKS):
    """Constrained random walk for a single synthetic trap, Algorithm 1: draws
    week 1 from the scaled template, then for each later week draws a change
    centered on the template's own week-to-week change (temporal constraint),
    clips it to the scaled empirical difference interval (volatility
    constraint), and clips the running total to the scaled magnitude band
    (magnitude constraint)."""
    trap_template = template_egg_trap_week * scale
    trap_ci_lower = np.maximum(0, ci_lower_egg_week * scale)
    trap_ci_upper = ci_upper_egg_week * scale
    trap_diff_ci = weekly_diff_ci * scale

    trap_floor = np.maximum(trap_ci_lower, anchor_floor * scale)
    trap_ceiling = np.minimum(trap_ci_upper, anchor_ceiling * scale)
    trap_ceiling = np.maximum(trap_ceiling, trap_floor)

    weekly_counts = np.zeros(n_weeks)
    weekly_counts[0] = np.clip(
        np.random.normal(trap_template[0], std_egg_week * scale),
        trap_floor[0], trap_ceiling[0]
    )

    for t in range(1, n_weeks):
        expected_change = trap_template[t] - trap_template[t - 1]
        deviation = np.random.normal(expected_change, std_diff * scale)

        lower_bound, upper_bound = trap_diff_ci[t - 1]
        deviation = np.clip(deviation, lower_bound, upper_bound)

        weekly_counts[t] = np.clip(weekly_counts[t - 1] + deviation,
                                   trap_floor[t], trap_ceiling[t])

    return weekly_counts


def generate_all_series(trap_scales, template_egg_trap_week, ci_lower_egg_week, ci_upper_egg_week,
                         anchor_floor, anchor_ceiling, weekly_diff_ci, std_egg_week, std_diff,
                         n_traps=N_TRAPS, n_weeks=N_WEEKS):
    """Loops generate_one_series across the batch of synthetic traps."""
    synthetic_data = np.zeros((n_traps, n_weeks))
    for i in range(n_traps):
        synthetic_data[i, :] = generate_one_series(
            trap_scales[i], template_egg_trap_week, ci_lower_egg_week, ci_upper_egg_week,
            anchor_floor, anchor_ceiling, weekly_diff_ci, std_egg_week, std_diff, n_weeks
        )
    return synthetic_data


def save_synthetic_series(synthetic_data, output_file=OUTPUT_FILE):
    n_traps, n_weeks = synthetic_data.shape
    df_synthetic = pd.DataFrame(synthetic_data, columns=[f"Week {i + 1}" for i in range(n_weeks)])
    df_synthetic.insert(0, "Trap ID", [f"Trap_{i + 1}" for i in range(n_traps)])
    df_synthetic.to_csv(output_file, index=False)
    return df_synthetic


def plot_synthetic_vs_empirical(df_synthetic, template_egg_trap_week, ci_lower_egg_week,
                                 ci_upper_egg_week, n_example_traps=5):
    """Reproduces Figure 4, five representative synthetic series against the
    empirical template and its 90% confidence interval."""
    n_weeks = len(template_egg_trap_week)

    plt.figure(figsize=(10, 5))
    for i in range(n_example_traps):
        plt.plot(range(1, n_weeks + 1), df_synthetic.iloc[i, 1:].values,
                 marker='o', linestyle='-', label=f"Synthetic Trap {i + 1}")
    plt.plot(range(1, n_weeks + 1), template_egg_trap_week, marker='s', linestyle='--',
             color='black', label="Empirical 2009-2013: Egg/Trap/Week")
    plt.fill_between(range(1, n_weeks + 1), ci_lower_egg_week, ci_upper_egg_week,
                     color='gray', alpha=0.2, label="Empirical 2009-2013: 90% CI")
    plt.xlabel("Week")
    plt.ylabel("Egg Count")
    plt.title("Synthetic Traps vs. Empirical 2009-2013 Template")
    plt.legend()
    plt.grid(True)
    plt.savefig("synthetic_vs_empirical.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    (
        weekly_means, weekly_variances, weekly_weights,
        total_eggs_means, total_eggs_variances, total_eggs_weights,
        weekly_differences, template_egg_trap_week,
    ) = load_synthesis_parameters()

    z_score = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)

    weekly_diff_ci, std_diff = compute_volatility_band(weekly_differences, z_score)
    ci_lower_egg_week, ci_upper_egg_week, std_egg_week = compute_template_band(
        template_egg_trap_week, z_score
    )
    anchor_floor, anchor_ceiling = compute_global_anchor(weekly_means, weekly_variances, z_score)

    total_eggs_per_trap = draw_seasonal_totals(
        total_eggs_means, total_eggs_variances, total_eggs_weights
    )
    trap_scales = compute_trap_scales(total_eggs_per_trap, template_egg_trap_week)

    synthetic_data = generate_all_series(
        trap_scales, template_egg_trap_week, ci_lower_egg_week, ci_upper_egg_week,
        anchor_floor, anchor_ceiling, weekly_diff_ci, std_egg_week, std_diff
    )

    df_synthetic = save_synthetic_series(synthetic_data)
    plot_synthetic_vs_empirical(df_synthetic, template_egg_trap_week, ci_lower_egg_week, ci_upper_egg_week)
