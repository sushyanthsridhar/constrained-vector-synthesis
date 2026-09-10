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

df_params = pd.read_excel(PARAMS_FILE, sheet_name=PARAMS_SHEET, header=None)

weekly_means = df_params.iloc[0, :3].dropna().values.astype(float)
weekly_variances = df_params.iloc[1, :3].dropna().values.astype(float)
weekly_weights = df_params.iloc[2, :3].dropna().values.astype(float)

total_eggs_means = df_params.iloc[3, :2].dropna().values.astype(float)
total_eggs_variances = df_params.iloc[4, :2].dropna().values.astype(float)
total_eggs_weights = df_params.iloc[5, :2].dropna().values.astype(float)

weekly_differences = df_params.iloc[6, :N_WEEKS - 1].dropna().values.astype(float)
template_egg_trap_week = df_params.iloc[7, :N_WEEKS].dropna().values.astype(float)

z_score = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)

std_diff = np.std(weekly_differences, ddof=1)
ci_lower_diff = weekly_differences - z_score * std_diff
ci_upper_diff = weekly_differences + z_score * std_diff
weekly_diff_ci = np.vstack((ci_lower_diff, ci_upper_diff)).T

std_egg_week = np.std(template_egg_trap_week, ddof=1)
ci_lower_egg_week = np.maximum(0, template_egg_trap_week - z_score * std_egg_week)
ci_upper_egg_week = template_egg_trap_week + z_score * std_egg_week

intensity_anchors = np.sort(weekly_means)
anchor_sds = np.sqrt(weekly_variances[np.argsort(weekly_means)])
anchor_floor = max(0.0, intensity_anchors[0] - z_score * anchor_sds[0])
anchor_ceiling = intensity_anchors[-1] + z_score * anchor_sds[-1]

trap_categories = np.random.choice(len(total_eggs_means), size=N_TRAPS, p=total_eggs_weights)
total_eggs_per_trap = np.random.normal(total_eggs_means[trap_categories],
                                       np.sqrt(total_eggs_variances[trap_categories]))
total_eggs_per_trap = np.maximum(total_eggs_per_trap, 0)

template_total = np.sum(template_egg_trap_week)
trap_scales = total_eggs_per_trap / template_total

synthetic_data = np.zeros((N_TRAPS, N_WEEKS))

for i in range(N_TRAPS):
    scale = trap_scales[i]
    trap_template = template_egg_trap_week * scale
    trap_ci_lower = np.maximum(0, ci_lower_egg_week * scale)
    trap_ci_upper = ci_upper_egg_week * scale
    trap_diff_ci = weekly_diff_ci * scale

    trap_floor = np.maximum(trap_ci_lower, anchor_floor * scale)
    trap_ceiling = np.minimum(trap_ci_upper, anchor_ceiling * scale)
    trap_ceiling = np.maximum(trap_ceiling, trap_floor)

    weekly_counts = np.zeros(N_WEEKS)
    weekly_counts[0] = np.clip(
        np.random.normal(trap_template[0], std_egg_week * scale),
        trap_floor[0], trap_ceiling[0]
    )

    for t in range(1, N_WEEKS):
        expected_change = trap_template[t] - trap_template[t - 1]
        deviation = np.random.normal(expected_change, std_diff * scale)

        lower_bound, upper_bound = trap_diff_ci[t - 1]
        deviation = np.clip(deviation, lower_bound, upper_bound)

        weekly_counts[t] = np.clip(weekly_counts[t - 1] + deviation,
                                   trap_floor[t], trap_ceiling[t])

    synthetic_data[i, :] = weekly_counts

df_synthetic = pd.DataFrame(synthetic_data, columns=[f"Week {i+1}" for i in range(N_WEEKS)])
df_synthetic.insert(0, "Trap ID", [f"Trap_{i+1}" for i in range(N_TRAPS)])
df_synthetic.to_csv(OUTPUT_FILE, index=False)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.plot(range(1, N_WEEKS + 1), df_synthetic.iloc[i, 1:].values,
             marker='o', linestyle='-', label=f"Synthetic Trap {i+1}")
plt.plot(range(1, N_WEEKS + 1), template_egg_trap_week, marker='s', linestyle='--',
         color='black', label="Empirical 2009-2013: Egg/Trap/Week")
plt.fill_between(range(1, N_WEEKS + 1), ci_lower_egg_week, ci_upper_egg_week,
                 color='gray', alpha=0.2, label="Empirical 2009-2013: 90% CI")
plt.xlabel("Week")
plt.ylabel("Egg Count")
plt.title("Synthetic Traps vs. Empirical 2009-2013 Template")
plt.legend()
plt.grid(True)
plt.savefig("synthetic_vs_empirical.png", dpi=150, bbox_inches="tight")
