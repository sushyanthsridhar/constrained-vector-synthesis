import numpy as np
import pandas as pd
from scipy.fft import fft
from math import pi

INPUT_FILE = "real_vs_synthetic_series.xlsx"
INPUT_SHEET = "Sheet1"
K = 5

df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET, header=None)

labels_cleaned = df.iloc[:, 0].astype(str).str.strip().str.lower()
data_cleaned = df.iloc[:, 1:].reset_index(drop=True)

years = sorted(set(label.replace("actual", "").replace("synth", "") for label in labels_cleaned))

results = {}

for year in years:
    try:
        idx_actual = labels_cleaned[labels_cleaned == f"{year}actual"].index[0]
        idx_synth = labels_cleaned[labels_cleaned == f"{year}synth"].index[0]
    except IndexError:
        continue

    signal_actual = np.nan_to_num(data_cleaned.iloc[idx_actual].values)
    signal_synth = np.nan_to_num(data_cleaned.iloc[idx_synth].values)
    N = len(signal_actual)

    freqs_actual = fft(signal_actual)
    freqs_synth = fft(signal_synth)

    terms_actual = []
    for k in range(K):
        a_k = 2 * np.real(freqs_actual[k]) / N
        b_k = -2 * np.imag(freqs_actual[k]) / N
        omega_k = 2 * pi * k / N
        if k == 0:
            terms_actual.append(f"{a_k:.3f}")
        else:
            terms_actual.append(f"{a_k:.3f}.cos({omega_k:.3f}.t) + {b_k:.3f}.sin({omega_k:.3f}.t)")

    terms_synth = []
    for k in range(K):
        a_k = 2 * np.real(freqs_synth[k]) / N
        b_k = -2 * np.imag(freqs_synth[k]) / N
        omega_k = 2 * pi * k / N
        if k == 0:
            terms_synth.append(f"{a_k:.3f}")
        else:
            terms_synth.append(f"{a_k:.3f}.cos({omega_k:.3f}.t) + {b_k:.3f}.sin({omega_k:.3f}.t)")

    error_table = []
    total_error = 0
    for k in range(1, K):
        A_act = np.abs(freqs_actual[k]) / N
        A_syn = np.abs(freqs_synth[k]) / N
        diff = np.abs(A_act - A_syn)
        total_error += diff
        error_table.append({"k": k, "A_actual": A_act, "A_synth": A_syn, "diff": diff})

    avg_error = total_error / (K - 1)

    results[year] = {
        "actual_terms": terms_actual,
        "synth_terms": terms_synth,
        "error_table": error_table,
        "avg_error": avg_error,
    }
