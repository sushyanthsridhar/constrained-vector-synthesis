import numpy as np
import pandas as pd
from scipy.fft import fft
from math import pi

INPUT_FILE = "real_vs_synthetic_series.xlsx"
INPUT_SHEET = "Sheet1"
K = 5

# Normalization convention, matches the manuscript exactly (Sec. 2.4,
# sec:fourier_validation): for the DFT F_k = sum_t x[t] exp(-2*pi*i*k*t/N),
#   a_k = (1/N) * sum_t x[t] cos(2*pi*k*t/N) =  Re(F_k) / N
#   b_k = (1/N) * sum_t x[t] sin(2*pi*k*t/N) = -Im(F_k) / N
#   A_k = sqrt(a_k^2 + b_k^2) = |F_k| / N
# This is the single convention used everywhere below, for the printable
# Fourier terms and for the reported amplitude spectrum and Error_k alike.
# (Not the 2/N convention used to reconstruct a real-valued Fourier series;
# that scaling is deliberately not used here, since it would double a_k and
# b_k relative to the manuscript's own definitions above.)


def load_paired_series(input_file=INPUT_FILE, input_sheet=INPUT_SHEET):
    df = pd.read_excel(input_file, sheet_name=input_sheet, header=None)

    labels_cleaned = df.iloc[:, 0].astype(str).str.strip().str.lower()
    data_cleaned = df.iloc[:, 1:].reset_index(drop=True)

    years = sorted(set(label.replace("actual", "").replace("synth", "") for label in labels_cleaned))
    return labels_cleaned, data_cleaned, years


def extract_year_pair(labels_cleaned, data_cleaned, year):
    idx_actual = labels_cleaned[labels_cleaned == f"{year}actual"].index[0]
    idx_synth = labels_cleaned[labels_cleaned == f"{year}synth"].index[0]

    signal_actual = np.nan_to_num(data_cleaned.iloc[idx_actual].values)
    signal_synth = np.nan_to_num(data_cleaned.iloc[idx_synth].values)
    return signal_actual, signal_synth


def compute_fourier_terms(freqs, n, k_max=K):
    """Builds the printable Fourier series terms a_k.cos(w_k.t) + b_k.sin(w_k.t)
    for k = 0..k_max-1, using the |F_k|/N convention throughout."""
    terms = []
    for k in range(k_max):
        a_k = np.real(freqs[k]) / n
        b_k = -np.imag(freqs[k]) / n
        omega_k = 2 * pi * k / n
        if k == 0:
            terms.append(f"{a_k:.3f}")
        else:
            terms.append(f"{a_k:.3f}.cos({omega_k:.3f}.t) + {b_k:.3f}.sin({omega_k:.3f}.t)")
    return terms


def compute_amplitude_error(freqs_actual, freqs_synth, n, k_max=K):
    """Error_k = |A_k^actual - A_k^synth| for k = 1..k_max-1, the dominant
    within-season frequencies, using A_k = |F_k|/N."""
    error_table = []
    total_error = 0
    for k in range(1, k_max):
        A_act = np.abs(freqs_actual[k]) / n
        A_syn = np.abs(freqs_synth[k]) / n
        diff = np.abs(A_act - A_syn)
        total_error += diff
        error_table.append({"k": k, "A_actual": A_act, "A_synth": A_syn, "diff": diff})

    avg_error = total_error / (k_max - 1)
    return error_table, avg_error


def validate_year(signal_actual, signal_synth, k_max=K):
    n = len(signal_actual)
    freqs_actual = fft(signal_actual)
    freqs_synth = fft(signal_synth)

    terms_actual = compute_fourier_terms(freqs_actual, n, k_max)
    terms_synth = compute_fourier_terms(freqs_synth, n, k_max)
    error_table, avg_error = compute_amplitude_error(freqs_actual, freqs_synth, n, k_max)

    return {
        "actual_terms": terms_actual,
        "synth_terms": terms_synth,
        "error_table": error_table,
        "avg_error": avg_error,
    }


if __name__ == "__main__":
    labels_cleaned, data_cleaned, years = load_paired_series()

    results = {}
    for year in years:
        try:
            signal_actual, signal_synth = extract_year_pair(labels_cleaned, data_cleaned, year)
        except IndexError:
            continue

        results[year] = validate_year(signal_actual, signal_synth)
