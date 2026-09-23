import pandas as pd
from sklearn.neighbors import NearestNeighbors

REAL_TRAP_INDICES_FILE = "real_trap_coordinates_with_indices.csv"
SYNTHETIC_TRAP_INDICES_FILE = "coordinates_with_indices.csv"
OUTPUT_FILE = "synthetic_trap_environmental_matches.csv"

MATCH_COLUMNS = ["NDVI", "NDBI", "NDWI"]


def normalize_columns(df, columns=MATCH_COLUMNS, reference_stats=None):
    if reference_stats is None:
        reference_stats = {col: (df[col].mean(), df[col].std(ddof=1)) for col in columns}
    normalized = df.copy()
    for col in columns:
        mean, std = reference_stats[col]
        normalized[col] = (df[col] - mean) / std
    return normalized, reference_stats


def match_synthetic_to_real_traps(real_df, synthetic_df, columns=MATCH_COLUMNS):
    real_normalized, reference_stats = normalize_columns(real_df, columns)
    synthetic_normalized, _ = normalize_columns(synthetic_df, columns, reference_stats)

    real_features = real_normalized[columns].values
    synthetic_features = synthetic_normalized[columns].values

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(real_features)
    distances, indices = knn.kneighbors(synthetic_features)

    matches = pd.DataFrame({
        "Synthetic_Trap_ID": synthetic_df["Trap_ID"].values,
        "Matched_Real_Trap_ID": real_df["Trap_ID"].values[indices.flatten()],
        "Environmental_Distance": distances.flatten(),
    })
    return matches


if __name__ == "__main__":
    real_df = pd.read_csv(REAL_TRAP_INDICES_FILE)
    synthetic_df = pd.read_csv(SYNTHETIC_TRAP_INDICES_FILE)

    matches = match_synthetic_to_real_traps(real_df, synthetic_df)
    matches.to_csv(OUTPUT_FILE, index=False)
