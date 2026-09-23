import pandas as pd

FEATURE_COLUMNS = [
    "NDVI", "NDBI", "NDWI",
    "Year", "Week", "Month",
    "Count_lag1", "Count_lag2", "Count_lag4", "Count_lag52",
    "Temperature", "Temperature_lag2", "Temperature_lag4", "Temperature_lag52",
    "Humidity", "Humidity_lag2", "Humidity_lag4", "Humidity_lag52",
    "Rainfall_roll3", "Rainfall_lag2", "Rainfall_lag4", "Rainfall_lag6", "Rainfall_lag52",
]

LAG52_COLUMNS = ["Count_lag52", "Temperature_lag52", "Humidity_lag52", "Rainfall_lag52"]

N_TIMESTEPS = 8
RAINFALL_ROLL_WINDOW = 3

WEEKLY_COUNTS_FILE = "weekly_egg_counts.csv"
CLIMATIC_FILE = "nasa_power_weekly.csv"
ENVIRONMENTAL_FILE = "coordinates_with_indices.csv"
TEST_TRAP_IDS_FILE = "test_trap_ids_2023_2024.csv"
OUTPUT_PANEL_FILE = "feature_panel.csv"


def merge_trap_week_sources(weekly_counts, climatic, environmental):
    panel = weekly_counts.merge(climatic, on=["Trap_ID", "Year", "Week"], how="left")
    panel = panel.merge(
        environmental[["Trap_ID", "NDVI", "NDBI", "NDWI"]], on="Trap_ID", how="left"
    )
    return panel


def add_calendar_features(panel):
    panel = panel.sort_values(["Trap_ID", "Year", "Week"]).reset_index(drop=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["Month"] = panel["Date"].dt.month
    return panel


def add_rolling_rainfall(panel, window=RAINFALL_ROLL_WINDOW):
    panel["Rainfall_roll3"] = (
        panel.groupby("Trap_ID")["Rainfall"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return panel


def add_lag_feature(panel, source_col, lag, new_col):
    panel[new_col] = panel.groupby("Trap_ID")[source_col].shift(lag)
    return panel


def build_autoregressive_features(panel):
    panel = add_lag_feature(panel, "Egg_Count", 1, "Count_lag1")
    panel = add_lag_feature(panel, "Egg_Count", 2, "Count_lag2")
    panel = add_lag_feature(panel, "Egg_Count", 4, "Count_lag4")
    panel = add_lag_feature(panel, "Egg_Count", 52, "Count_lag52")

    panel = add_lag_feature(panel, "Temperature", 2, "Temperature_lag2")
    panel = add_lag_feature(panel, "Temperature", 4, "Temperature_lag4")
    panel = add_lag_feature(panel, "Temperature", 52, "Temperature_lag52")

    panel = add_lag_feature(panel, "Humidity", 2, "Humidity_lag2")
    panel = add_lag_feature(panel, "Humidity", 4, "Humidity_lag4")
    panel = add_lag_feature(panel, "Humidity", 52, "Humidity_lag52")

    panel = add_lag_feature(panel, "Rainfall_roll3", 2, "Rainfall_lag2")
    panel = add_lag_feature(panel, "Rainfall_roll3", 4, "Rainfall_lag4")
    panel = add_lag_feature(panel, "Rainfall_roll3", 6, "Rainfall_lag6")
    panel = add_lag_feature(panel, "Rainfall_roll3", 52, "Rainfall_lag52")

    return panel


def zero_fill_lag52_for_new_deployments(panel, test_trap_ids, columns=LAG52_COLUMNS):
    mask = panel["Trap_ID"].isin(test_trap_ids)
    for col in columns:
        panel.loc[mask, col] = panel.loc[mask, col].fillna(0.0)
    return panel


def build_sequence_tensor(panel, trap_id, origin_week, n_timesteps=N_TIMESTEPS,
                           feature_columns=FEATURE_COLUMNS):
    trap_panel = (
        panel[panel["Trap_ID"] == trap_id]
        .sort_values(["Year", "Week"])
        .reset_index(drop=True)
    )
    origin_idx = trap_panel.index[trap_panel["Week"] == origin_week][0]
    start_idx = origin_idx - n_timesteps + 1
    window = trap_panel.iloc[start_idx: origin_idx + 1]
    return window[feature_columns].values.astype(float)


if __name__ == "__main__":
    weekly_counts = pd.read_csv(WEEKLY_COUNTS_FILE)
    climatic = pd.read_csv(CLIMATIC_FILE)
    environmental = pd.read_csv(ENVIRONMENTAL_FILE)
    test_trap_ids = pd.read_csv(TEST_TRAP_IDS_FILE)["Trap_ID"].tolist()

    panel = merge_trap_week_sources(weekly_counts, climatic, environmental)
    panel = add_calendar_features(panel)
    panel = add_rolling_rainfall(panel)
    panel = build_autoregressive_features(panel)
    panel = zero_fill_lag52_for_new_deployments(panel, test_trap_ids)

    panel.to_csv(OUTPUT_PANEL_FILE, index=False)
