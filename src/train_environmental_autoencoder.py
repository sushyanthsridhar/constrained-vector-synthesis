import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

EMBEDDING_DIM = 4
INPUT_COLUMNS = ["NDVI", "NDBI", "NDWI", "Temperature", "Humidity", "Rainfall_roll3"]
N_INPUT_FEATURES = len(INPUT_COLUMNS)

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100

FEATURE_PANEL_FILE = "feature_panel.csv"
OUTPUT_EMBEDDING_FILE = "environmental_context_embedding.csv"


def build_autoencoder(n_input_features=N_INPUT_FEATURES, embedding_dim=EMBEDDING_DIM,
                       learning_rate=LEARNING_RATE):
    inputs = Input(shape=(n_input_features,))
    encoded = Dense(8, activation='relu')(inputs)
    embedding = Dense(embedding_dim, activation='linear', name='embedding')(encoded)
    decoded = Dense(8, activation='relu')(embedding)
    outputs = Dense(n_input_features, activation='linear')(decoded)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    encoder = Model(inputs, embedding)
    return autoencoder, encoder


def normalize_inputs(df, columns=INPUT_COLUMNS, reference_stats=None):
    if reference_stats is None:
        reference_stats = {col: (df[col].mean(), df[col].std(ddof=1)) for col in columns}
    normalized = np.column_stack([
        (df[col] - reference_stats[col][0]) / reference_stats[col][1] for col in columns
    ])
    return normalized, reference_stats


def train_autoencoder(panel, columns=INPUT_COLUMNS, batch_size=BATCH_SIZE, epochs=EPOCHS):
    X, reference_stats = normalize_inputs(panel, columns)
    autoencoder, encoder = build_autoencoder()
    autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=0)
    return autoencoder, encoder, reference_stats


def compute_embedding(encoder, panel, reference_stats, columns=INPUT_COLUMNS):
    X, _ = normalize_inputs(panel, columns, reference_stats)
    return encoder.predict(X, verbose=0)


if __name__ == "__main__":
    panel = pd.read_csv(FEATURE_PANEL_FILE)

    autoencoder, encoder, reference_stats = train_autoencoder(panel)
    encoder.trainable = False

    embedding = compute_embedding(encoder, panel, reference_stats)

    embedding_df = pd.DataFrame(
        embedding, columns=[f"EnvEmbed_{i + 1}" for i in range(EMBEDDING_DIM)]
    )
    embedding_df.insert(0, "Trap_ID", panel["Trap_ID"].values)
    embedding_df.insert(1, "Year", panel["Year"].values)
    embedding_df.insert(2, "Week", panel["Week"].values)
    embedding_df.to_csv(OUTPUT_EMBEDDING_FILE, index=False)
