import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

N_TIMESTEPS = 8
N_RAW_FEATURES = 23
N_FEATURES = 27
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.0005
N_FOLDS = 4

# Inner folds used only to generate out-of-fold LSTM predictions for the XGBoost
# residual stage (see compute_oof_lstm_predictions / train_hybrid_model below).
N_INNER_FOLDS = 5
INNER_SEED = 42

XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.02
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.7
XGB_COLSAMPLE_BYTREE = 0.8


def build_lstm_model(n_timesteps=N_TIMESTEPS, n_features=N_FEATURES, learning_rate=LEARNING_RATE):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(n_timesteps, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(1, activation='linear'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val,
                      batch_size=BATCH_SIZE, epochs=EPOCHS):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-6
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[reduce_lr],
        verbose=0,
    )
    return model


def build_xgb_model():
    return XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        random_state=42,
    )


def build_xgb_features(X_seq_raw, week_idx, y_lstm_pred):
    X_flat = X_seq_raw.reshape(X_seq_raw.shape[0], -1)
    week = X_seq_raw[:, -1, week_idx].reshape(-1, 1)
    y_lstm_pred = np.asarray(y_lstm_pred).reshape(-1, 1)
    return np.hstack([X_flat, week, y_lstm_pred])


def compute_oof_lstm_predictions(X_full, y, X_val_full=None, y_val=None,
                                  n_splits=N_INNER_FOLDS, seed=INNER_SEED,
                                  batch_size=BATCH_SIZE, epochs=EPOCHS):
    """Generate out-of-fold LSTM predictions on the training set.

    The original implementation trained one LSTM on X_full and then predicted on
    that same X_full to build residuals for the XGBoost stage, in-sample
    predictions are optimistically accurate, so those residuals understated the
    LSTM's real error and gave XGBoost an unrealistically easy target.

    Here, X_full is split into n_splits inner folds; for each inner fold, a
    freshly trained LSTM predicts only on the slice it did NOT train on. The
    concatenated predictions are therefore honest, held-out-style estimates of
    what the LSTM would produce on data it hasn't seen, which is what the
    XGBoost stage should be learning to correct.

    This trains n_splits extra LSTMs per outer fold purely to build residuals;
    the final LSTM used for actual validation/test predictions is still trained
    once on the full X_full/y in train_hybrid_model, this function does not
    replace that model.
    """
    oof_pred = np.zeros(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for inner_train_idx, inner_held_idx in kf.split(X_full):
        inner_model = build_lstm_model()
        # Prefer the outer validation fold for the LR schedule if one is given;
        # otherwise monitor on the inner held-out slice itself.
        monitor_full = X_val_full if X_val_full is not None else X_full[inner_held_idx]
        monitor_y = y_val if y_val is not None else y[inner_held_idx]
        inner_model = train_lstm_model(
            inner_model, X_full[inner_train_idx], y[inner_train_idx],
            monitor_full, monitor_y, batch_size=batch_size, epochs=epochs
        )
        oof_pred[inner_held_idx] = inner_model.predict(
            X_full[inner_held_idx], verbose=0
        ).flatten()

    return oof_pred


def train_hybrid_model(X_train_full, X_train_raw, y_train, X_val_full, X_val_raw, y_val,
                        week_idx, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        n_inner_folds=N_INNER_FOLDS, seed=INNER_SEED):
    # Final LSTM, trained once on the full training set. This is the model used
    # for every downstream prediction (validation fold, test set), unchanged
    # from the original implementation.
    lstm_model = build_lstm_model()
    lstm_model = train_lstm_model(lstm_model, X_train_full, y_train, X_val_full, y_val,
                                   batch_size=batch_size, epochs=epochs)

    # Out-of-fold predictions on the training set, used ONLY to build residuals
    # for the XGBoost stage, see compute_oof_lstm_predictions for why this
    # replaces the original in-sample residual_train = y_train - y_train_lstm.
    y_train_oof = compute_oof_lstm_predictions(
        X_train_full, y_train, X_val_full, y_val,
        n_splits=n_inner_folds, seed=seed,
        batch_size=batch_size, epochs=epochs
    )
    residual_train = y_train - y_train_oof

    X_train_xgb = build_xgb_features(X_train_raw, week_idx, y_train_oof)

    xgb_model = build_xgb_model()
    xgb_model.fit(X_train_xgb, residual_train)

    return lstm_model, xgb_model


def predict_hybrid(lstm_model, xgb_model, X_full, X_raw, week_idx):
    y_lstm = lstm_model.predict(X_full, verbose=0).flatten()
    X_xgb = build_xgb_features(X_raw, week_idx, y_lstm)
    r_hat = xgb_model.predict(X_xgb)
    return y_lstm + r_hat


def run_four_fold_cv(X_synth_full, X_synth_raw, y_synth,
                      real_seasons_full, real_seasons_raw, real_seasons_y,
                      X_test_full, X_test_raw, y_test,
                      week_idx, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      n_inner_folds=N_INNER_FOLDS, seed=INNER_SEED):
    n_folds = len(real_seasons_full)
    fold_results = []

    for fold_idx in range(n_folds):
        val_full = real_seasons_full[fold_idx]
        val_raw = real_seasons_raw[fold_idx]
        val_y = real_seasons_y[fold_idx]

        train_full = np.concatenate(
            [X_synth_full] + [real_seasons_full[i] for i in range(n_folds) if i != fold_idx], axis=0
        )
        train_raw = np.concatenate(
            [X_synth_raw] + [real_seasons_raw[i] for i in range(n_folds) if i != fold_idx], axis=0
        )
        train_y = np.concatenate(
            [y_synth] + [real_seasons_y[i] for i in range(n_folds) if i != fold_idx], axis=0
        )

        lstm_model, xgb_model = train_hybrid_model(
            train_full, train_raw, train_y, val_full, val_raw, val_y,
            week_idx, batch_size=batch_size, epochs=epochs,
            n_inner_folds=n_inner_folds, seed=seed
        )

        y_val_lstm = lstm_model.predict(val_full, verbose=0).flatten()
        y_val_hybrid = predict_hybrid(lstm_model, xgb_model, val_full, val_raw, week_idx)

        y_test_lstm = lstm_model.predict(X_test_full, verbose=0).flatten()
        y_test_hybrid = predict_hybrid(lstm_model, xgb_model, X_test_full, X_test_raw, week_idx)

        fold_results.append({
            "fold": fold_idx + 1,
            "lstm_model": lstm_model,
            "xgb_model": xgb_model,
            "val_lstm_r2": r2_score(val_y, y_val_lstm),
            "val_lstm_mse": mean_squared_error(val_y, y_val_lstm),
            "val_hybrid_r2": r2_score(val_y, y_val_hybrid),
            "val_hybrid_mse": mean_squared_error(val_y, y_val_hybrid),
            "test_lstm_r2": r2_score(y_test, y_test_lstm),
            "test_lstm_mse": mean_squared_error(y_test, y_test_lstm),
            "test_hybrid_r2": r2_score(y_test, y_test_hybrid),
            "test_hybrid_mse": mean_squared_error(y_test, y_test_hybrid),
        })

    summary = {}
    for metric in ["test_lstm_r2", "test_lstm_mse", "test_hybrid_r2", "test_hybrid_mse"]:
        values = [f[metric] for f in fold_results]
        summary[metric] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}

    return fold_results, summary


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    week_idx = 4

    def make_dummy(n_samples):
        X_full = rng.normal(size=(n_samples, N_TIMESTEPS, N_FEATURES)).astype("float32")
        X_raw = X_full[:, :, :N_RAW_FEATURES]
        y = rng.normal(size=(n_samples,)).astype("float32")
        return X_full, X_raw, y

    X_synth_full, X_synth_raw, y_synth = make_dummy(512)

    real_seasons_full, real_seasons_raw, real_seasons_y = [], [], []
    for _ in range(N_FOLDS):
        f, r, y = make_dummy(64)
        real_seasons_full.append(f)
        real_seasons_raw.append(r)
        real_seasons_y.append(y)

    X_test_full, X_test_raw, y_test = make_dummy(96)

    fold_results, summary = run_four_fold_cv(
        X_synth_full, X_synth_raw, y_synth,
        real_seasons_full, real_seasons_raw, real_seasons_y,
        X_test_full, X_test_raw, y_test,
        week_idx, epochs=3, n_inner_folds=3
    )
