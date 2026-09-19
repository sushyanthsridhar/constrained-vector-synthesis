import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

y_train_lstm = model.predict(X_train, verbose=0).flatten()
y_test_lstm = model.predict(X_test, verbose=0).flatten()

residual_train = y_train - y_train_lstm
residual_test = y_test - y_test_lstm


def build_residual_features(X_seq, week_col_idx, y_lstm_pred):
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    week = X_seq[:, -1, week_col_idx].reshape(-1, 1)
    y_lstm_pred = np.asarray(y_lstm_pred).reshape(-1, 1)
    return np.hstack([X_flat, week, y_lstm_pred])


week_idx = features.index('Week')

X_train_resid = build_residual_features(X_train, week_idx, y_train_lstm)
X_test_resid = build_residual_features(X_test, week_idx, y_test_lstm)
y_train_resid = residual_train


def evaluate_model(fitted_model, X_test_feats, y_test_true, y_pred_lstm):
    y_pred = y_pred_lstm.flatten() + fitted_model.predict(X_test_feats)
    r2 = r2_score(y_test_true, y_pred)
    mse = mean_squared_error(y_test_true, y_pred)
    return r2, mse, fitted_model.get_params()


xgb_params = {
    'n_estimators': randint(100, 501),
    'learning_rate': uniform(0.01, 0.19),
    'max_depth': randint(3, 11),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

rf_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'max_depth': [4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9]
}

N_CONFIGS = 20

xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    xgb_params, n_iter=N_CONFIGS, scoring='r2', cv=4, verbose=0, random_state=42
)
xgb_search.fit(X_train_resid, y_train_resid)

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=N_CONFIGS, scoring='r2', cv=4, verbose=0, random_state=42
)
rf_search.fit(X_train_resid, y_train_resid)

gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params, n_iter=N_CONFIGS, scoring='r2', cv=4, verbose=0, random_state=42
)
gb_search.fit(X_train_resid, y_train_resid)

results = {}
for name, search in [("XGBoost", xgb_search), ("Random Forest", rf_search), ("Gradient Boosting", gb_search)]:
    r2, mse, params = evaluate_model(search.best_estimator_, X_test_resid, y_test, y_test_lstm)
    results[name] = {"r2": r2, "mse": mse, "params": params}
