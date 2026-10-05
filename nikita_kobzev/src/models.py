"""Small, fixed model comparison; no tuning on the final holdout."""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CATEGORICAL, FEATURES


def make_model(name, config, params=None, features=None):
    columns = FEATURES if features is None else features
    cats = [c for c in CATEGORICAL if c in columns]
    if name == "linear_regression":
        preprocessor = ColumnTransformer([
            ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("numeric", StandardScaler(), [c for c in columns if c not in cats])])
        return make_pipeline(preprocessor, LinearRegression())
    if name == "catboost":
        return CatBoostRegressor(**(params or config["catboost_candidates"][0]),
            loss_function="RMSE", random_seed=config["seed"], thread_count=1,
            cat_features=cats, verbose=False, allow_writing_files=False)
    return None


def predict_model(name, model, x):
    if name == "daily_persistence":
        return x.price_prev_day_same_hour.to_numpy(dtype=float)
    if name == "weekly_persistence":
        return x.price_prev_week_same_hour.to_numpy(dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def fit_predict(name, config, train_x, train_y, test_x, params=None):
    model = make_model(name, config, params, list(train_x.columns))
    if model is not None:
        model.fit(train_x, train_y)
    return predict_model(name, model, test_x), model
