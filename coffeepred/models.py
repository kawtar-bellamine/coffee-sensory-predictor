"""Features, models and evaluation.

The question: from what is known about a green coffee before anyone tastes it (origin,
process, variety, altitude, moisture, defects), how well can its cup profile be
predicted? Every model is a scikit-learn pipeline, so preprocessing is fitted inside each
training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["country", "process", "variety", "species", "color"]
NUMERIC = ["altitude_m", "moisture", "category_one_defects", "category_two_defects", "quakers"]
FEATURES = CATEGORICAL + NUMERIC
# Harvest year is left out by design: a future lot always has a year outside the training range,
# so the model would extrapolate. experiments/run_all.py reports the effect of adding it back.
FEATURE_GROUPS = {"origin": ["country"], "process": ["process"], "variety and species": ["variety", "species"],
                  "altitude": ["altitude_m"], "bean quality (moisture, defects, quakers, colour)":
                      ["moisture", "category_one_defects", "category_two_defects", "quakers", "color"]}
MODEL_NAMES = {"mean": "Mean of training lots", "ridge": "Ridge regression",
               "random_forest": "Random forest", "lightgbm": "Gradient boosting (LightGBM)"}


def _preprocess(scale: bool, categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    num = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        num.append(StandardScaler())
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
         categorical),
        ("num", make_pipeline(*num), numeric),
    ])


def make_model(name: str, seed: int = 0, categorical=CATEGORICAL, numeric=NUMERIC):
    if name == "mean":
        return DummyRegressor()
    if name == "ridge":
        return make_pipeline(_preprocess(True, categorical, numeric), Ridge(alpha=10.0))
    if name == "random_forest":
        return make_pipeline(_preprocess(False, categorical, numeric),
                             RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                   random_state=seed, n_jobs=-1))
    if name == "lightgbm":
        return make_pipeline(_preprocess(False, categorical, numeric),
                             LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.8, random_state=seed, verbose=-1))
    raise ValueError(name)


def splits(df: pd.DataFrame, scheme: str, n_splits: int = 5, seed: int = 0):
    """Train/test index pairs.

    grouped  5 folds, all lots of one owner in the same fold (the honest default);
    random   5 folds ignoring owners (lots of one farm on both sides: optimistic);
    time     graded before 2016 for training, 2016 to 2018 for testing.
    """
    idx = np.arange(len(df))
    if scheme == "grouped":
        yield from GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, groups=df.group)
    elif scheme == "random":
        yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx)
    elif scheme == "time":
        known = df.graded_on.notna().values
        train = idx[known & (df.graded_on < "2016-01-01").values]
        test = idx[known & (df.graded_on >= "2016-01-01").values]
        yield train, test
    else:
        raise ValueError(scheme)


def out_of_fold(df: pd.DataFrame, target: str, model: str, scheme: str, seed: int = 0,
                features: list[str] = FEATURES, **kw) -> pd.Series:
    """Predictions for every lot from a model that never saw it (NaN where not tested)."""
    pred = pd.Series(np.nan, index=df.index)
    for tr, te in splits(df, scheme, seed=seed):
        m = make_model(model, seed, **kw)
        m.fit(df.iloc[tr][features], df.iloc[tr][target])
        pred.iloc[te] = m.predict(df.iloc[te][features])
    return pred


def score(y: pd.Series, pred: pd.Series) -> dict:
    ok = pred.notna()
    return {"r2": r2_score(y[ok], pred[ok]), "mae": mean_absolute_error(y[ok], pred[ok]), "n": int(ok.sum())}


def group_bootstrap_r2(y: pd.Series, pred: pd.Series, groups: pd.Series, n_boot: int = 1000,
                       seed: int = 0) -> tuple[float, float]:
    """95% interval of R^2, resampling whole owners."""
    rng = np.random.default_rng(seed)
    ok = pred.notna()
    y, pred, groups = y[ok].values, pred[ok].values, groups[ok].values
    uniq = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([pos[g] for g in rng.choice(uniq, len(uniq))])
        vals.append(r2_score(y[take], pred[take]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def conformal_intervals(df: pd.DataFrame, target: str, model: str, alpha: float = 0.1, seed: int = 0,
                        features: list[str] = FEATURES) -> pd.DataFrame:
    """Split-conformal prediction intervals inside grouped cross-validation.

    In each outer fold, the training owners are split again: 75% fit the model, 25% calibrate the
    interval half-width as the (1 - alpha) quantile of absolute errors. The test fold then tells
    whether the promised coverage holds on new owners.
    """
    rows = []
    for tr, te in splits(df, "grouped", seed=seed):
        train = df.iloc[tr]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        fit_i, cal_i = next(inner.split(train, groups=train.group))
        m = make_model(model, seed)
        m.fit(train.iloc[fit_i][features], train.iloc[fit_i][target])
        resid = np.abs(train.iloc[cal_i][target].values - m.predict(train.iloc[cal_i][features]))
        q = conformal_width(resid, alpha)
        p = m.predict(df.iloc[te][features])
        rows.append(pd.DataFrame({"lot_id": df.iloc[te].lot_id.values, "pred": p, "low": p - q, "high": p + q,
                                  "actual": df.iloc[te][target].values, "fold": len(rows)}))
    out = pd.concat(rows, ignore_index=True)
    out["covered"] = (out.actual >= out.low) & (out.actual <= out.high)
    return out


def conformal_width(abs_residuals: np.ndarray, alpha: float) -> float:
    """Half-width of a split-conformal interval: the ceil((n + 1)(1 - alpha))-th smallest residual."""
    r = np.sort(np.asarray(abs_residuals))
    k = int(np.ceil((len(r) + 1) * (1 - alpha)))
    return float(r[min(k, len(r)) - 1])


def fit_conformal(df: pd.DataFrame, target: str, model: str, alpha: float = 0.2, seed: int = 0,
                  features: list[str] = FEATURES):
    """One split-conformal pass on all lots: 75% of owners fit the model, 25% calibrate the width."""
    fit_i, cal_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(df, groups=df.group))
    m = make_model(model, seed)
    m.fit(df.iloc[fit_i][features], df.iloc[fit_i][target])
    resid = np.abs(df.iloc[cal_i][target].values - m.predict(df.iloc[cal_i][features]))
    return m, conformal_width(resid, alpha)


def permutation_importance_grouped(df: pd.DataFrame, target: str, model: str, seed: int = 0,
                                   n_repeats: int = 5, groups: dict = FEATURE_GROUPS) -> pd.Series:
    """Drop in R^2 on the test folds when one feature group is shuffled (grouped CV)."""
    rng = np.random.default_rng(seed)
    drops = {g: [] for g in groups}
    for tr, te in splits(df, "grouped", seed=seed):
        m = make_model(model, seed)
        m.fit(df.iloc[tr][FEATURES], df.iloc[tr][target])
        test = df.iloc[te][FEATURES]
        y = df.iloc[te][target]
        base = r2_score(y, m.predict(test))
        for g, cols in groups.items():
            for _ in range(n_repeats):
                shuffled = test.copy()
                perm = rng.permutation(len(test))
                for c in cols:
                    shuffled[c] = test[c].values[perm]
                drops[g].append(base - r2_score(y, m.predict(shuffled)))
    return pd.Series({g: float(np.mean(v)) for g, v in drops.items()}).sort_values(ascending=False)
