import numpy as np
import pandas as pd
import pytest

from coffeepred import lineage
from coffeepred.models import (FEATURES, conformal_intervals, conformal_width, fit_conformal, group_bootstrap_r2,
                               out_of_fold, permutation_importance_grouped, score, splits)
from coffeepred.validate import ATTRIBUTES, Rule, validate

DATA = "data/coffee_ratings.csv"


@pytest.fixture(scope="module")
def clean():
    return validate(pd.read_csv(DATA))


def _synthetic(n=600, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    countries = rng.choice(["A", "B", "C", "D"], n)
    alt = rng.uniform(800, 2200, n)
    df = pd.DataFrame({
        "country": countries, "process": rng.choice(["Washed", "Natural"], n),
        "variety": rng.choice(["V1", "V2"], n), "species": "Arabica", "color": "Green",
        "altitude_m": alt, "moisture": rng.uniform(0.09, 0.12, n),
        "category_one_defects": rng.integers(0, 3, n), "category_two_defects": rng.integers(0, 5, n),
        "quakers": 0, "group": rng.integers(0, 120, n),
        "graded_on": pd.Timestamp("2014-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
    })
    df["lot_id"] = [f"l{i}" for i in range(n)]
    df["y"] = (alt - 1500) / 300 * (1 if signal else 0) + rng.normal(0, 1, n)
    return df


def test_validation_matches_known_problems(clean):
    df, report = clean
    r = report.set_index("rule").rows_failing
    assert r["empty_scores"] == 1 and r["altitude_implausible"] == 35 and r["moisture_zero"] == 264
    assert len(df) == 1338
    assert df.altitude_m.dropna().between(100, 3000).all() and not (df.moisture == 0).any()
    assert (df.total_cup_points - df[ATTRIBUTES].sum(axis=1)).abs().max() <= 0.5
    assert df.lot_id.is_unique


def test_rule_actions():
    raw = pd.read_csv(DATA).head(20).copy()
    raw.loc[0, "moisture"] = 0.5
    rules = [Rule("drop_first", "", lambda d: d.index == d.index[0], "drop"),
             Rule("blank_moisture", "", lambda d: d.moisture > 0.3, "set_missing:moisture")]
    df, report = validate(raw, rules)
    assert len(df) == 19 and report.rows_after.tolist() == [19, 19]


def test_manifest_is_traceable(tmp_path):
    m = lineage.manifest(DATA, {"seed": 0}, [{"rule": "x"}])
    assert m["data_sha256"] == lineage.sha256(DATA) and len(m["data_sha256"]) == 64
    assert {"packages", "config", "cleaning_steps", "python"} <= set(m)
    lineage.write(m, tmp_path / "m.json")
    assert (tmp_path / "m.json").read_text().startswith("{")


def test_grouped_and_time_splits_do_not_leak():
    df = _synthetic()
    for tr, te in splits(df, "grouped"):
        assert not set(df.group.iloc[tr]) & set(df.group.iloc[te])
    (tr, te), = list(splits(df, "time"))
    assert df.graded_on.iloc[tr].max() < pd.Timestamp("2016-01-01") <= df.graded_on.iloc[te].min()


def test_models_learn_signal_and_not_noise():
    sig = _synthetic()
    assert score(sig.y, out_of_fold(sig, "y", "ridge", "grouped"))["r2"] > 0.4
    noise = _synthetic(signal=False, seed=1)
    assert score(noise.y, out_of_fold(noise, "y", "ridge", "grouped"))["r2"] < 0.05


def test_conformal_coverage_close_to_nominal():
    df = _synthetic(n=1500, seed=2)
    out = conformal_intervals(df, "y", "ridge", alpha=0.2)
    assert 0.74 < out.covered.mean() < 0.86
    assert conformal_width(np.arange(1, 101), 0.2) == 81      # the ceil(101 * 0.8) = 81st smallest
    model, half = fit_conformal(df, "y", "ridge", alpha=0.2)
    assert half > 0 and model.predict(df[FEATURES].head(2)).shape == (2,)


def test_bootstrap_interval_and_importance():
    df = _synthetic()
    pred = out_of_fold(df, "y", "ridge", "grouped")
    lo, hi = group_bootstrap_r2(df.y, pred, df.group, n_boot=200)
    assert lo < score(df.y, pred)["r2"] < hi
    imp = permutation_importance_grouped(df, "y", "ridge", n_repeats=2)
    assert imp.index[0] == "altitude"


def test_results_regression_guard():
    b = pd.read_csv("results/benchmark.csv")
    t = b[(b.target == "total_cup_points") & (b.model == "ridge")].set_index("scheme").r2
    assert 0 < t["grouped"] < 0.3 and t["random"] > t["grouped"] > t["time"]
    assert set(FEATURES).isdisjoint({"harvest_year", "lab", "owner"})
