"""Run every experiment and write results/ and powerbi/.

    python experiments/run_all.py        (about 2 minutes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coffeepred import lineage  # noqa: E402
from coffeepred.models import (FEATURES, MODEL_NAMES, NUMERIC, conformal_intervals, group_bootstrap_r2,  # noqa: E402
                               out_of_fold, permutation_importance_grouped, score)
from coffeepred.validate import ATTRIBUTES, TARGETS, validate  # noqa: E402

DATA = ROOT / "data" / "coffee_ratings.csv"
OUT = ROOT / "results"
PBI = ROOT / "powerbi"
SEED = 0
MODELS = ["mean", "ridge", "random_forest", "lightgbm"]
SCHEMES = ["grouped", "random", "time"]


def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))


def main():
    OUT.mkdir(exist_ok=True)
    PBI.mkdir(exist_ok=True)

    print("E1 data validation")
    df, report = validate(pd.read_csv(DATA))
    report.to_csv(OUT / "data_quality_report.csv", index=False)
    config = {"seed": SEED, "features": FEATURES, "targets": TARGETS, "models": MODELS, "schemes": SCHEMES,
              "folds": 5, "group": "owner", "time_split": "graded before 2016 vs 2016 to 2018"}
    lineage.write(lineage.manifest(DATA, config, report.to_dict("records")), OUT / "run_manifest.json")

    print("E2 benchmark: 7 targets x 4 models x 3 validation schemes")
    rows, oof = [], {}
    for t in TARGETS:
        for m in MODELS:
            for sc in SCHEMES:
                p = out_of_fold(df, t, m, sc, seed=SEED)
                s = score(df[t], p)
                rows.append({"target": t, "model": m, "scheme": sc, **s})
                if sc == "grouped":
                    oof[(t, m)] = p
    bench = pd.DataFrame(rows)
    base = bench[bench.model == "mean"].set_index(["target", "scheme"]).mae
    bench["mae_vs_mean"] = 1 - bench.mae / bench.set_index(["target", "scheme"]).index.map(base).values
    bench.to_csv(OUT / "benchmark.csv", index=False)

    grouped = bench[(bench.scheme == "grouped") & (bench.model != "mean")]
    best = grouped.loc[grouped.groupby("target").r2.idxmax()].set_index("target")
    ci = {t: group_bootstrap_r2(df[t], oof[(t, best.loc[t, "model"])], df.group, seed=SEED) for t in TARGETS}
    best["r2_ci_low"] = [ci[t][0] for t in best.index]
    best["r2_ci_high"] = [ci[t][1] for t in best.index]
    best.to_csv(OUT / "best_models.csv")

    print("E3 sensitivity: harvest year as a feature")
    year = []
    for sc in ["grouped", "time"]:
        for m in ["ridge", "random_forest"]:
            p = out_of_fold(df, "total_cup_points", m, sc, seed=SEED, features=FEATURES + ["harvest_year"],
                            numeric=NUMERIC + ["harvest_year"])
            year.append({"scheme": sc, "model": m, **score(df.total_cup_points, p)})
    pd.DataFrame(year).to_csv(OUT / "harvest_year_sensitivity.csv", index=False)

    print("E4 what the models use")
    imp = {}
    for t in ["total_cup_points", "acidity", "body", "flavor"]:
        imp[t] = permutation_importance_grouped(df, t, best.loc[t, "model"], seed=SEED)
    pd.DataFrame(imp).to_csv(OUT / "importance.csv")

    print("E5 prediction intervals (split conformal, grouped CV)")
    conf_rows = []
    m_total = best.loc["total_cup_points", "model"]
    intervals = None
    for alpha in [0.2, 0.1]:
        ci_df = conformal_intervals(df, "total_cup_points", m_total, alpha=alpha, seed=SEED)
        per_fold = ci_df.groupby("fold").covered.mean()
        conf_rows.append({"target": "total_cup_points", "model": m_total, "nominal": 1 - alpha,
                          "coverage": ci_df.covered.mean(), "mean_width": (ci_df.high - ci_df.low).mean(),
                          "fold_coverage_min": per_fold.min(), "fold_coverage_max": per_fold.max()})
        if alpha == 0.2:
            intervals = ci_df
    pd.DataFrame(conf_rows).to_csv(OUT / "conformal.csv", index=False)

    print("E6 who graded the coffee: lab effect")
    with_lab = out_of_fold(df, "total_cup_points", "ridge", "grouped", seed=SEED, features=FEATURES + ["lab"],
                           categorical=["country", "process", "variety", "species", "color", "lab"])
    no_lab = oof[("total_cup_points", "ridge")]
    lab_gain = {"r2_product_only": score(df.total_cup_points, no_lab)["r2"],
                "r2_product_and_lab": score(df.total_cup_points, with_lab)["r2"]}
    resid = df.total_cup_points - no_lab
    rng = np.random.default_rng(SEED)
    lab_rows = []
    for country, g in df.assign(resid=resid).groupby("country"):
        labs = g.lab.value_counts()
        labs = labs[labs >= 15]
        if len(labs) < 2:
            continue
        for lab in labs.index:
            sub = g[g.lab == lab]
            owners = sub.group.unique()
            by_owner = {o: sub.resid[sub.group == o].values for o in owners}
            boots = [np.concatenate([by_owner[o] for o in rng.choice(owners, len(owners))]).mean()
                     for _ in range(1000)]
            lab_rows.append({"country": country, "lab": lab, "lots": len(sub), "owners": len(owners),
                             "largest_owner_share": sub.group.value_counts(normalize=True).iloc[0],
                             "first_year": sub.graded_on.dt.year.min(), "last_year": sub.graded_on.dt.year.max(),
                             "mean_residual": sub.resid.mean(), "ci_low": np.quantile(boots, 0.025),
                             "ci_high": np.quantile(boots, 0.975)})
    labs = pd.DataFrame(lab_rows)
    labs.to_csv(OUT / "lab_effect.csv", index=False)
    lab_gain["pairs"] = len(labs)
    lab_gain["pairs_dominated_by_one_owner"] = int((labs.largest_owner_share >= 0.9).sum())
    dump(lab_gain, "lab_effect.json")

    print("E7 the circular model")
    circ = out_of_fold(df, "total_cup_points", "ridge", "grouped", seed=SEED, features=ATTRIBUTES,
                       categorical=[], numeric=ATTRIBUTES)
    dump({"r2_total_from_attributes": score(df.total_cup_points, circ)["r2"]}, "circular.json")

    print("Power BI export")
    lots = df[["lot_id", "country", "process", "variety", "species", "altitude_m", "moisture", "lab",
               "graded_on"] + TARGETS].copy()
    for t in TARGETS:
        lots[f"pred_{t}"] = oof[(t, best.loc[t, "model"])].round(3).values
    lots = lots.merge(intervals[["lot_id", "pred", "low", "high"]].rename(
        columns={"pred": "pred_total_conformal", "low": "pred_total_low80", "high": "pred_total_high80"}),
        on="lot_id", how="left")
    lots.to_csv(PBI / "lots_with_predictions.csv", index=False)
    bench.to_csv(PBI / "model_metrics.csv", index=False)
    report.to_csv(PBI / "data_quality.csv", index=False)
    print(best[["model", "r2", "r2_ci_low", "r2_ci_high", "mae", "mae_vs_mean"]].round(3))
    print(pd.DataFrame(conf_rows).round(3))
    print(json.dumps(lab_gain, indent=1, default=float))


if __name__ == "__main__":
    main()
