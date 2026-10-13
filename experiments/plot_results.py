"""Figures for the README (reads results/, writes results/figures/)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"mean": "Mean", "ridge": "Ridge", "random_forest": "Random forest", "lightgbm": "LightGBM"}
LABELS = {"total_cup_points": "total score", "aroma": "aroma", "flavor": "flavour", "aftertaste": "aftertaste",
          "acidity": "acidity", "body": "body", "balance": "balance"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_validation():
    b = pd.read_csv(RES / "benchmark.csv")
    b = b[(b.target == "total_cup_points") & (b.model != "mean")]
    piv = b.pivot(index="model", columns="scheme", values="r2").loc[["ridge", "random_forest", "lightgbm"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(piv))
    for i, (sc, col, lab) in enumerate([("random", MUTED, "random split (lots of one owner on both sides)"),
                                        ("grouped", BLUE, "new owners (grouped by owner)"),
                                        ("time", ORANGE, "later years (graded 2016 to 2018)")]):
        ax.bar(x + (i - 1) * 0.27, piv[sc], width=0.27, color=col, label=lab)
        for xi, v in zip(x, piv[sc]):
            ax.text(xi + (i - 1) * 0.27, v + (0.006 if v >= 0 else -0.018), f"{v:.2f}", ha="center", fontsize=8.5,
                    color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(min(-0.09, piv.values.min() - 0.04), piv.values.max() + 0.04)
    ax.set_xticks(x, [NAMES[m] for m in piv.index])
    ax.set_ylabel("R² of the total score")
    ax.set_title("How the model is tested changes the answer")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig1_validation.png")


def fig_targets():
    best = pd.read_csv(RES / "best_models.csv", index_col=0).sort_values("r2")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(best))
    ax.errorbar(best.r2, y, xerr=[best.r2 - best.r2_ci_low, best.r2_ci_high - best.r2], fmt="o", color=BLUE,
                ms=7, lw=1.5, capsize=3)
    for yi, (t, r) in zip(y, best.iterrows()):
        ax.text(r.r2_ci_high + 0.008, yi, f"{NAMES[r.model]}: error {r.mae:.2f}, {r.mae_vs_mean:.0%} below the mean model",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [LABELS[t] for t in best.index])
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("R² on new owners (95% interval, resampling owners)")
    ax.set_title("Product data explain about a tenth of the cup profile")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_targets.png")


def fig_intervals():
    lots = pd.read_csv(ROOT / "powerbi" / "lots_with_predictions.csv")
    conf = pd.read_csv(RES / "conformal.csv")
    s = lots.dropna(subset=["pred_total_low80"]).sample(120, random_state=0).sort_values("pred_total_conformal")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(s))
    inside = (s.total_cup_points >= s.pred_total_low80) & (s.total_cup_points <= s.pred_total_high80)
    ax.vlines(x, s.pred_total_low80, s.pred_total_high80, color=GRID, lw=3)
    ax.scatter(x, s.pred_total_conformal, s=10, color=BLUE, label="prediction", zorder=3)
    ax.scatter(x[inside.values], s.total_cup_points[inside], s=14, color=INK2, marker="x", label="actual, inside",
               zorder=4)
    ax.scatter(x[~inside.values], s.total_cup_points[~inside], s=18, color=ORANGE, marker="x",
               label="actual, outside", zorder=4)
    c80 = conf[conf.nominal == 0.8].iloc[0]
    c90 = conf[conf.nominal == 0.9].iloc[0]
    ax.set_title(f"80% intervals hold ({c80.coverage:.0%} covered; 90%: {c90.coverage:.0%}) but span "
                 f"{c80.mean_width:.1f} points")
    ax.set_xlabel("120 lots from new owners, sorted by prediction")
    ax.set_ylabel("total cup score")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=8.5, ncol=3)
    save(fig, "fig3_intervals.png")


def fig_importance():
    imp = pd.read_csv(RES / "importance.csv", index_col=0)
    imp = imp.loc[imp["total_cup_points"].sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(imp))
    for i, (t, col) in enumerate([("total_cup_points", BLUE), ("acidity", AQUA), ("body", ORANGE)]):
        ax.barh(y + (i - 1) * 0.25, imp[t], height=0.25, color=col, label=LABELS[t])
    ax.set_yticks(y, imp.index)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("drop in R² when the group is shuffled (new owners)")
    ax.set_title("What the models use: bean defects and origin, hardly the process")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_importance.png")


def fig_quality():
    r = pd.read_csv(RES / "data_quality_report.csv")
    r = r[r.rows_failing > 0].sort_values("rows_failing")
    colors = {"drop": ORANGE, "flag": MUTED}
    fig, ax = plt.subplots(figsize=(9, 3.0))
    y = np.arange(len(r))
    ax.barh(y, r.rows_failing, color=[colors.get(a, BLUE) for a in r.action], height=0.6)
    for yi, (n, a) in enumerate(zip(r.rows_failing, r.action)):
        ax.text(n + 3, yi, f"{n} ({a.replace('set_missing:', 'set to missing: ')})", va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(y, r.rule.str.replace("_", " "))
    ax.set_xlim(0, r.rows_failing.max() * 1.6)
    ax.set_xlabel("rows failing the rule (of 1,339)")
    ax.set_title("Data-quality rules applied before any model")
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_data_quality.png")


if __name__ == "__main__":
    fig_validation()
    fig_targets()
    fig_intervals()
    fig_importance()
    fig_quality()
