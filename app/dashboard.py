"""Coffee sensory predictor: KPI dashboard.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coffeepred import lineage  # noqa: E402
from coffeepred.models import FEATURES, MODEL_NAMES, fit_conformal, make_model  # noqa: E402
from coffeepred.validate import TARGETS, validate  # noqa: E402

RES = ROOT / "results"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"
LABELS = {"total_cup_points": "total score", "aroma": "aroma", "flavor": "flavour", "aftertaste": "aftertaste",
          "acidity": "acidity", "body": "body", "balance": "balance"}

st.set_page_config(page_title="Coffee sensory predictor", layout="wide")


@st.cache_data
def data():
    df, report = validate(pd.read_csv(ROOT / "data" / "coffee_ratings.csv"))
    return df, report


@st.cache_resource
def fitted_models():
    df, _ = data()
    best = pd.read_csv(RES / "best_models.csv", index_col=0)
    models = {}
    for t in TARGETS:
        m = make_model(best.loc[t, "model"], seed=0)
        m.fit(df[FEATURES], df[t])
        models[t] = m
    return models


@st.cache_resource
def total_with_interval():
    """Split-conformal model for the total score, fitted on all lots (75% of owners fit, 25% calibrate)."""
    df, _ = data()
    return fit_conformal(df, "total_cup_points", best.loc["total_cup_points", "model"], alpha=0.2, seed=0)


df, report = data()
best = pd.read_csv(RES / "best_models.csv", index_col=0)
bench = pd.read_csv(RES / "benchmark.csv")
conf = pd.read_csv(RES / "conformal.csv")
manifest = json.loads((RES / "run_manifest.json").read_text())

page = st.sidebar.radio("View", ["KPIs", "Predict a coffee", "Model benchmark", "Data quality", "Traceability"])
same_data = lineage.sha256(ROOT / "data" / "coffee_ratings.csv") == manifest["data_sha256"]
st.sidebar.caption(f"{len(df):,} lots, {df.country.nunique()} origins, {df.group.nunique()} owners. "
                   f"Data file SHA-256 {manifest['data_sha256'][:12]} "
                   f"({'matches the results' if same_data else 'DOES NOT match the results: rerun the experiments'}).")

if page == "KPIs":
    st.title("Coffee sensory predictor")
    st.write("How well can the cup profile of a coffee be predicted from its product data, before anyone tastes "
             "it? Unless stated otherwise, every figure comes from owners the model never saw during training.")
    c80 = conf[conf.nominal == 0.8].iloc[0]
    t = best.loc["total_cup_points"]
    k = st.columns(4)
    k[0].metric("Lots after validation", f"{len(df):,}", f"{int(report.rows_failing.sum())} rule hits", delta_color="off")
    k[1].metric("R² of the total score, new owners", f"{t.r2:.2f}", f"95% CI {t.r2_ci_low:.2f} to {t.r2_ci_high:.2f}",
                delta_color="off")
    k[2].metric("80% interval coverage", f"{c80.coverage:.0%}", f"width {c80.mean_width:.1f} points",
                delta_color="off")
    tm = bench[(bench.target == "total_cup_points") & (bench.model == t.model)].set_index("scheme").r2
    k[3].metric("R² on later years", f"{tm['time']:.2f}", f"random split would say {tm['random']:.2f}",
                delta_color="off")
    b = best.reset_index().assign(attribute=lambda d: d.target.map(LABELS))
    st.altair_chart(alt.Chart(b).mark_bar(color=BLUE).encode(
        x=alt.X("r2:Q", title="R² on new owners", scale=alt.Scale(domain=[0, 0.3])),
        y=alt.Y("attribute:N", sort="-x", title=None),
        tooltip=["attribute", "model", alt.Tooltip("r2:Q", format=".3f"), alt.Tooltip("mae:Q", format=".3f")])
        .properties(height=260), width="stretch")
    st.info("Product data explain about a tenth of the cup profile. The cup still has to be tasted; the model is a "
            "screening aid, not a replacement for the panel.")

elif page == "Predict a coffee":
    st.title("Predict a coffee")
    c = st.columns(3)
    country = c[0].selectbox("Origin", sorted(df.country.unique()), index=sorted(df.country.unique()).index("Ethiopia"))
    process = c[1].selectbox("Process", sorted(df.process.unique()), index=sorted(df.process.unique()).index("Washed"))
    variety = c[2].selectbox("Variety", sorted(df.variety.unique()))
    c = st.columns(3)
    species = c[0].selectbox("Species", sorted(df.species.unique()))
    color = c[1].selectbox("Bean colour", sorted(df.color.unique()))
    altitude = c[2].slider("Altitude (m)", 100, 3000, 1800, 50)
    c = st.columns(3)
    moisture = c[0].slider("Moisture (%)", 5.0, 20.0, 11.0, 0.5) / 100
    cat1 = c[1].number_input("Category 1 defects", 0, 70, 0)
    cat2 = c[2].number_input("Category 2 defects", 0, 60, 2)
    x = pd.DataFrame([{"country": country, "process": process, "variety": variety, "species": species,
                       "color": color, "altitude_m": altitude, "moisture": moisture,
                       "category_one_defects": cat1, "category_two_defects": cat2, "quakers": 0}])[FEATURES]
    models = fitted_models()
    preds = {t: float(models[t].predict(x)[0]) for t in TARGETS}
    cmodel, half = total_with_interval()
    centre = float(cmodel.predict(x)[0])
    st.metric("Predicted total score", f"{centre:.1f}", f"80% interval {centre - half:.1f} to {centre + half:.1f}",
              delta_color="off")
    prof = pd.DataFrame({"attribute": [LABELS[t] for t in TARGETS[:-1]], "predicted": [preds[t] for t in TARGETS[:-1]],
                         "average lot": [df[t].mean() for t in TARGETS[:-1]]}).melt(id_vars="attribute")
    st.altair_chart(alt.Chart(prof).mark_circle(size=120).encode(
        x=alt.X("value:Q", scale=alt.Scale(zero=False), title="score (0 to 10)"),
        y=alt.Y("attribute:N", title=None), color=alt.Color("variable:N", scale=alt.Scale(range=[MUTED, BLUE]),
                                                             title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["attribute", "variable", alt.Tooltip("value:Q", format=".2f")]).properties(height=280),
        width="stretch")
    st.caption("Attribute models are refitted on all lots. The total score and its 80% interval come from a "
               "split-conformal model: 75% of owners fit it, the other 25% set the interval width.")

elif page == "Model benchmark":
    st.title("Model benchmark")
    target = st.selectbox("Target", TARGETS, format_func=lambda t: LABELS[t])
    b = bench[bench.target == target].copy()
    b["model name"] = b.model.map(MODEL_NAMES)
    b["validation"] = b.scheme.map({"grouped": "new owners", "random": "random split", "time": "later years"})
    st.altair_chart(alt.Chart(b).mark_bar().encode(
        x=alt.X("validation:N", title=None, sort=["random split", "new owners", "later years"]),
        y=alt.Y("r2:Q", title="R²"), color=alt.Color("validation:N", scale=alt.Scale(
            domain=["random split", "new owners", "later years"], range=[MUTED, BLUE, ORANGE]), legend=None),
        column=alt.Column("model name:N", title=None),
        tooltip=["model name", "validation", alt.Tooltip("r2:Q", format=".3f"), alt.Tooltip("mae:Q", format=".3f")])
        .properties(width=140, height=260))
    st.dataframe(b[["model name", "validation", "r2", "mae", "mae_vs_mean", "n"]].round(3), hide_index=True,
                 width="stretch")

elif page == "Data quality":
    st.title("Data quality")
    st.dataframe(report, hide_index=True, width="stretch")
    st.altair_chart(alt.Chart(report[report.rows_failing > 0]).mark_bar(color=BLUE).encode(
        x=alt.X("rows_failing:Q", title="rows failing"), y=alt.Y("rule:N", sort="-x", title=None),
        tooltip=["rule", "description", "rows_failing", "action"]).properties(height=240), width="stretch")

else:
    st.title("Traceability")
    st.write("Every run of `experiments/run_all.py` writes this manifest: the exact data file (hash), code version, "
             "package versions, settings and cleaning steps behind the results.")
    st.json(manifest)
