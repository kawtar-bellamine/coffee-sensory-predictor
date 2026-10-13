"""Data-quality rules for the coffee table.

Every rule says what it checks, how many rows fail, and what is done about it (drop the row,
set the value to missing, or only flag it). The report is saved with every run, so any number
in the results can be traced back to the cleaning decisions behind it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

GRADED = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance"]
ATTRIBUTES = GRADED + ["uniformity", "clean_cup", "sweetness", "cupper_points"]
TARGETS = GRADED + ["total_cup_points"]

COUNTRY_FIXES = {
    "Tanzania, United Republic Of": "Tanzania",
    "Cote d?Ivoire": "Côte d'Ivoire",
    "United States (Hawaii)": "United States",
    "United States (Puerto Rico)": "Puerto Rico",
}
PROCESS_FIXES = {
    "Washed / Wet": "Washed",
    "Natural / Dry": "Natural",
    "Semi-washed / Semi-pulped": "Semi-washed",
    "Pulped natural / honey": "Honey",
    "Other": "Other",
}


@dataclass
class Rule:
    name: str
    description: str
    failing: Callable[[pd.DataFrame], pd.Series]   # boolean mask of failing rows
    action: str                                    # "drop", "set_missing:<column>" or "flag"


RULES = [
    Rule("empty_scores", "every attribute score is zero",
         lambda d: (d[ATTRIBUTES] == 0).all(axis=1), "drop"),
    Rule("total_not_sum", "total score differs from the sum of the ten attributes by more than 0.5",
         lambda d: (d.total_cup_points - d[ATTRIBUTES].sum(axis=1)).abs() > 0.5, "drop"),
    Rule("score_out_of_range", "an attribute score is outside 0 to 10",
         lambda d: ((d[ATTRIBUTES] < 0) | (d[ATTRIBUTES] > 10)).any(axis=1), "drop"),
    Rule("altitude_implausible", "altitude below 100 m or above 3,000 m (lost decimals, feet typed as metres)",
         lambda d: d.altitude_mean_meters.notna() & ~d.altitude_mean_meters.between(100, 3000),
         "set_missing:altitude_mean_meters"),
    Rule("moisture_zero", "moisture recorded as 0 (green coffee is usually 8 to 13%): missing value coded as zero",
         lambda d: d.moisture == 0, "set_missing:moisture"),
    Rule("moisture_high", "moisture above 20%",
         lambda d: d.moisture > 0.20, "flag"),
    Rule("process_missing", "processing method missing",
         lambda d: d.processing_method.isna(), "flag"),
    Rule("grading_date_unparsed", "grading date cannot be parsed",
         lambda d: _parse_date(d.grading_date).isna(), "flag"),
    Rule("duplicate_record", "same owner, country, date and identical scores as an earlier row",
         lambda d: d.duplicated(subset=["owner", "country_of_origin", "grading_date"] + ATTRIBUTES), "flag"),
]


def _parse_date(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
    return pd.to_datetime(cleaned, format="%B %d, %Y", errors="coerce")


def _year(text) -> float:
    years = re.findall(r"20\d\d", str(text))
    return float(years[0]) if years else np.nan


def validate(raw: pd.DataFrame, rules: list[Rule] = RULES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the rules in order. Returns the cleaned table and the data-quality report."""
    d = raw.copy()
    rows = []
    for r in rules:
        mask = pd.Series(r.failing(d), index=d.index).fillna(False).astype(bool)
        rows.append({"rule": r.name, "description": r.description, "rows_failing": int(mask.sum()),
                     "action": r.action, "rows_before": len(d)})
        if r.action == "drop":
            d = d[~mask]
        elif r.action.startswith("set_missing:"):
            d.loc[mask, r.action.split(":", 1)[1]] = np.nan
    report = pd.DataFrame(rows)
    report["rows_after"] = report.rows_before - np.where(report.action == "drop", report.rows_failing, 0)
    return _standardise(d.reset_index(drop=True)), report


def _standardise(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["country"] = d.country_of_origin.replace(COUNTRY_FIXES).fillna("Unknown")
    d["process"] = d.processing_method.map(PROCESS_FIXES).fillna("Unknown")
    d["variety"] = d.variety.fillna("Unknown")
    d["color"] = d.color.fillna("Unknown")
    d["altitude_m"] = d.altitude_mean_meters
    d["harvest_year"] = d.harvest_year.map(_year)
    d["graded_on"] = _parse_date(d.grading_date)
    d["lab"] = d.in_country_partner.fillna("Unknown")
    # one group per owner, so that lots from one producer are never split between training and test
    owner = d.owner.astype(object)
    d["group"] = pd.factorize(owner.where(owner.notna(), [f"_lot{i}" for i in range(len(d))]))[0]
    d["lot_id"] = [f"lot_{i + 1:04d}" for i in range(len(d))]
    return d
