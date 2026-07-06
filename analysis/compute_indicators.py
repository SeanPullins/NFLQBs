"""
Univariate indicator analysis: for every candidate college metric, how well
does it separate NFL hits from non-hits among the FINAL-label draft classes?

Outputs
-------
- model/indicators.csv           (deliverable E: tidy ranked indicator table)
- analysis/_indicator_stats.json (numbers reused by findings.md / figures)

Method (small-n honest):
- Restrict to FINAL labels (2015-2022 classes, fully observed outcomes).
- Univariate AUC = P(a random hit out-ranks a random non-hit on this metric).
  Reported directionally (0.5 = no signal; distance from 0.5 = strength).
- Cliff's delta = 2*AUC - 1 (rank effect size, robust to outliers & small n).
- Mann-Whitney U p-value + point-biserial r as supporting evidence.
- PFF metrics carry a coverage flag: PFF charting only exists for 2014-2015
  college seasons, so it is a TRUE final-season snapshot for 2015-2016 draft
  classes only; later classes reflect a 2015 underclassman season. PFF metrics
  are additionally filtered to >=200 final-season dropbacks to drop cameo noise.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from model.data import (CANDIDATES, REPO, add_derived, join_labels_profiles)

PFF_MIN_DROPBACKS = 200
OUT_CSV = os.path.join(REPO, "model", "indicators.csv")
OUT_JSON = os.path.join(REPO, "analysis", "_indicator_stats.json")


def _evidence(auc: float, n: int, p: float) -> str:
    d = abs(auc - 0.5)
    if n < 12:
        return "insufficient"
    if d >= 0.20 and p < 0.05:
        return "strong"
    if d >= 0.13 and (p < 0.15 or n >= 40):
        return "moderate"
    if d >= 0.08:
        return "weak"
    return "negligible"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    fin = df[df["label_status"] == "final"].copy()
    y_all = fin["hit"].astype(int)
    rows = []
    for f in CANDIDATES:
        if f.col not in fin.columns:
            continue
        sub = fin[[f.col, "hit"]].copy()
        note = ""
        if f.group == "pff":
            db = fin.get("final_grades_dropbacks")
            if db is not None:
                sub = sub[db >= PFF_MIN_DROPBACKS]
            note = "PFF coverage 2014-15 only; true final-season for 2015-16 classes, else a 2015 underclassman season"
        sub = sub.dropna(subset=[f.col])
        x = sub[f.col].astype(float).values
        y = sub["hit"].astype(int).values
        n, n_hit, n_non = len(y), int(y.sum()), int((y == 0).sum())
        if n_hit < 2 or n_non < 2 or np.nanstd(x) == 0:
            continue
        auc = roc_auc_score(y, x)  # directional: >0.5 => higher metric favors hits
        try:
            u, p = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
        except ValueError:
            p = np.nan
        r_pb = stats.pointbiserialr(y, x).statistic
        cliffs = 2 * auc - 1
        direction = "higher = more hits" if auc >= 0.5 else "lower = more hits"
        rows.append({
            "metric": f.label,
            "column": f.col,
            "group": f.group,
            "direction": direction,
            "n": n,
            "n_hit": n_hit,
            "n_nonhit": n_non,
            "median_hit": round(float(np.median(x[y == 1])), 3),
            "median_nonhit": round(float(np.median(x[y == 0])), 3),
            "mean_hit": round(float(np.mean(x[y == 1])), 3),
            "mean_nonhit": round(float(np.mean(x[y == 0])), 3),
            "auc": round(float(auc), 3),
            "abs_auc": round(float(max(auc, 1 - auc)), 3),
            "cliffs_delta": round(float(cliffs), 3),
            "point_biserial_r": round(float(r_pb), 3),
            "mannwhitney_p": round(float(p), 4) if p == p else np.nan,
            "evidence_strength": _evidence(auc, n, p if p == p else 1.0),
            "coverage_note": note,
        })
    out = pd.DataFrame(rows).sort_values("abs_auc", ascending=False).reset_index(drop=True)
    return out


def main():
    joined, unjoinable = join_labels_profiles(verbose=True)
    joined = add_derived(joined)
    ind = compute(joined)
    ind.to_csv(OUT_CSV, index=False)

    fin = joined[joined["label_status"] == "final"]
    meta = {
        "n_labels_total": int(len(join_labels_profiles()[0]) + len(unjoinable)),
        "n_joined": int(len(joined)),
        "n_unjoinable": int(len(unjoinable)),
        "unjoinable": unjoinable.to_dict("records") if len(unjoinable) else [],
        "n_final": int(len(fin)),
        "n_final_hits": int(fin["hit"].sum()),
        "final_base_rate": round(float(fin["hit"].mean()), 4),
        "pff_min_dropbacks": PFF_MIN_DROPBACKS,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {OUT_CSV}  ({len(ind)} indicators)")
    show = ["metric", "group", "direction", "n", "n_hit", "auc", "abs_auc",
            "cliffs_delta", "mannwhitney_p", "evidence_strength"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(ind[show].to_string(index=False))


if __name__ == "__main__":
    main()
