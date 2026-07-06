"""
Train the QB NFL-success projection models.

Run end-to-end:
    python3 -m model.train

What it does
------------
1. Joins labels -> profiles (via model.data), builds the training frame from
   FINAL labels (weight 1.0) + PROVISIONAL 2023 labels (weight 0.5, documented).
   projection_target rows (2024-25) are NEVER trained on.
2. Trains regularized logistic models on a hand-curated, PFF-gap-tolerant
   feature set (CFBD PPA + combine + context):
       - PRE-DRAFT  : college + athletic testing only (usable on the 2026 class)
       - POST-DRAFT : PRE-DRAFT + draft capital (-log pick)
   plus two baselines:
       - PICK-ONLY  : logistic on draft capital alone (the market's own bet)
       - R1 heuristic: round-1 pick == hit
3. Evaluation is leave-one-draft-class-out (grouped by draft_year), reporting
   AUC / log-loss / Brier / calibration on the 84 final-label QBs.
4. Fits production models on all training rows and writes artifacts:
       model/artifacts/{pre_draft,post_draft,pick_only}.joblib
       model/artifacts/features_used.json
       model/artifacts/cv_metrics.json

Deterministic: fixed seed, fixed hyper-grid, no randomness in logistic solver.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy import stats as _spstats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.data import (RANDOM_SEED, REPO, add_derived, join_labels_profiles)

ART = os.path.join(REPO, "model", "artifacts")
os.makedirs(ART, exist_ok=True)

# --- hand-curated deployable feature set (works for every class incl. 2026) --
# Kept intentionally small & low-collinearity: efficiency, production, and a
# situational-efficiency signal. (final_average_ppa_all is dropped because it is
# ~0.85 correlated with career_average_ppa_all and caused sign-flip instability.)
PPA_FEATS = [
    "career_average_ppa_all",           # career efficiency (top college AUC ~.77)
    "career_total_ppa_all",             # career production / usage volume
    "final_average_ppa_passing_downs",  # efficiency on obvious passing downs
]
COMBINE_FEATS = ["combine_weight", "combine_forty", "combine_broad"]
CONTEXT_FEATS = ["power5", "career_cfbd_seasons"]

PRE_DRAFT_FEATS = PPA_FEATS + COMBINE_FEATS + CONTEXT_FEATS
POST_DRAFT_FEATS = PRE_DRAFT_FEATS + ["neg_log_pick"]
PICK_ONLY_FEATS = ["neg_log_pick"]

# combine columns are ~30% missing -> add explicit missingness indicators

C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0]


# --------------------------------------------------------------------------- #
def build_training_frame():
    joined, unjoinable = join_labels_profiles(verbose=True)
    joined = add_derived(joined)
    train = joined[joined["label_status"].isin(["final", "provisional"])].copy()
    train["y"] = train["hit"].astype(int)
    train["w"] = np.where(train["label_status"] == "final", 1.0, 0.5)
    train = train.reset_index(drop=True)
    return train, joined, unjoinable


def make_pipeline(feats, C):
    """Median-impute + scale + L2 logistic.

    No missing-indicators: they proved to be a data-availability leak (QBs
    lacking combine/CFBD data are lower-pedigree), which spuriously rewarded
    small-school QBs. We evaluate on actual football signal, imputing missing
    athletic tests to the population median (neutral)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, solver="lbfgs", max_iter=5000,
            random_state=RANDOM_SEED)),  # default penalty is L2
    ])


def make_tier_pipeline(feats, alpha=5.0):
    """Ridge regression onto success_tier (0-4) -> expected_tier."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
    ])


def loco_tier(train, feats, alpha=5.0):
    """LOCO OOF expected-tier predictions on final rows."""
    eval_mask = train["label_status"] == "final"
    oof = pd.Series(index=train.index, dtype=float)
    for yr in sorted(train.loc[eval_mask, "draft_year"].unique()):
        te = train.index[(train["draft_year"] == yr) & eval_mask]
        tr = train.index[train["draft_year"] != yr]
        pipe = make_tier_pipeline(feats, alpha)
        pipe.fit(_Xy(train.loc[tr], feats), train.loc[tr, "success_tier"],
                 reg__sample_weight=train.loc[tr, "w"])
        oof.loc[te] = np.clip(pipe.predict(_Xy(train.loc[te], feats)), 0, 4)
    ev = train.loc[eval_mask]
    return ev["success_tier"].values, oof.loc[ev.index].values


def _Xy(df, feats):
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(float)
    return X


def loco_oof(train, feats, C, eval_status=("final",)):
    """Leave-one-draft-class-out OOF predictions.

    Test folds are restricted to `eval_status` classes (final => trustworthy
    outcomes); provisional rows are always kept in TRAIN, never tested.
    Returns (y_true, p_pred, weights_ignored) aligned to eval rows.
    """
    eval_mask = train["label_status"].isin(eval_status)
    eval_years = sorted(train.loc[eval_mask, "draft_year"].unique())
    oof = pd.Series(index=train.index, dtype=float)
    for yr in eval_years:
        te = train.index[(train["draft_year"] == yr) & eval_mask]
        tr = train.index[train["draft_year"] != yr]
        pipe = make_pipeline(feats, C)
        pipe.fit(_Xy(train.loc[tr], feats), train.loc[tr, "y"],
                 clf__sample_weight=train.loc[tr, "w"])
        oof.loc[te] = pipe.predict_proba(_Xy(train.loc[te], feats))[:, 1]
    ev = train.loc[eval_mask]
    return ev["y"].values, oof.loc[ev.index].values


def metrics(y, p):
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "n": int(len(y)),
        "n_hit": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
    }


def pick_best_C(train, feats):
    best, best_auc = C_GRID[0], -1
    for C in C_GRID:
        y, p = loco_oof(train, feats, C)
        a = roc_auc_score(y, p)
        if a > best_auc:
            best_auc, best = a, C
    return best


def calibration_table(y, p, bins=(0, 0.1, 0.2, 0.35, 0.6, 1.01)):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], bins=list(bins), right=False)
    g = df.groupby("bin", observed=True).agg(n=("y", "size"),
                                             pred=("p", "mean"),
                                             obs=("y", "mean")).reset_index()
    g["bin"] = g["bin"].astype(str)
    return g


# --------------------------------------------------------------------------- #
def main():
    train, joined, unjoinable = build_training_frame()
    fin = train[train["label_status"] == "final"]
    print(f"\nTraining frame: {len(train)} rows "
          f"(final={len(fin)}/{int(fin['y'].sum())} hits, "
          f"provisional={int((train['label_status']=='provisional').sum())} weighted 0.5)")

    results = {}

    # --- baselines & models ------------------------------------------------
    specs = {
        "pick_only": PICK_ONLY_FEATS,
        "pre_draft": PRE_DRAFT_FEATS,
        "post_draft": POST_DRAFT_FEATS,
    }
    chosen_C = {}
    for name, feats in specs.items():
        C = pick_best_C(train, feats)
        chosen_C[name] = C
        y, p = loco_oof(train, feats, C)
        results[name] = metrics(y, p)
        results[name]["C"] = C
        results[name]["features"] = feats
        print(f"\n[{name}] C={C}  LOCO-CV: {metrics(y,p)}")
        if name == "post_draft":
            print("  calibration:\n", calibration_table(y, p).to_string(index=False))

    # round-1 heuristic baseline
    r1 = (fin["round"] == 1).astype(int).values
    yv = fin["y"].values
    results["round1_heuristic"] = {
        "auc": round(float(roc_auc_score(yv, r1)), 4),
        "precision": round(float((yv[r1 == 1].mean()) if r1.sum() else 0), 4),
        "recall": round(float(yv[r1 == 1].sum() / yv.sum()), 4),
        "n_round1": int(r1.sum()),
        "note": "predict hit iff round==1",
    }
    print(f"\n[round1_heuristic] {results['round1_heuristic']}")

    # value of college stats beyond draft capital
    delta = results["post_draft"]["auc"] - results["pick_only"]["auc"]
    print(f"\nAUC lift of (college+combine) over draft-capital-only: {delta:+.4f}")
    results["college_lift_over_pick"] = round(float(delta), 4)

    # --- secondary ordinal model: expected success tier (0-4) --------------
    t_true, t_pred = loco_tier(train, PRE_DRAFT_FEATS)
    tier_metrics = {
        "spearman_rho": round(float(_spstats.spearmanr(t_true, t_pred).statistic), 4),
        "mae": round(float(mean_absolute_error(t_true, t_pred)), 4),
        "n": int(len(t_true)),
        "note": "Ridge onto success_tier, PRE-DRAFT features, LOCO-CV on final rows",
    }
    results["tier_pre_draft"] = tier_metrics
    print(f"\n[tier_pre_draft] LOCO expected-tier: {tier_metrics}")

    # --- fit production models on ALL training rows ------------------------
    prod = {}
    for name, feats in specs.items():
        pipe = make_pipeline(feats, chosen_C[name])
        pipe.fit(_Xy(train, feats), train["y"], clf__sample_weight=train["w"])
        joblib.dump(pipe, os.path.join(ART, f"{name}.joblib"))
        prod[name] = pipe

    tier_pipe = make_tier_pipeline(PRE_DRAFT_FEATS)
    tier_pipe.fit(_Xy(train, PRE_DRAFT_FEATS), train["success_tier"],
                  reg__sample_weight=train["w"])
    joblib.dump(tier_pipe, os.path.join(ART, "tier_pre_draft.joblib"))

    features_used = {
        "pre_draft": PRE_DRAFT_FEATS,
        "post_draft": POST_DRAFT_FEATS,
        "pick_only": PICK_ONLY_FEATS,
        "chosen_C": chosen_C,
        "imputation": "median (no missing-indicators; power5=0 for FCS/missing)",
        "notes": "PPA+combine+context; PFF excluded from deployable model due to "
                 "2014-15-only coverage. Trained on final(1.0)+provisional(0.5).",
    }
    with open(os.path.join(ART, "features_used.json"), "w") as fh:
        json.dump(features_used, fh, indent=2)
    with open(os.path.join(ART, "cv_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # standardized coefficients (interpretability) for pre & post
    coef_dump = {}
    for name in ("pre_draft", "post_draft"):
        pipe = prod[name]
        feats = specs[name]
        coefs = pipe.named_steps["clf"].coef_[0]
        order = np.argsort(-np.abs(coefs))
        print(f"\n[{name}] standardized logistic coefficients (production fit):")
        coef_dump[name] = {}
        for i in order:
            print(f"   {feats[i]:34s} {coefs[i]:+.3f}")
            coef_dump[name][feats[i]] = round(float(coefs[i]), 4)
    with open(os.path.join(ART, "coefficients.json"), "w") as fh:
        json.dump(coef_dump, fh, indent=2)

    print("\nArtifacts written to", ART)
    return results


if __name__ == "__main__":
    main()
