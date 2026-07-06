"""
Train the QB NFL-success projection models.

Run end-to-end:
    python3 -m model.train

What it does
------------
1. Joins labels -> profiles (via model.data), builds the training frame from
   FINAL labels (weight 1.0) + PROVISIONAL 2023 labels (weight 0.5, documented)
   for production fits. Validation metrics use FINAL rows only.
2. Trains regularized logistic models on hand-curated feature sets:
       - PRE-DRAFT-NO-PFF : CFBD PPA + combine + context
       - PRE-DRAFT        : PRE-DRAFT-NO-PFF + selected PFF charting
       - POST-DRAFT       : PRE-DRAFT-NO-PFF + draft capital (-log pick)
       - POST-DRAFT-PFF   : PRE-DRAFT + draft capital (-log pick), audit only
   plus two baselines:
       - PICK-ONLY  : logistic on draft capital alone (the market's own bet)
       - R1 heuristic: round-1 pick == hit
3. Headline evaluation is forward-by-draft-year, reporting AUC / log-loss /
   Brier on final-label QBs whose draft year has enough prior training data.
   Leave-one-class-out is retained as a secondary robustness check.
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

PRE_DRAFT_NO_PFF_FEATS = PPA_FEATS + COMBINE_FEATS + CONTEXT_FEATS
PFF_FEATS = [
    "final_concept_no_screen_grades_pass",
    "final_concept_no_screen_accuracy_percent",
    "final_concept_no_screen_btt_rate",
    "final_concept_no_screen_twp_rate",
    "final_concept_no_screen_pressure_to_sack_rate",
    "final_pressure_pressure_grades_pass",
    "final_pressure_no_pressure_grades_pass",
    "final_pressure_grades_run",
    "final_concept_dropbacks",
]

PRE_DRAFT_FEATS = PRE_DRAFT_NO_PFF_FEATS + PFF_FEATS
POST_DRAFT_NO_PFF_FEATS = PRE_DRAFT_NO_PFF_FEATS + ["neg_log_pick"]
POST_DRAFT_PFF_FEATS = PRE_DRAFT_FEATS + ["neg_log_pick"]
POST_DRAFT_FEATS = POST_DRAFT_NO_PFF_FEATS
PICK_ONLY_FEATS = ["neg_log_pick"]

MODEL_SPECS = {
    "pick_only": PICK_ONLY_FEATS,
    "pre_draft_no_pff": PRE_DRAFT_NO_PFF_FEATS,
    "pre_draft": PRE_DRAFT_FEATS,
    "post_draft": POST_DRAFT_FEATS,
    "post_draft_pff": POST_DRAFT_PFF_FEATS,
}

FIXED_C = {
    "pick_only": 1.0,
    "pre_draft_no_pff": 0.25,
    "pre_draft": 0.1,
    "post_draft": 0.25,
    "post_draft_pff": 0.25,
}


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


def loco_oof(train, feats, C, eval_status=("final",), train_status=("final",)):
    """Leave-one-draft-class-out OOF predictions.

    Test folds are restricted to `eval_status` classes (final => trustworthy
    outcomes); provisional rows are always kept in TRAIN, never tested.
    Returns (y_true, p_pred, weights_ignored) aligned to eval rows.
    """
    eval_mask = train["label_status"].isin(eval_status)
    train_mask = train["label_status"].isin(train_status)
    eval_years = sorted(train.loc[eval_mask, "draft_year"].unique())
    oof = pd.Series(index=train.index, dtype=float)
    for yr in eval_years:
        te = train.index[(train["draft_year"] == yr) & eval_mask]
        tr = train.index[(train["draft_year"] != yr) & train_mask]
        pipe = make_pipeline(feats, C)
        pipe.fit(_Xy(train.loc[tr], feats), train.loc[tr, "y"],
                 clf__sample_weight=train.loc[tr, "w"])
        oof.loc[te] = pipe.predict_proba(_Xy(train.loc[te], feats))[:, 1]
    ev = train.loc[eval_mask]
    return ev["y"].values, oof.loc[ev.index].values


def forward_oof(train, feats, C, eval_status=("final",), train_status=("final",)):
    """Forward-chaining predictions by draft class.

    For held-out year Y, fit only on rows with draft_year < Y. Years whose
    prior training set has a single outcome class are skipped. This is a
    stricter forecasting check than LOCO because it never trains on future
    draft classes.
    """
    eval_mask = train["label_status"].isin(eval_status)
    train_mask = train["label_status"].isin(train_status)
    y_true, p_pred, years = [], [], []
    for yr in sorted(train.loc[eval_mask, "draft_year"].unique()):
        tr = train[(train["draft_year"] < yr) & train_mask]
        te = train[(train["draft_year"] == yr) & eval_mask]
        if te.empty or tr["y"].nunique() < 2:
            continue
        pipe = make_pipeline(feats, C)
        pipe.fit(_Xy(tr, feats), tr["y"], clf__sample_weight=tr["w"])
        p = pipe.predict_proba(_Xy(te, feats))[:, 1]
        y_true.extend(te["y"].astype(int).tolist())
        p_pred.extend(p.tolist())
        years.extend([int(yr)] * len(te))
    return np.array(y_true), np.array(p_pred), years


def metrics(y, p):
    if len(y) == 0 or len(set(y)) < 2:
        return {"auc": None, "log_loss": None, "brier": None, "n": int(len(y)), "n_hit": int(np.sum(y))}
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "n": int(len(y)),
        "n_hit": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
    }


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
    chosen_C = {}
    for name, feats in MODEL_SPECS.items():
        C = FIXED_C[name]
        chosen_C[name] = C
        y_fwd, p_fwd, fwd_years = forward_oof(train, feats, C)
        y_loco, p_loco = loco_oof(train, feats, C)
        fwd_metrics = metrics(y_fwd, p_fwd)
        loco_metrics = metrics(y_loco, p_loco)
        results[name] = {
            **fwd_metrics,
            "validation": "forward_by_draft_year",
            "forward_years": sorted(set(fwd_years)),
            "loco": loco_metrics,
            "C": C,
            "features": feats,
        }
        print(f"\n[{name}] C={C}  FORWARD: {fwd_metrics}")
        print(f"[{name}] C={C}  LOCO:    {loco_metrics}")
        if name == "post_draft":
            print("  forward calibration:\n", calibration_table(y_fwd, p_fwd).to_string(index=False))

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
    pff_delta = results["pre_draft"]["auc"] - results["pre_draft_no_pff"]["auc"]
    print(f"\nAUC lift of (college+combine) over draft-capital-only: {delta:+.4f}")
    print(f"AUC lift of PFF over no-PFF pre-draft model: {pff_delta:+.4f}")
    results["college_lift_over_pick"] = round(float(delta), 4)
    results["pff_lift_over_no_pff"] = round(float(pff_delta), 4)

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
    for name, feats in MODEL_SPECS.items():
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
        "pre_draft_no_pff": PRE_DRAFT_NO_PFF_FEATS,
        "post_draft": POST_DRAFT_FEATS,
        "post_draft_pff": POST_DRAFT_PFF_FEATS,
        "pick_only": PICK_ONLY_FEATS,
        "chosen_C": chosen_C,
        "imputation": "median (no missing-indicators; power5=0 for FCS/missing)",
        "validation": "headline metrics are forward-by-draft-year on final labels only; LOCO stored as secondary",
        "notes": "pre_draft is PPA+combine+context+selected PFF concept/pressure features. "
                 "pre_draft_no_pff preserves the old no-PFF baseline. post_draft intentionally "
                 "uses the no-PFF feature set because PFF hurt forward validation once draft "
                 "capital was known; post_draft_pff is retained for audit only. Production fits "
                 "use final(1.0)+provisional(0.5); validation uses final only.",
    }
    with open(os.path.join(ART, "features_used.json"), "w") as fh:
        json.dump(features_used, fh, indent=2)
    with open(os.path.join(ART, "cv_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # standardized coefficients (interpretability) for pre & post
    coef_dump = {}
    for name in ("pre_draft", "post_draft"):
        pipe = prod[name]
        feats = MODEL_SPECS[name]
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
