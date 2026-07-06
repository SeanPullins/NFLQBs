"""
Score QB profiles with the trained models and produce human-readable outputs.

Run:
    python3 -m model.predict         # regenerates historical_scored.csv + projections.csv

Library API:
    score(profiles_df, model="pre_draft") -> np.ndarray of hit probabilities
    per_player_indicators(profiles_df, model="pre_draft") -> list[dict] top +/- drivers
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from model.data import (CAND_BY_COL, REPO, add_derived, join_labels_profiles,
                        load_profiles)
from model.train import (POST_DRAFT_FEATS, POST_DRAFT_PFF_FEATS,
                         PRE_DRAFT_FEATS, PRE_DRAFT_NO_PFF_FEATS)
from pipeline.normalize import normalize_name

ART = os.path.join(REPO, "model", "artifacts")
WATCHLIST_2027_PATH = os.path.join(REPO, "data", "watchlists", "qb_2027_watchlist.csv")

# human-readable feature labels (fall back to the candidate catalogue)
_LABELS = {c: f.label for c, f in CAND_BY_COL.items()}
_LABELS.update({
    "career_average_ppa_all": "Career PPA per play",
    "career_total_ppa_all": "Career total production (PPA)",
    "final_average_ppa_passing_downs": "Passing-down efficiency (final yr)",
    "combine_weight": "Weight/frame",
    "combine_forty": "40-yard speed",
    "combine_broad": "Broad-jump explosiveness",
    "power5": "Power-5 competition",
    "career_cfbd_seasons": "College seasons (age/declare proxy)",
    "neg_log_pick": "Draft capital",
})


def _load(model: str):
    return joblib.load(os.path.join(ART, f"{model}.joblib"))


def _feats(model: str):
    return {
        "pre_draft": PRE_DRAFT_FEATS,
        "pre_draft_no_pff": PRE_DRAFT_NO_PFF_FEATS,
        "post_draft": POST_DRAFT_FEATS,
        "post_draft_pff": POST_DRAFT_PFF_FEATS,
        "pick_only": ["neg_log_pick"],
    }[model]


def _prep(profiles_df: pd.DataFrame) -> pd.DataFrame:
    df = profiles_df.copy()
    if "power5" not in df.columns or "neg_log_pick" not in df.columns or "speed_score" not in df.columns:
        df = add_derived(df)
    return df


def score(profiles_df: pd.DataFrame, model: str = "pre_draft") -> np.ndarray:
    df = _prep(profiles_df)
    pipe = _load(model)
    feats = _feats(model)
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(float)
    return pipe.predict_proba(X)[:, 1]


def expected_tier(profiles_df: pd.DataFrame) -> np.ndarray:
    df = _prep(profiles_df)
    pipe = joblib.load(os.path.join(ART, "tier_pre_draft.joblib"))
    X = df[PRE_DRAFT_FEATS].apply(pd.to_numeric, errors="coerce").astype(float)
    return np.clip(pipe.predict(X), 0, 4)


def _contributions(df: pd.DataFrame, model: str):
    """Standardized contribution of each BASE feature to the logit, per player.

    contribution_j = coef_j * scaled(feature_j). Features that were actually
    MISSING for a player (median-imputed) are suppressed -- we don't want to
    tell a fan 'weak 40 time' when the QB simply never ran one.
    """
    pipe = _load(model)
    feats = _feats(model)
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(float)
    missing = X.isna().values
    # transform through impute+scale (all but final clf step)
    Z = pipe[:-1].transform(X)
    coef = pipe.named_steps["clf"].coef_[0]
    contrib = Z[:, :len(feats)] * coef[:len(feats)]  # base features only
    contrib = np.where(missing, np.nan, contrib)     # suppress imputed
    return feats, contrib


def per_player_indicators(profiles_df: pd.DataFrame, model: str = "pre_draft",
                          k: int = 3):
    df = _prep(profiles_df)
    feats, contrib = _contributions(df, model)
    out = []
    for i in range(len(df)):
        c = contrib[i]
        order = np.argsort(np.where(np.isnan(c), 0, c))  # ascending
        pos = [feats[j] for j in order[::-1] if c[j] == c[j] and c[j] > 0][:k]
        neg = [feats[j] for j in order if c[j] == c[j] and c[j] < 0][:k]
        out.append({
            "top_positive_indicators": "; ".join(_LABELS.get(f, f) for f in pos),
            "top_negative_indicators": "; ".join(_LABELS.get(f, f) for f in neg),
        })
    return out


def coverage_note(row) -> str:
    parts = []
    if not row.get("in_pff", False):
        parts.append("no PFF")
    elif pd.isna(row.get("final_concept_dropbacks")) and pd.isna(row.get("final_grades_dropbacks")):
        parts.append("PFF partial")
    else:
        parts.append("PFF")
    if not row.get("in_combine", False):
        parts.append("no combine testing")
    elif pd.isna(row.get("combine_forty")):
        parts.append("combine (no 40)")
    else:
        parts.append("combine")
    if not row.get("in_cfbd", False):
        parts.append("no CFBD PPA")
    return ", ".join(parts)


def _best_watchlist_profile(profiles: pd.DataFrame, normalized_name: str):
    candidates = profiles[profiles["normalized_name"].eq(normalized_name)].copy()
    if candidates.empty:
        return None

    final_season = pd.to_numeric(candidates.get("final_season"), errors="coerce").fillna(0)
    dropbacks = pd.to_numeric(candidates.get("final_concept_dropbacks"), errors="coerce").fillna(0)
    candidates["_watch_score"] = (
        candidates["in_pff"].astype(int) * 8
        + candidates["in_cfbd"].astype(int) * 4
        + candidates["in_combine"].astype(int)
        + final_season / 10000
        + np.log1p(dropbacks) / 100
    )
    best = candidates.sort_values("_watch_score", ascending=False).iloc[0]
    return best.drop(labels="_watch_score")


def _watchlist_note(row: pd.Series) -> str:
    tier = str(row.get("watchlist_tier", "watch")).strip() or "watch"
    source = str(row.get("source_note", "")).strip()
    if source:
        source = f"{source}. "
    return (
        f"2027 early watchlist ({tier}): {source}"
        "Scored from through-2025 PFF/CFBD only; no 2026 season, combine, or draft slot yet."
    )


def _load_2027_watchlist_profiles(profiles: pd.DataFrame, existing_names: set[str]) -> pd.DataFrame:
    if not os.path.exists(WATCHLIST_2027_PATH):
        return profiles.iloc[0:0].copy()

    watchlist = pd.read_csv(WATCHLIST_2027_PATH)
    rows = []
    missing = []
    for _, wr in watchlist.iterrows():
        normalized = normalize_name(wr["player"])
        if normalized in existing_names:
            continue
        best = _best_watchlist_profile(profiles, normalized)
        if best is None:
            missing.append(wr["player"])
            continue

        row = best.to_dict()
        row["draft_year"] = 2027
        row["draft_season"] = 2027
        row["round"] = np.nan
        row["pick"] = np.nan
        row["success_tier"] = np.nan
        row["label_status"] = "watchlist"
        row["college"] = wr.get("display_college") or row.get("colleges")
        row["notes"] = _watchlist_note(wr)
        rows.append(row)

    if missing:
        print("missing 2027 watchlist profiles:", ", ".join(missing))
    if not rows:
        return profiles.iloc[0:0].copy()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _percentiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference)
    return np.array([round(100 * np.searchsorted(ref, v, side="right") / len(ref), 1)
                     for v in values])


def build_outputs():
    joined, unjoinable = join_labels_profiles(verbose=True)
    joined = add_derived(joined)
    profiles = add_derived(load_profiles())

    # historical reference distribution = PFF-enabled pre-draft probs of all labeled QBs
    hist_prob = score(joined, "pre_draft")
    hist_prob_no_pff = score(joined, "pre_draft_no_pff")

    # -------- historical_scored.csv --------
    hs = joined.copy()
    hs["model_hit_prob_pre_draft"] = hist_prob
    hs["model_hit_prob_pre_draft_no_pff"] = hist_prob_no_pff
    hs["pff_model_delta"] = hs["model_hit_prob_pre_draft"] - hs["model_hit_prob_pre_draft_no_pff"]
    has_pick = hs["pick"].notna()
    hs["model_hit_prob_post_draft"] = np.nan
    hs["model_hit_prob_post_draft_pff"] = np.nan
    hs["model_hit_prob_pick_only"] = np.nan
    hs["draft_adjusted_hit_prob"] = hs["model_hit_prob_pre_draft"]
    if has_pick.any():
        hs.loc[has_pick, "model_hit_prob_post_draft"] = score(hs[has_pick], "post_draft")
        hs.loc[has_pick, "model_hit_prob_post_draft_pff"] = score(hs[has_pick], "post_draft_pff")
        hs.loc[has_pick, "model_hit_prob_pick_only"] = score(hs[has_pick], "pick_only")
        hs.loc[has_pick, "draft_adjusted_hit_prob"] = np.maximum(
            hs.loc[has_pick, "model_hit_prob_post_draft"],
            hs.loc[has_pick, "model_hit_prob_pick_only"],
        )
    hs["model_expected_tier"] = expected_tier(hs)
    hs["percentile_vs_history"] = _percentiles(hist_prob, hist_prob)
    ind = per_player_indicators(hs, "pre_draft")
    hs["top_positive_indicators"] = [d["top_positive_indicators"] for d in ind]
    hs["top_negative_indicators"] = [d["top_negative_indicators"] for d in ind]
    hs["data_coverage"] = hs.apply(coverage_note, axis=1)
    hist_cols = ["canonical_name", "college", "draft_year", "round", "pick",
                 "label_status", "success_tier", "hit", "seasons_primary_starter",
                 "draft_adjusted_hit_prob", "model_hit_prob_pre_draft", "model_hit_prob_post_draft",
                 "model_hit_prob_pre_draft_no_pff", "model_hit_prob_post_draft_pff",
                 "model_hit_prob_pick_only", "pff_model_delta", "model_expected_tier", "percentile_vs_history",
                 "top_positive_indicators", "top_negative_indicators",
                 "data_coverage", "notes"]
    hs = hs.rename(columns={"colleges": "college_prof"})
    hs["college"] = hs["college"].fillna(hs.get("college_prof"))
    hs[hist_cols].sort_values("model_hit_prob_pre_draft", ascending=False)\
        .to_csv(os.path.join(REPO, "model", "historical_scored.csv"), index=False)

    # -------- projections.csv --------
    # Use the labeled draft universe for 2023-2026. If a future run lacks real
    # 2026 draft labels, append a compact combine/watchlist slice rather than
    # every PFF-only college QB whose final observed season implies 2026.
    proj = joined[joined["label_status"].isin(["provisional", "projection_target"])].copy()
    # Label-backed rows should display their real draft year/slot even when
    # their best feature profile is a PFF-only proxy season.
    proj["draft_season"] = proj["draft_year"]

    has_2026_draft_labels = proj["draft_year"].eq(2026).any()
    if has_2026_draft_labels:
        watch_2026 = profiles.iloc[0:0].copy()
    else:
        watch_2026 = profiles[(profiles["draft_season"].eq(2026)) & (profiles["in_combine"])].copy()
    if not watch_2026.empty:
        watch_2026 = watch_2026[~watch_2026["normalized_name"].isin(proj["normalized_name"])].copy()
        watch_2026["draft_year"] = watch_2026["draft_season"]
        watch_2026["round"] = np.nan
        watch_2026["pick"] = np.nan
        watch_2026["success_tier"] = np.nan
        watch_2026["label_status"] = "watchlist"
        watch_2026["notes"] = "2026 combine/watchlist row; not an outcome label"
        proj = pd.concat([proj, watch_2026], ignore_index=True, sort=False)

    watch_2027 = _load_2027_watchlist_profiles(
        profiles,
        set(proj["normalized_name"].dropna()),
    )
    if not watch_2027.empty:
        proj = pd.concat([proj, watch_2027], ignore_index=True, sort=False)

    proj["model_hit_prob"] = score(proj, "pre_draft")
    proj["model_hit_prob_no_pff"] = score(proj, "pre_draft_no_pff")
    proj["pff_model_delta"] = proj["model_hit_prob"] - proj["model_hit_prob_no_pff"]
    proj["model_hit_prob_pick_only"] = np.nan
    proj["model_hit_prob_post_draft"] = np.nan
    proj["model_hit_prob_post_draft_pff"] = np.nan
    proj["draft_adjusted_hit_prob"] = proj["model_hit_prob"]
    proj_has_pick = proj["pick"].notna()
    if proj_has_pick.any():
        proj.loc[proj_has_pick, "model_hit_prob_pick_only"] = score(proj[proj_has_pick], "pick_only")
        proj.loc[proj_has_pick, "model_hit_prob_post_draft"] = score(proj[proj_has_pick], "post_draft")
        proj.loc[proj_has_pick, "model_hit_prob_post_draft_pff"] = score(proj[proj_has_pick], "post_draft_pff")
        proj.loc[proj_has_pick, "draft_adjusted_hit_prob"] = np.maximum(
            proj.loc[proj_has_pick, "model_hit_prob_post_draft"],
            proj.loc[proj_has_pick, "model_hit_prob_pick_only"],
        )
    proj["expected_tier"] = expected_tier(proj)
    draft_adjusted_ref = hs["draft_adjusted_hit_prob"].dropna().values
    proj["percentile_vs_history"] = _percentiles(proj["draft_adjusted_hit_prob"].values, draft_adjusted_ref)
    pind = per_player_indicators(proj, "pre_draft")
    proj["top_positive_indicators"] = [d["top_positive_indicators"] for d in pind]
    proj["top_negative_indicators"] = [d["top_negative_indicators"] for d in pind]
    proj["data_coverage"] = proj.apply(coverage_note, axis=1)

    # Attach/normalize outcome metadata for labeled projection rows.
    if "draft_year" not in proj.columns:
        proj["draft_year"] = proj["draft_season"]
    if "colleges" in proj.columns:
        if "college" not in proj.columns:
            proj["college"] = proj["colleges"]
        else:
            proj["college"] = proj["college"].fillna(proj["colleges"])

    proj = proj.rename(columns={"success_tier": "actual_tier_or_projection", "notes": "scout_note"})
    for col in ["draft_season", "round", "pick"]:
        proj[col] = pd.to_numeric(proj[col], errors="coerce").astype("Int64")
    proj_cols = ["canonical_name", "college", "draft_season", "round", "pick",
                 "draft_adjusted_hit_prob", "model_hit_prob", "model_hit_prob_no_pff",
                 "model_hit_prob_pick_only", "model_hit_prob_post_draft",
                 "model_hit_prob_post_draft_pff", "pff_model_delta",
                 "expected_tier", "percentile_vs_history",
                 "actual_tier_or_projection", "label_status",
                 "top_positive_indicators", "top_negative_indicators",
                 "data_coverage", "scout_note"]
    proj[proj_cols].sort_values(["draft_season", "draft_adjusted_hit_prob"],
                                ascending=[True, False])\
        .to_csv(os.path.join(REPO, "model", "projections.csv"), index=False)

    print("wrote model/historical_scored.csv and model/projections.csv")
    return hs, proj


if __name__ == "__main__":
    build_outputs()
