"""
Shared data layer for the QB projection model.

Responsibilities
----------------
1. Load the canonical one-row-per-QB profiles + the hand-curated outcome labels.
2. Join labels -> profiles robustly (name+draft-year, with a name+college
   fallback for the handful of QBs whose profile ``draft_season`` is a PFF
   proxy year rather than their real draft year, e.g. Chad Kelly, Alex McGough).
3. Expose a hand-curated, football-grounded candidate feature list (NOT the
   ~2,300 raw columns) split into groups that tolerate the PFF coverage gap:
       - ppa      : CFBD expected-points-added, present for classes 2015-2026
       - combine  : athletic testing / measurables, present 2016-2026
       - pff      : PFF charting, present only for classes ~2015-2019 today
       - context  : conference strength, experience
       - capital  : draft slot (post-draft model only)
4. Build derived features (power-5 flag, log(pick), etc.).

Everything is deterministic and import-safe (no side effects at import).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.normalize import normalize_name

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(REPO, "data", "processed", "qb_draft_profiles.parquet")
LABELS_PATH = os.path.join(REPO, "data", "labels", "nfl_qb_outcomes.csv")

RANDOM_SEED = 20260706

POWER5 = {"SEC", "ACC", "B1G", "PAC", "B12", "Ind"}  # Ind ~= Notre Dame/BYU-era indep.


# --------------------------------------------------------------------------- #
# Candidate feature catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Feat:
    col: str          # column in the profile/feature frame
    label: str        # human-readable name
    group: str        # ppa | combine | pff | context | capital
    higher_is_better: int  # +1 higher -> more likely hit (prior), -1 lower better, 0 unknown


CANDIDATES: list[Feat] = [
    # --- CFBD PPA (all classes 2015-2026) ---------------------------------
    Feat("final_average_ppa_all", "PPA per play (final yr)", "ppa", +1),
    Feat("final_average_ppa_pass", "PPA per pass (final yr)", "ppa", +1),
    Feat("final_average_ppa_passing_downs", "PPA on passing downs (final yr)", "ppa", +1),
    Feat("final_average_ppa_third_down", "PPA on 3rd down (final yr)", "ppa", +1),
    Feat("final_total_ppa_all", "Total PPA production (final yr)", "ppa", +1),
    Feat("final_total_ppa_pass", "Total pass PPA (final yr)", "ppa", +1),
    Feat("career_average_ppa_all", "PPA per play (career)", "ppa", +1),
    Feat("career_average_ppa_pass", "PPA per pass (career)", "ppa", +1),
    Feat("career_total_ppa_all", "Total PPA production (career)", "ppa", +1),
    # --- Combine / physical (2016-2026) -----------------------------------
    # NOTE: hand_size, arm_length, ras_score, speed_score columns exist in the
    # combine table but are 100% empty in the current ingest -> excluded.
    Feat("combine_height", "Height", "combine", +1),
    Feat("combine_weight", "Weight", "combine", +1),
    Feat("combine_bmi", "BMI", "combine", 0),
    Feat("combine_forty", "40-yard dash", "combine", -1),
    Feat("combine_three_cone", "3-cone drill", "combine", -1),
    Feat("combine_vertical", "Vertical jump", "combine", +1),
    Feat("combine_broad", "Broad jump", "combine", +1),
    Feat("speed_score", "Speed score (derived)", "combine", +1),
    # --- PFF charting (classes ~2015-2019 only today) ---------------------
    Feat("final_grades_grades_pass", "PFF passing grade (final yr)", "pff", +1),
    Feat("final_grades_grades_offense", "PFF offense grade (final yr)", "pff", +1),
    Feat("final_grades_accuracy_percent", "PFF accuracy % (final yr)", "pff", +1),
    Feat("final_grades_completion_percent", "Completion % (final yr)", "pff", +1),
    Feat("final_grades_ypa", "Yards per attempt (final yr)", "pff", +1),
    Feat("final_grades_btt_rate", "Big-time-throw rate (final yr)", "pff", +1),
    Feat("final_grades_twp_rate", "Turnover-worthy-play rate (final yr)", "pff", -1),
    Feat("final_grades_avg_depth_of_target", "Avg depth of target (final yr)", "pff", 0),
    Feat("final_grades_avg_time_to_throw", "Avg time to throw (final yr)", "pff", 0),
    Feat("final_grades_pressure_to_sack_rate", "Pressure-to-sack rate (final yr)", "pff", -1),
    Feat("final_grades_sack_percent", "Sack % (final yr)", "pff", -1),
    Feat("final_grades_qb_rating", "Passer rating (final yr)", "pff", +1),
    Feat("final_grades_positive_epa_percent", "Positive-EPA play % (final yr)", "pff", +1),
    Feat("final_grades_grades_run", "PFF rushing grade (final yr)", "pff", +1),
    Feat("career_grades_grades_pass", "PFF passing grade (career)", "pff", +1),
    Feat("career_grades_accuracy_percent", "PFF accuracy % (career)", "pff", +1),
    Feat("career_grades_twp_rate", "Turnover-worthy-play rate (career)", "pff", -1),
    Feat("final_grades_dropbacks", "Dropbacks (final yr, experience)", "pff", +1),
    # --- Context ----------------------------------------------------------
    Feat("power5", "Power-5 conference (final yr)", "context", +1),
    Feat("career_cfbd_seasons", "College seasons played", "context", 0),
    # --- Draft capital (post-draft only) ----------------------------------
    Feat("neg_log_pick", "Draft capital (-log pick)", "capital", +1),
]

CAND_BY_COL = {f.col: f for f in CANDIDATES}


def group_cols(group: str) -> list[str]:
    return [f.col for f in CANDIDATES if f.group == group]


# --------------------------------------------------------------------------- #
# Loading & joining
# --------------------------------------------------------------------------- #
def load_profiles() -> pd.DataFrame:
    return pd.read_parquet(PROFILES_PATH)


def load_labels() -> pd.DataFrame:
    lab = pd.read_csv(LABELS_PATH)
    lab["normalized_name"] = lab["player"].map(normalize_name)
    return lab


def _college_agrees(label_college: str, profile_colleges: str) -> bool:
    lc = str(label_college).lower().split("(")[0].strip()
    pc = str(profile_colleges).lower()
    if not lc or lc == "nan":
        return False
    return (lc[:6] in pc) or (pc.split(",")[0][:6] in lc)


def join_labels_profiles(verbose: bool = False):
    """Return (joined_df, unjoinable_df).

    joined_df: one row per labeled QB that matched a profile, carrying every
    profile column + the label columns + a ``join_method`` column.
    unjoinable_df: labeled QBs with no profile (with a reason).
    """
    labels = load_labels()
    prof = load_profiles()

    # 1) primary join: normalized_name + draft_year == draft_season
    primary = labels.merge(
        prof,
        left_on=["normalized_name", "draft_year"],
        right_on=["normalized_name", "draft_season"],
        how="left",
        suffixes=("", "_prof"),
    )
    matched = primary[primary["canonical_id"].notna()].copy()
    matched["join_method"] = "name+year"

    unmatched = labels[~labels["player"].isin(matched["player"])].copy()

    # 2) fallback: name-only against a profile whose college agrees (handles
    #    PFF-proxy draft_season, e.g. Chad Kelly labeled 2017 but profile 2016)
    prof_by_name = {n: g for n, g in prof.groupby("normalized_name")}
    fb_rows, still_unmatched = [], []
    for _, lr in unmatched.iterrows():
        cand = prof_by_name.get(lr["normalized_name"])
        if cand is not None:
            ok = cand[cand["colleges"].apply(lambda c: _college_agrees(lr["college"], c))]
            if len(ok):
                # prefer the profile with the most data (PFF then CFBD)
                ok = ok.assign(_score=ok["in_pff"].astype(int) * 2 + ok["in_cfbd"].astype(int))
                best = ok.sort_values("_score", ascending=False).iloc[0].drop(labels="_score")
                row = {**lr.to_dict(), **best.to_dict(), "join_method": "name+college"}
                fb_rows.append(row)
                continue
        still_unmatched.append(lr.to_dict())

    if fb_rows:
        matched = pd.concat([matched, pd.DataFrame(fb_rows)], ignore_index=True)

    unjoinable = pd.DataFrame(still_unmatched)
    if len(unjoinable):
        unjoinable = unjoinable[["draft_year", "player", "college", "success_tier",
                                 "hit", "label_status", "normalized_name"]].copy()
        unjoinable["reason"] = "no PFF and no combine seed row (CFBD-only / FCS / non-combine)"

    if verbose:
        print(f"labels={len(labels)}  joined={len(matched)}  unjoinable={len(unjoinable)}")
        by = matched["join_method"].value_counts().to_dict()
        print("  join methods:", by)

    return matched.reset_index(drop=True), unjoinable.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # power-5 flag from CFBD final-season conference. A missing conference in
    # CFBD means no FBS season matched -> treat as NOT power-5 (FCS/small
    # school), rather than NaN-then-impute-to-yes which wrongly credited FCS QBs.
    conf = out["final_cfbd_conference"] if "final_cfbd_conference" in out.columns else pd.Series(np.nan, index=out.index)
    out["power5"] = conf.isin(POWER5).astype(float)
    # derived speed score: (weight * 200) / forty^4  (classic athleticism composite)
    if "combine_weight" in out.columns and "combine_forty" in out.columns:
        w = pd.to_numeric(out["combine_weight"], errors="coerce")
        f = pd.to_numeric(out["combine_forty"], errors="coerce")
        out["speed_score"] = (w * 200.0) / (f ** 4)
    else:
        out["speed_score"] = np.nan
    # draft capital: -log(pick) so higher = better slot (only where pick known)
    if "pick" in out.columns:
        pick = pd.to_numeric(out["pick"], errors="coerce")
        out["neg_log_pick"] = -np.log(pick.clip(lower=1))
    else:
        out["neg_log_pick"] = np.nan
    return out


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Non-null count per candidate feature (helps pick the model feature set)."""
    rows = []
    for f in CANDIDATES:
        n = df[f.col].notna().sum() if f.col in df.columns else 0
        rows.append({"feature": f.col, "label": f.label, "group": f.group, "n_nonnull": int(n)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    joined, unjoinable = join_labels_profiles(verbose=True)
    joined = add_derived(joined)
    print("\n=== unjoinable labeled QBs ===")
    if len(unjoinable):
        print(unjoinable.to_string(index=False))
    print("\n=== final-label training rows ===")
    fin = joined[joined["label_status"] == "final"]
    print(f"n={len(fin)}  hits={int(fin['hit'].sum())}  base_rate={fin['hit'].mean():.3f}")
    print("\n=== feature coverage among joined labels ===")
    print(coverage_report(joined).to_string(index=False))
