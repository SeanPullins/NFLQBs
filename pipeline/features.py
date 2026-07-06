"""
Feature construction: turns the tidy per-source tables plus the crosswalk
into one row per resolved QB entity (qb_draft_profiles).

For each entity we attach:
  - final college season PFF stats (prefixed final_)
  - career college PFF aggregates, weighted by season dropbacks (prefixed career_)
  - final season + career CFBD PPA
  - all combine/testing measurables (prefixed combine_)
  - college(s), conference of final season, draft_season, season counts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PFF_NON_METRIC_COLS = {"player_id", "season", "player", "position", "team_name", "franchise_id", "normalized_name", "canonical_school"}
PFF_WEIGHT_CANDIDATES = ["grades_dropbacks", "pocket_dropbacks", "pressure_base_dropbacks", "concept_dropbacks"]

CFBD_NON_METRIC_COLS = {"season", "cfbd_id", "player", "position", "team", "conference", "normalized_name", "canonical_school"}
CFBD_TOTAL_COLS_HINT = "total_ppa"  # total_ppa_* columns are summed across career; average_ppa_* are averaged

COMBINE_KEY_COLS = {
    "player", "normalized_player", "pfr_player_id", "gsis_id", "cfb_player_id", "draft_season", "position",
    "college", "normalized_name", "canonical_school",
}


def _pff_final_and_career(pff_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pff_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    numeric_cols = [c for c in pff_df.select_dtypes(include="number").columns if c not in PFF_NON_METRIC_COLS]

    # --- final season -----------------------------------------------------
    final = pff_df.sort_values("season").groupby("player_id", as_index=False).tail(1).copy()
    rename_map = {c: f"final_{c}" for c in pff_df.columns if c not in {"player_id"}}
    final = final.rename(columns=rename_map)

    # --- career, dropback-weighted -----------------------------------------
    weight = pd.Series(np.nan, index=pff_df.index, dtype="float64")
    for col in PFF_WEIGHT_CANDIDATES:
        if col in pff_df.columns:
            weight = weight.combine_first(pff_df[col].astype("float64"))
    weight = weight.fillna(0.0)

    numeric = pff_df[numeric_cols].astype("float64")
    mask = numeric.notna()
    filled = numeric.fillna(0.0)

    weighted_vals = filled.mul(weight, axis=0)
    weight_mask = mask.mul(weight, axis=0)

    player_id = pff_df["player_id"]
    num_sum = weighted_vals.groupby(player_id).sum()
    den_sum = weight_mask.groupby(player_id).sum()
    simple_mean = numeric.groupby(player_id).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_mean = num_sum / den_sum
    career_numeric = weighted_mean.where(den_sum > 0, simple_mean)
    career_numeric = career_numeric.rename(columns={c: f"career_{c}" for c in career_numeric.columns})

    career_meta = pd.DataFrame(
        {
            "career_college_seasons": pff_df.groupby("player_id")["season"].nunique(),
            "career_first_season": pff_df.groupby("player_id")["season"].min(),
            "career_last_season": pff_df.groupby("player_id")["season"].max(),
            "career_games_played_sum": pff_df.groupby("player_id")["player_game_count"].sum(min_count=1)
            if "player_game_count" in pff_df.columns
            else np.nan,
            "career_weight_dropbacks": weight.groupby(player_id).sum(),
        }
    )

    career = pd.concat([career_meta, career_numeric], axis=1).reset_index().copy()
    return final, career


def _cfbd_final_and_career(cfbd_df: pd.DataFrame, cfbd_row_matches: dict) -> pd.DataFrame:
    """Build one row per canonical_id of final-season + career CFBD PPA,
    using the exact (season, cfbd_id) rows crosswalk.py matched -- not a
    bare id join -- so CFBD's occasional id-reassignment quirk doesn't drop
    real seasons of data (see match_report.md)."""
    if cfbd_df.empty or not cfbd_row_matches:
        return pd.DataFrame()

    numeric_cols = [c for c in cfbd_df.select_dtypes(include="number").columns if c not in CFBD_NON_METRIC_COLS]
    total_cols = [c for c in numeric_cols if c.startswith("total_ppa")]
    avg_cols = [c for c in numeric_cols if c.startswith("average_ppa")]

    key_to_rows = cfbd_df.set_index(["season", "cfbd_id"])

    records = []
    for canonical_id, hits in cfbd_row_matches.items():
        if not hits:
            continue
        keys = [k for k in hits if k in key_to_rows.index]
        if not keys:
            continue
        rows = key_to_rows.loc[keys]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        rows = rows.sort_values("season") if "season" in rows.columns else rows.reset_index().sort_values("season")
        if "season" not in rows.columns:
            rows = rows.reset_index()
        last = rows.iloc[-1]

        rec = {"canonical_id": canonical_id, "final_cfbd_season": int(last["season"]), "final_cfbd_team": last.get("team"), "final_cfbd_conference": last.get("conference")}
        for c in avg_cols + total_cols:
            rec[f"final_{c}"] = last.get(c)

        rec["career_cfbd_seasons"] = rows["season"].nunique()
        for c in avg_cols:
            rec[f"career_{c}"] = rows[c].mean()
        for c in total_cols:
            rec[f"career_{c}"] = rows[c].sum(min_count=1)
        records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _combine_features(combine_df: pd.DataFrame) -> pd.DataFrame:
    if combine_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in combine_df.columns if c not in COMBINE_KEY_COLS]
    out = combine_df[["normalized_player"] + metric_cols].copy()
    out = out.rename(columns={c: f"combine_{c}" for c in metric_cols})
    return out


def build_qb_draft_profiles(
    pff_df: pd.DataFrame,
    cfbd_df: pd.DataFrame,
    combine_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    cfbd_row_matches: dict,
) -> pd.DataFrame:
    profiles = crosswalk_df.copy()

    # pff_final/pff_career are keyed by player_id; crosswalk uses pff_player_id
    pff_final, pff_career = _pff_final_and_career(pff_df)
    if not pff_final.empty:
        pff_final = pff_final.rename(columns={"player_id": "pff_player_id"})
        profiles = profiles.merge(pff_final, on="pff_player_id", how="left")

    if not pff_career.empty:
        pff_career = pff_career.rename(columns={"player_id": "pff_player_id"})
        profiles = profiles.merge(pff_career, on="pff_player_id", how="left")

    cfbd_feats = _cfbd_final_and_career(cfbd_df, cfbd_row_matches)
    if not cfbd_feats.empty:
        profiles = profiles.merge(cfbd_feats, on="canonical_id", how="left")

    combine_feats = _combine_features(combine_df)
    if not combine_feats.empty:
        profiles = profiles.merge(
            combine_feats, left_on="combine_normalized_player", right_on="normalized_player", how="left"
        )
        if "normalized_player" in profiles.columns:
            profiles = profiles.drop(columns=["normalized_player"])

    return profiles.copy()  # defragment after many sequential merges
