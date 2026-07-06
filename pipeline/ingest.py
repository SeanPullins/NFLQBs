"""
Raw-data loaders. Each function auto-discovers the files for its source
under data/raw/ and returns a tidy DataFrame. Nothing here does entity
resolution -- that is crosswalk.py's job. These loaders just parse, clean
column names, and stack years.
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd

RAW_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# PFF family file-name prefix -> the column prefix used once families are
# merged together (per the spec: grades_, depth_, pressure_, concept_,
# pocket_, allowed_).
PFF_FAMILIES = {
    "passing-grades": "grades",
    "passing-depth": "depth",
    "passing-pressure": "pressure",
    "passing-concept": "concept",
    "time-in-pocket": "pocket",
    "allowed-pressure": "allowed",
}

# Columns that identify a player-season and are shared verbatim across all
# six PFF families (not prefixed, deduplicated on merge).
PFF_IDENTITY_COLS = ["player", "player_id", "position", "team_name", "player_game_count", "franchise_id"]


def discover_pff_years(raw_root: str = RAW_ROOT) -> list[int]:
    """Auto-discover PFF season years by globbing data/raw/pff/*/."""
    pattern = os.path.join(raw_root, "pff", "*")
    years = []
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            base = os.path.basename(path.rstrip("/"))
            if re.fullmatch(r"\d{4}", base):
                years.append(int(base))
    return sorted(years)


def _load_pff_family_file(year: int, family: str, raw_root: str = RAW_ROOT) -> pd.DataFrame | None:
    path = os.path.join(raw_root, "pff", str(year), f"{family}__QB__{year}.csv")
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def load_pff_season(year: int, raw_root: str = RAW_ROOT) -> tuple[pd.DataFrame, dict]:
    """Load and merge all six PFF families for one season into one
    player-season table. Returns (df, load_notes) where load_notes records
    which family files were missing/empty for this year (useful for the
    match report).
    """
    identity_frames = []
    metric_frames = []
    notes = {"year": year, "families_loaded": [], "families_missing": []}

    for family, prefix in PFF_FAMILIES.items():
        raw = _load_pff_family_file(year, family, raw_root)
        if raw is None:
            notes["families_missing"].append(family)
            continue
        notes["families_loaded"].append(family)

        identity_cols = [c for c in PFF_IDENTITY_COLS if c in raw.columns]
        identity_frames.append(raw[identity_cols].copy())

        metric_cols = [c for c in raw.columns if c not in PFF_IDENTITY_COLS]
        metrics = raw[["player_id"] + metric_cols].copy()
        metrics = metrics.rename(columns={c: f"{prefix}_{c}" for c in metric_cols})
        metric_frames.append(metrics)

    if not identity_frames:
        return pd.DataFrame(), notes

    # Build one identity row per player_id, preferring the first family that
    # had non-null values (families agree on team_name in practice, see
    # match report for the rare cases that don't).
    identity = pd.concat(identity_frames, ignore_index=True)
    identity = identity.groupby("player_id", as_index=False).first()

    merged = identity
    for metrics in metric_frames:
        merged = merged.merge(metrics, on="player_id", how="outer")

    merged.insert(1, "season", year)
    return merged, notes


def load_pff_all(raw_root: str = RAW_ROOT) -> tuple[pd.DataFrame, list[dict]]:
    """Load every discovered PFF season and stack them into one player-season
    table (one row per player_id, season)."""
    years = discover_pff_years(raw_root)
    frames = []
    all_notes = []
    for year in years:
        df, notes = load_pff_season(year, raw_root)
        all_notes.append(notes)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), all_notes
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, all_notes


# ---------------------------------------------------------------------------
# CFBD player PPA
# ---------------------------------------------------------------------------

_CFBD_COLUMN_RENAMES = {
    "id": "cfbd_id",
    "name": "player",
    "averagePPA.all": "average_ppa_all",
    "averagePPA.pass": "average_ppa_pass",
    "averagePPA.rush": "average_ppa_rush",
    "averagePPA.firstDown": "average_ppa_first_down",
    "averagePPA.secondDown": "average_ppa_second_down",
    "averagePPA.thirdDown": "average_ppa_third_down",
    "averagePPA.standardDowns": "average_ppa_standard_downs",
    "averagePPA.passingDowns": "average_ppa_passing_downs",
    "totalPPA.all": "total_ppa_all",
    "totalPPA.pass": "total_ppa_pass",
    "totalPPA.rush": "total_ppa_rush",
    "totalPPA.firstDown": "total_ppa_first_down",
    "totalPPA.secondDown": "total_ppa_second_down",
    "totalPPA.thirdDown": "total_ppa_third_down",
    "totalPPA.standardDowns": "total_ppa_standard_downs",
    "totalPPA.passingDowns": "total_ppa_passing_downs",
}


def discover_cfbd_years(raw_root: str = RAW_ROOT) -> list[int]:
    pattern = os.path.join(raw_root, "cfbd_ppa", "player_ppa_season_*.csv")
    years = []
    for path in glob.glob(pattern):
        m = re.search(r"player_ppa_season_(\d{4})\.csv$", path)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def load_cfbd_ppa(raw_root: str = RAW_ROOT) -> pd.DataFrame:
    """Load all CFBD player PPA season files and filter to QBs.

    Handles a known raw-data quirk: some (season, id) rows are duplicated
    with only the `conference` field differing (populated vs NaN) -- we keep
    the version with a populated conference.
    """
    pattern = os.path.join(raw_root, "cfbd_ppa", "player_ppa_season_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["position"] == "QB"].copy()

    df = df.sort_values(by="conference", key=lambda s: s.isna()).drop_duplicates(
        subset=["season", "id", "name", "team"], keep="first"
    )
    df = df.drop_duplicates(subset=["season", "id"], keep="first")

    df = df.rename(columns=_CFBD_COLUMN_RENAMES)
    df = df.sort_values(["player", "season"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Combine / pro day
# ---------------------------------------------------------------------------

def load_combine(raw_root: str = RAW_ROOT) -> pd.DataFrame:
    pattern = os.path.join(raw_root, "combine", "combine_pro_day_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["position"] == "QB"].copy()
    df = df.drop_duplicates()
    df = df.sort_values(["draft_season", "player"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# ESPN QBR (stub -- files are expected to be empty until a later data drop)
# ---------------------------------------------------------------------------

def load_espn_qbr(raw_root: str = RAW_ROOT) -> tuple[pd.DataFrame, list[str]]:
    """Load ESPN QBR season files if/when they show up with real data.

    As of this writing data/raw/espn_qbr/ only contains a README -- the
    qbr_season_YYYY.csv exports were uploaded empty. This loader auto
    -discovers qbr_season_*.csv, tolerates missing/zero-byte files, and
    returns an empty frame (plus a list of skip reasons) until real data
    arrives. Once populated files show up, rerunning the pipeline picks
    them up automatically -- no code changes needed here.
    """
    pattern = os.path.join(raw_root, "espn_qbr", "qbr_season_*.csv")
    paths = sorted(glob.glob(pattern))
    skipped = []
    frames = []
    for path in paths:
        if os.path.getsize(path) == 0:
            skipped.append(f"{os.path.basename(path)} (0 bytes)")
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            skipped.append(f"{os.path.basename(path)} (no columns to parse)")
            continue
        if df.empty:
            skipped.append(f"{os.path.basename(path)} (empty)")
            continue
        m = re.search(r"qbr_season_(\d{4})\.csv$", path)
        if m and "season" not in df.columns:
            df["season"] = int(m.group(1))
        frames.append(df)

    if not paths:
        skipped.append("no qbr_season_*.csv files found under data/raw/espn_qbr/")

    if not frames:
        return pd.DataFrame(), skipped
    return pd.concat(frames, ignore_index=True, sort=False), skipped
