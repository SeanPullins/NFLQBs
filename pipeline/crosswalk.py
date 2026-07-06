"""
Entity resolution across PFF college grades, CFBD player PPA, and the
combine/pro-day file.

Design
------
Base entities are the union of:
  * each distinct PFF ``player_id`` (grouped across its season rows), and
  * each combine QB row,
one-to-one linked together where they clearly refer to the same person
(same normalized name + a school the PFF player actually attended in some
season). CFBD is treated as an *enrichment* source, not a base entity
source: we look up CFBD rows for an already-built entity rather than
minting new entities purely from CFBD, because CFBD's player universe is
every FBS/FCS QB ever (thousands), most of whom are irrelevant to a draft
model, and because CFBD occasionally reassigns numeric player ids to the
same real person mid-career (a genuine data quirk -- see match_report.md),
which makes CFBD ids untrustworthy as a base entity key on their own.

Matching keys are always (normalized_name, canonical_school[, season]) --
never a bare name -- so same-name-different-player collisions can only
merge if the schools/seasons actually line up. Whenever a name matches more
than one plausible candidate we do NOT guess: the ambiguity is recorded and
both sides are left unmatched for a human to resolve.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from pipeline.normalize import add_normalized_columns, normalize_name, slugify

CFBD_SEASON_WINDOW = 6  # years back from draft_season to still consider a CFBD row, for combine-only entities


def _pff_person_table(pff_df: pd.DataFrame) -> dict[int, dict]:
    """One summary record per PFF player_id."""
    persons: dict[int, dict] = {}
    for player_id, group in pff_df.groupby("player_id"):
        group = group.sort_values("season")
        names = group["normalized_name"].dropna()
        display_names = group["player"].dropna()
        season_school = [
            (int(row.season), row.canonical_school)
            for row in group.itertuples()
            if pd.notna(row.canonical_school)
        ]
        schools = sorted({s for _, s in season_school})
        last_row = group.iloc[-1]
        persons[player_id] = {
            "player_id": player_id,
            "normalized_name": names.iloc[0] if len(names) else None,
            "display_name": display_names.iloc[0] if len(display_names) else None,
            "season_school_pairs": season_school,
            "schools": schools,
            "seasons": sorted({s for s, _ in season_school}),
            "last_season": int(last_row["season"]),
            "last_school": last_row["canonical_school"],
        }
    return persons


def _cfbd_lookup(cfbd_df: pd.DataFrame) -> tuple[dict, dict]:
    """Build a (normalized_name, canonical_school, season) -> [cfbd_id,...]
    lookup, plus a set of ambiguous keys (name+school+season resolving to
    more than one distinct id -- true same-name-same-school-same-season
    collisions, which we refuse to guess on).
    """
    lookup: dict[tuple, set[int]] = defaultdict(set)
    for row in cfbd_df.itertuples():
        if pd.isna(row.normalized_name) or pd.isna(row.canonical_school):
            continue
        key = (row.normalized_name, row.canonical_school, int(row.season))
        lookup[key].add(row.cfbd_id)
    ambiguous_keys = {k for k, ids in lookup.items() if len(ids) > 1}
    return lookup, ambiguous_keys


def _match_combine_to_pff(pff_persons: dict[int, dict], combine_df: pd.DataFrame):
    """Return (combine_idx -> player_id link dict, list of ambiguous-match notes)."""
    # Index PFF persons by (normalized_name, school) for any school they
    # ever played at, so a transfer's final combine school still matches.
    by_name_school: dict[tuple, set[int]] = defaultdict(set)
    by_name: dict[str, set[int]] = defaultdict(set)
    for pid, person in pff_persons.items():
        if not person["normalized_name"]:
            continue
        by_name[person["normalized_name"]].add(pid)
        for school in person["schools"]:
            by_name_school[(person["normalized_name"], school)].add(pid)

    combine_to_pff: dict[int, int] = {}
    ambiguous = []
    for row in combine_df.itertuples():
        idx = row.Index
        if pd.isna(row.normalized_name):
            continue
        candidates = by_name_school.get((row.normalized_name, row.canonical_school), set())
        if len(candidates) == 1:
            combine_to_pff[idx] = next(iter(candidates))
            continue
        if len(candidates) > 1:
            ambiguous.append(
                {
                    "type": "combine_to_pff",
                    "combine_player": row.player,
                    "combine_college": row.college,
                    "candidates": sorted(candidates),
                    "reason": "multiple PFF player_ids share this name+school",
                }
            )
            continue
        # No school-scoped hit. Fall back to name-only, but only trust it
        # when unambiguous (guards against two different real people who
        # share a name at different schools).
        name_candidates = by_name.get(row.normalized_name, set())
        if len(name_candidates) == 1:
            candidate_pid = next(iter(name_candidates))
            person = pff_persons[candidate_pid]
            # Only accept if the combine college isn't a hard contradiction,
            # i.e. we simply have no PFF school overlap on record (fine --
            # PFF only covers 2014-2015 today) rather than a clean mismatch
            # against a fully different, unrelated school history. We still
            # accept because a unique name match across the whole PFF QB
            # corpus is strong signal on its own.
            combine_to_pff[idx] = candidate_pid
        elif len(name_candidates) > 1:
            ambiguous.append(
                {
                    "type": "combine_to_pff_name_only",
                    "combine_player": row.player,
                    "combine_college": row.college,
                    "candidates": sorted(name_candidates),
                    "reason": "name matches multiple PFF player_ids at different schools, "
                    "and none matches the combine college",
                }
            )
    return combine_to_pff, ambiguous


def build_crosswalk(pff_df: pd.DataFrame, cfbd_df: pd.DataFrame, combine_df: pd.DataFrame):
    """Build the QB crosswalk table.

    Returns (crosswalk_df, cfbd_row_matches, report):
      - crosswalk_df: one row per resolved QB entity (CSV-ready).
      - cfbd_row_matches: canonical_id -> set of (season, cfbd_id) rows that
        were matched in CFBD for that entity, at full row granularity (not
        collapsed to a single id) so features.py can pull the exact CFBD
        rows for final/career stats without re-deriving the match logic.
      - report: dict with counts and ambiguous-match lists for
        match_report.md.
    """
    report: dict = {"ambiguous": [], "unmapped_schools": []}

    pff_df = add_normalized_columns(pff_df, "player", "team_name", "pff") if not pff_df.empty else pff_df
    cfbd_df = add_normalized_columns(cfbd_df, "player", "team", "cfbd") if not cfbd_df.empty else cfbd_df
    combine_df = (
        add_normalized_columns(combine_df, "player", "college", "combine") if not combine_df.empty else combine_df
    )

    pff_persons = _pff_person_table(pff_df) if not pff_df.empty else {}
    cfbd_lookup, cfbd_ambiguous_keys = _cfbd_lookup(cfbd_df) if not cfbd_df.empty else ({}, set())

    combine_to_pff, combine_ambiguous = _match_combine_to_pff(pff_persons, combine_df) if not combine_df.empty else ({}, [])
    report["ambiguous"].extend(combine_ambiguous)

    pff_to_combine = {v: k for k, v in combine_to_pff.items()}  # 1:1 by construction

    entities = []

    # 1) PFF-anchored entities (with or without a combine link)
    for pid, person in pff_persons.items():
        combine_idx = pff_to_combine.get(pid)
        entities.append({"pff_player_id": pid, "pff": person, "combine_idx": combine_idx})

    # 2) Combine rows with no PFF link become their own (combine-only) entity
    linked_combine_idx = set(combine_to_pff.keys())
    if not combine_df.empty:
        for row in combine_df.itertuples():
            if row.Index not in linked_combine_idx:
                entities.append({"pff_player_id": None, "pff": None, "combine_idx": row.Index})

    # --- CFBD enrichment per entity -----------------------------------
    cfbd_ambiguous_notes = []

    def cfbd_matches_for_entity(normalized_name, season_school_pairs=None, schools=None, season_range=None):
        """Returns the set of (season, cfbd_id) rows that plausibly belong to
        this entity. Kept at (season, id) granularity -- not collapsed to a
        single id -- so a CFBD-side id-split (see match_report.md) doesn't
        silently drop seasons of real data.

        Two complementary strategies, both keyed on (name, school) so a
        same-name-different-school collision can't merge:
          * season_school_pairs: exact (season, school) rows we have direct
            evidence for (from PFF or combine).
          * schools + season_range: same known school(s), but scanned across
            a padded season window -- needed because CFBD sometimes only has
            partial-career coverage for small/FCS programs (e.g. a player's
            PFF seasons and their one CFBD season for the same school don't
            actually overlap; see Easton Stick in match_report.md).
        """
        matched: set[tuple[int, int]] = set()
        candidate_keys = []
        if season_school_pairs:
            candidate_keys.extend((normalized_name, sch, season) for season, sch in season_school_pairs)
        if schools and season_range is not None:
            candidate_keys.extend((normalized_name, sch, season) for sch in schools for season in season_range)
        for key in candidate_keys:
            if key in cfbd_ambiguous_keys:
                cfbd_ambiguous_notes.append(
                    {
                        "type": "cfbd_ambiguous",
                        "normalized_name": key[0],
                        "school": key[1],
                        "season": key[2],
                        "reason": "multiple distinct CFBD ids for this name+school+season",
                    }
                )
                continue
            ids = cfbd_lookup.get(key)
            if not ids:
                continue
            for cfbd_id in ids:
                matched.add((key[2], cfbd_id))
        return matched

    cfbd_row_matches: dict[str, set[tuple[int, int]]] = {}

    crosswalk_rows = []
    for entity in entities:
        pid = entity["pff_player_id"]
        person = entity["pff"]
        combine_idx = entity["combine_idx"]
        combine_row = combine_df.loc[combine_idx] if combine_idx is not None else None

        if person is not None:
            normalized_name = person["normalized_name"]
            display_name = person["display_name"]
            season_school_pairs = person["season_school_pairs"]
            schools = set(person["schools"])
            last_school = person["last_school"]
            last_season = person["last_season"]
        else:
            normalized_name = combine_row["normalized_name"]
            display_name = combine_row["player"]
            season_school_pairs = []
            schools = set()
            last_school = None
            last_season = None

        draft_season = None
        if combine_row is not None:
            draft_season = int(combine_row["draft_season"])
            if combine_row["canonical_school"]:
                schools.add(combine_row["canonical_school"])
            if last_school is None:
                last_school = combine_row["canonical_school"]

        # Season window for the school-scoped (non-exact-season) CFBD pass:
        # anchored on known PFF seasons when we have them (padded a couple
        # years either side -- covers cases like Easton Stick, whose CFBD
        # row and PFF rows land in different, non-overlapping seasons of the
        # same NDSU career), else anchored on draft_season for combine-only
        # entities (padded CFBD_SEASON_WINDOW years back).
        if person is not None:
            season_range = range(min(person["seasons"]) - 2, max(person["seasons"]) + 5)
        elif combine_row is not None:
            season_range = range(draft_season - CFBD_SEASON_WINDOW, draft_season)
        else:
            season_range = None

        cfbd_row_hits = set()
        if normalized_name:
            cfbd_row_hits = cfbd_matches_for_entity(
                normalized_name,
                season_school_pairs=season_school_pairs or None,
                schools=schools or None,
                season_range=season_range,
            )
        cfbd_ids = {i for _, i in cfbd_row_hits}

        final_school_for_id = last_school or (combine_row["canonical_school"] if combine_row is not None else None)
        id_year = last_season if last_season is not None else draft_season
        canonical_id = slugify(display_name, final_school_for_id, id_year)

        college_seasons = ";".join(f"{sch}:{season}" for season, sch in sorted(season_school_pairs, key=lambda x: x[0]))
        if not draft_season:
            draft_season = (last_season + 1) if last_season is not None else None

        crosswalk_rows.append(
            {
                "canonical_id": canonical_id,
                "canonical_name": display_name,
                "normalized_name": normalized_name,
                "pff_player_id": pid,
                "cfbd_id": ";".join(str(i) for i in sorted(cfbd_ids)) if cfbd_ids else None,
                "combine_normalized_player": combine_row["normalized_player"] if combine_row is not None else None,
                "combine_cfb_player_id": combine_row["cfb_player_id"] if combine_row is not None else None,
                "combine_pfr_player_id": combine_row["pfr_player_id"] if combine_row is not None else None,
                "combine_gsis_id": combine_row["gsis_id"] if combine_row is not None else None,
                "colleges": ";".join(sorted(schools)) if schools else None,
                "college_seasons": college_seasons or None,
                "college_seasons_count": len(person["seasons"]) if person is not None else 0,
                "draft_season": draft_season,
                "in_pff": pid is not None,
                "in_combine": combine_row is not None,
                "in_cfbd": bool(cfbd_ids),
                "_cfbd_row_hits": cfbd_row_hits,
            }
        )

    crosswalk_df = pd.DataFrame(crosswalk_rows)

    # De-duplicate canonical_id collisions (defensive -- shouldn't normally
    # happen since entities are already disjoint people).
    if not crosswalk_df.empty and crosswalk_df["canonical_id"].duplicated().any():
        counts = defaultdict(int)
        new_ids = []
        for cid in crosswalk_df["canonical_id"]:
            counts[cid] += 1
            new_ids.append(cid if counts[cid] == 1 else f"{cid}-{counts[cid]}")
        crosswalk_df["canonical_id"] = new_ids

    cfbd_row_matches = dict(zip(crosswalk_df["canonical_id"], crosswalk_df["_cfbd_row_hits"])) if not crosswalk_df.empty else {}
    if not crosswalk_df.empty:
        crosswalk_df = crosswalk_df.drop(columns=["_cfbd_row_hits"])

    report["ambiguous"].extend(cfbd_ambiguous_notes)
    report["counts"] = {
        "pff_persons": len(pff_persons),
        "combine_rows": 0 if combine_df.empty else len(combine_df),
        "combine_linked_to_pff": len(combine_to_pff),
        "entities_total": len(crosswalk_df),
        "entities_with_pff": int(crosswalk_df["in_pff"].sum()) if not crosswalk_df.empty else 0,
        "entities_with_combine": int(crosswalk_df["in_combine"].sum()) if not crosswalk_df.empty else 0,
        "entities_with_cfbd": int(crosswalk_df["in_cfbd"].sum()) if not crosswalk_df.empty else 0,
    }
    return crosswalk_df, cfbd_row_matches, report
