"""
Pipeline orchestrator.

    python3 -m pipeline.build

Ingests every raw college-QB source under data/raw/, normalizes names and
schools, resolves entities across sources, builds the tidy processed
tables under data/processed/, and prints a sanity-check summary. Re-running
this after new PFF years (2016-2025) or ESPN QBR files land requires no
code changes -- everything is discovered by glob.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

from pipeline import crosswalk, features, ingest, normalize

# Building qb_draft_profiles does many sequential merges onto a very wide
# (2000+ column) DataFrame; pandas warns about fragmentation but it's a
# performance note, not a correctness issue -- the frame is explicitly
# defragmented (.copy()) right before being written out.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROCESSED_DIR, "reports")

SPOT_CHECK_QBS = [
    ("Jared Goff", "California"),
    ("Carson Wentz", "North Dakota State"),
    ("Patrick Mahomes", "Texas Tech"),
    ("Deshaun Watson", "Clemson"),
]


def _save_table(df: pd.DataFrame, name: str) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    csv_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    parquet_path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive, pyarrow should be present
        print(f"  [warn] could not write {parquet_path}: {exc}")
    print(f"  wrote {csv_path} ({len(df)} rows, {len(df.columns)} cols)")


def main() -> None:
    print("=" * 78)
    print("NFL QB college data pipeline")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    pff_years = ingest.discover_pff_years()
    cfbd_years = ingest.discover_cfbd_years()
    print(f"\nDiscovered PFF years: {pff_years}")
    print(f"Discovered CFBD years: {cfbd_years}")

    pff_df, pff_load_notes = ingest.load_pff_all()
    cfbd_df = ingest.load_cfbd_ppa()
    combine_df = ingest.load_combine()
    espn_df, espn_skip_notes = ingest.load_espn_qbr()

    print(f"\nRaw PFF QB player-seasons: {len(pff_df)}")
    print(f"Raw CFBD QB player-seasons: {len(cfbd_df)}")
    print(f"Raw combine QB rows: {len(combine_df)}")
    print(f"ESPN QBR rows: {len(espn_df)} ({len(espn_skip_notes)} file(s) skipped)")

    # ------------------------------------------------------------------
    # 2. Normalize + write per-source processed tables
    # ------------------------------------------------------------------
    print("\nWriting per-source processed tables...")

    pff_out = normalize.add_normalized_columns(pff_df, "player", "team_name", "pff") if not pff_df.empty else pff_df
    _save_table(pff_out, "pff_qb_college")

    cfbd_out = normalize.add_normalized_columns(cfbd_df, "player", "team", "cfbd") if not cfbd_df.empty else cfbd_df
    _save_table(cfbd_out, "cfbd_ppa_qb")

    combine_out = (
        normalize.add_normalized_columns(combine_df, "player", "college", "combine") if not combine_df.empty else combine_df
    )
    _save_table(combine_out, "combine_qb")

    # ------------------------------------------------------------------
    # 3. Entity resolution
    # ------------------------------------------------------------------
    print("\nResolving QB entities across sources...")
    crosswalk_df, cfbd_row_matches, report = crosswalk.build_crosswalk(pff_df, cfbd_df, combine_df)
    _save_table(crosswalk_df, "crosswalk_qb")

    # ------------------------------------------------------------------
    # 4. Feature construction
    # ------------------------------------------------------------------
    print("\nBuilding qb_draft_profiles...")
    profiles_df = features.build_qb_draft_profiles(pff_df, cfbd_df, combine_df, crosswalk_df, cfbd_row_matches)
    _save_table(profiles_df, "qb_draft_profiles")

    # ------------------------------------------------------------------
    # 5. Match report
    # ------------------------------------------------------------------
    _write_match_report(pff_load_notes, espn_skip_notes, report, pff_df, cfbd_df, combine_df, crosswalk_df)

    # ------------------------------------------------------------------
    # 6. Sanity checks / spot checks
    # ------------------------------------------------------------------
    _print_sanity_summary(pff_df, cfbd_df, combine_df, crosswalk_df, profiles_df, report)


def _write_match_report(pff_load_notes, espn_skip_notes, report, pff_df, cfbd_df, combine_df, crosswalk_df) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "match_report.md")

    lines = []
    lines.append("# QB data match report\n")
    lines.append("Generated by `python3 -m pipeline.build`.\n")

    lines.append("## Row counts per source\n")
    lines.append(f"- PFF QB player-seasons: **{len(pff_df)}**")
    lines.append(f"- CFBD QB player-seasons: **{len(cfbd_df)}**")
    lines.append(f"- Combine QB rows: **{len(combine_df)}**")
    lines.append("")

    lines.append("## PFF family coverage by year\n")
    lines.append("| year | families loaded | families missing |")
    lines.append("|---|---|---|")
    for note in pff_load_notes:
        loaded = ", ".join(note["families_loaded"]) or "(none)"
        missing = ", ".join(note["families_missing"]) or "(none)"
        lines.append(f"| {note['year']} | {loaded} | {missing} |")
    lines.append("")
    lines.append(
        "Note: `allowed-pressure` is byte-identical to `passing-grades` in both 2014 and 2015 "
        "(same columns, same values for every player) -- looks like a duplicated export on PFF's "
        "side rather than a distinct allowed-pressure metric. Kept as its own `allowed_` family "
        "per spec, but treat `allowed_*` columns as redundant with `grades_*` until PFF ships "
        "real allowed-pressure data.\n"
    )

    lines.append("## ESPN QBR\n")
    if espn_skip_notes:
        for note in espn_skip_notes:
            lines.append(f"- skipped: {note}")
    else:
        lines.append("- no issues")
    lines.append(
        "- QBR ingestion is a stub (`pipeline.ingest.load_espn_qbr`): it auto-discovers "
        "`data/raw/espn_qbr/qbr_season_*.csv` and will activate automatically once those files "
        "contain real rows. No pipeline changes needed when that data arrives.\n"
    )

    counts = report.get("counts", {})
    lines.append("## Entity resolution summary\n")
    lines.append(f"- Distinct PFF QB persons (by player_id): **{counts.get('pff_persons', 0)}**")
    lines.append(f"- Combine QB rows: **{counts.get('combine_rows', 0)}**")
    lines.append(f"- Combine rows linked to a PFF player: **{counts.get('combine_linked_to_pff', 0)}**")
    lines.append(f"- Total resolved QB entities: **{counts.get('entities_total', 0)}**")
    lines.append(f"  - with PFF data: {counts.get('entities_with_pff', 0)}")
    lines.append(f"  - with combine data: {counts.get('entities_with_combine', 0)}")
    lines.append(f"  - with CFBD PPA data: {counts.get('entities_with_cfbd', 0)}")
    lines.append(
        "\nNote on the combine -> PFF rate below: PFF only covers 2014-2015 today, so only the 2016 "
        "combine class (final college season 2015) can fully match, plus any player from a later "
        "class who happened to log an early-career (backup/freshman) season in 2014-2015. Expect "
        "this rate to climb automatically as PFF 2016-2025 files are added -- no code changes "
        "needed.\n"
    )
    if counts.get("combine_rows"):
        rate = 100.0 * counts.get("combine_linked_to_pff", 0) / counts["combine_rows"]
        lines.append(f"\nCombine -> PFF match rate: **{rate:.1f}%** ({counts.get('combine_linked_to_pff', 0)}/{counts['combine_rows']})")
        cfbd_hits = crosswalk_df[crosswalk_df["in_combine"]]["in_cfbd"].sum() if not crosswalk_df.empty else 0
        rate_cfbd = 100.0 * cfbd_hits / counts["combine_rows"]
        lines.append(f"\nCombine -> CFBD match rate: **{rate_cfbd:.1f}%** ({cfbd_hits}/{counts['combine_rows']})")
    lines.append("")

    if not crosswalk_df.empty:
        missing_cfbd = crosswalk_df[(crosswalk_df["in_combine"]) & (~crosswalk_df["in_cfbd"])]
        lines.append(f"### Combine QBs with no CFBD PPA match ({len(missing_cfbd)})\n")
        if missing_cfbd.empty:
            lines.append("None -- every combine QB has at least one matched CFBD season.\n")
        else:
            lines.append(
                "Not necessarily errors -- some of these are real FCS/small-school gaps in CFBD's "
                "coverage (e.g. tiny programs) or true PPA-data gaps for that player's seasons. "
                "Others may be a first-name variant our nickname table doesn't cover yet "
                "(`pipeline/normalize.py:_NICKNAMES`) or a school alias gap -- worth a human "
                "glance either way.\n"
            )
            lines.append("| player | college(s) | draft_season | in_pff |")
            lines.append("|---|---|---|---|")
            for _, r in missing_cfbd.sort_values("draft_season").iterrows():
                lines.append(f"| {r['canonical_name']} | {r['colleges']} | {int(r['draft_season']) if pd.notna(r['draft_season']) else ''} | {r['in_pff']} |")
            lines.append("")

    ambiguous = report.get("ambiguous", [])
    lines.append(f"## Ambiguous / unmatched cases requiring human review ({len(ambiguous)})\n")
    if not ambiguous:
        lines.append("None found.\n")
    else:
        lines.append(
            "These were intentionally **left unmatched** rather than guessed. Each entry below "
            "kept its own entity (or, for CFBD ambiguity, simply did not get that season's CFBD "
            "row attached).\n"
        )
        for item in ambiguous:
            lines.append(f"- {item}")
        lines.append("")

    unmapped = sorted(normalize.UNMAPPED_SCHOOLS)
    lines.append(f"## School names with no explicit alias entry ({len(unmapped)})\n")
    if not unmapped:
        lines.append("None -- every school seen in this run has an explicit alias in `pipeline/school_aliases.py`.\n")
    else:
        lines.append(
            "These fell back to a best-effort auto-normalization. If they represent a real school "
            "(not already covered under a different spelling), add them to `pipeline/school_aliases.py` "
            "explicitly so future matching stays precise.\n"
        )
        for source, raw in unmapped:
            lines.append(f"- `{source}`: {raw!r} -> {normalize.get_canonical_school(raw, source)!r}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  wrote {path}")


def _print_sanity_summary(pff_df, cfbd_df, combine_df, crosswalk_df, profiles_df, report) -> None:
    print("\n" + "=" * 78)
    print("SANITY CHECK SUMMARY")
    print("=" * 78)

    print(f"\nRow counts:")
    print(f"  pff_qb_college:     {len(pff_df)}")
    print(f"  cfbd_ppa_qb:        {len(cfbd_df)}")
    print(f"  combine_qb:         {len(combine_df)}")
    print(f"  crosswalk_qb:       {len(crosswalk_df)}")
    print(f"  qb_draft_profiles:  {len(profiles_df)}")

    print(f"\nEntity resolution:")
    counts = report.get("counts", {})
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  ambiguous/unmatched cases logged: {len(report.get('ambiguous', []))}")
    print(f"  schools needing an explicit alias: {len(normalize.UNMAPPED_SCHOOLS)}")

    if not crosswalk_df.empty and "draft_season" in crosswalk_df.columns:
        print(
            "\nQB counts per draft class (note: classes with few/no combine rows, e.g. most PFF-only "
            "entities, use the fallback draft_season = last college season + 1 -- these are NOT real "
            "NFL draft classes, just every QB with data for that would-be season; combine-anchored "
            "classes -- 2016 onward, one real class per year -- are the trustworthy draft-class counts):"
        )
        by_class = crosswalk_df.dropna(subset=["draft_season"]).groupby("draft_season").size().sort_index()
        combine_by_class = (
            crosswalk_df[crosswalk_df["in_combine"]].dropna(subset=["draft_season"]).groupby("draft_season").size()
        )
        for season, n in by_class.items():
            n_combine = int(combine_by_class.get(season, 0))
            print(f"  {int(season)}: {n} total ({n_combine} from actual combine data)")

    print("\nSpot checks:")
    for name, school in SPOT_CHECK_QBS:
        norm_name = normalize.normalize_name(name)
        matches = crosswalk_df[crosswalk_df["normalized_name"] == norm_name] if not crosswalk_df.empty else pd.DataFrame()
        if matches.empty:
            print(f"  [MISSING] {name} ({school}) -- not found in crosswalk_qb at all")
            continue
        for _, row in matches.iterrows():
            profile = profiles_df[profiles_df["canonical_id"] == row["canonical_id"]]
            has_pff = bool(row.get("in_pff"))
            has_combine = bool(row.get("in_combine"))
            has_cfbd = bool(row.get("in_cfbd"))
            n_profile_cols_nonnull = int(profile.notna().sum(axis=1).iloc[0]) if not profile.empty else 0
            print(
                f"  [OK] {name}: canonical_id={row['canonical_id']!r} colleges={row.get('colleges')!r} "
                f"draft_season={row.get('draft_season')} in_pff={has_pff} in_combine={has_combine} "
                f"in_cfbd={has_cfbd} profile_nonnull_cols={n_profile_cols_nonnull}/{len(profiles_df.columns) if not profiles_df.empty else 0}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
