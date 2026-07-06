# Data still needed (as of 2026-07-06)

Status of each data source and what would improve the model most, in priority order.

## 1. PFF college QB files for seasons 2014–2025 (DONE — now imported)
The `QB_2014.zip` through `QB_2025.zip` exports were imported into
`data/raw/pff/<season>/` and now feed the pipeline. To refresh this data later,
drop the zip files in a local folder and run:

```bash
python3 scripts/import_pff_qb_zips.py /path/to/pff_zips
python3 -m pipeline.build
python3 -m model.train
python3 -m model.predict
```

Important PFF caveats:
- Some family exports are missing or empty by year, including empty
  `passing-depth` in 2014, empty `passing-pressure` in 2021, and empty
  `passing-grades` in 2025.
- `allowed-pressure` appears redundant/suspect and is not used as a model
  signal.
- The PFF-enabled model uses selected concept/pressure features because those
  cover modern classes better than the top-level `passing-grades` file.

## 2. ESPN QBR files (MEDIUM — currently broken export)
**Every `qbr_season_*.csv` uploaded so far (2014–2025 batch, then 2017–2020
re-upload) is 0 bytes.** The export step itself is producing empty files —
check file sizes before uploading next time. If the source keeps failing, tell
me what tool/site you're exporting from and I'll suggest a fix. QBR is
nice-to-have (opponent-adjusted efficiency), not blocking.

## 3. 2026 draft results (DONE — drafted QBs added)
The 2026 drafted QB class is now in `data/labels/nfl_qb_outcomes.csv` as
`projection_target` rows with real round/pick/team data and blank NFL outcome
fields. The board uses a draft-adjusted display score for these rows:
`max(post_draft_model, pick_only_market_baseline)`.

## 4. NFL outcome verification data (LOW — have curated substitute)
`data/labels/nfl_qb_outcomes.csv` is hand-curated from knowledge. To verify /
auto-update it, either upload nflverse `draft_picks.csv` + seasonal QB stats,
or add the `nflverse/nflverse-data` GitHub repo to this session (network policy
blocks me from fetching it myself).

## 5. Optional enrichers (LOW)
- College team context: SP+/FEI team ratings, offensive line grades (isolates QB from supporting cast)
- Recruiting data (247 composite) — age/pedigree features
- QB age at draft, breakout age
- Senior Bowl / East-West participation
