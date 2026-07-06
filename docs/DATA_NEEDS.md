# Data still needed (as of 2026-07-06)

Status of each data source and what would improve the model most, in priority order.

## 1. PFF college QB files for seasons 2016–2025 (HIGH — biggest win)
You've uploaded `QB_2014.zip` and `QB_2015.zip` (6 CSVs each: passing-grades,
passing-depth, passing-pressure, passing-concept, time-in-pocket,
allowed-pressure). The same exports for **2016–2025** would let the model use
PFF grades/stability metrics for draft classes 2017–2026 — right now PFF
features only cover the 2015–2016 classes.

Drop them in as `data/raw/pff/<season>/<family>__QB__<season>.csv` (or just
upload the zips — the pipeline auto-discovers season folders).

## 2. ESPN QBR files (MEDIUM — currently broken export)
**Every `qbr_season_*.csv` uploaded so far (2014–2025 batch, then 2017–2020
re-upload) is 0 bytes.** The export step itself is producing empty files —
check file sizes before uploading next time. If the source keeps failing, tell
me what tool/site you're exporting from and I'll suggest a fix. QBR is
nice-to-have (opponent-adjusted efficiency), not blocking.

## 3. 2026 draft results (MEDIUM)
The 2026 NFL draft happened after my knowledge cutoff (Jan 2026). To put draft
capital next to 2026-class projections, upload a small CSV:
`player, college, round, pick, team` for 2026 drafted QBs.

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
