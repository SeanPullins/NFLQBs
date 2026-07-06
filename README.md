# NFLQBs — College-to-Pro QB Projection

Live site: https://seanpullins.github.io/NFLQBs/

Model + website that projects incoming NFL QBs and watchlist prospects from
their college production, PFF charting data, athletic testing, and draft capital
when available. The 2027 class is an early watchlist scored from through-2025
college data only.

## Repo layout

```
data/raw/        # uploaded source data (PFF college QB, CFBD PPA, combine/pro-day, ESPN QBR)
data/labels/     # curated NFL outcome labels (the "success" ground truth)
data/processed/  # pipeline outputs: tidy tables, crosswalk, one-row-per-QB draft profiles
pipeline/        # re-runnable ingestion/normalization/join code  → python3 -m pipeline.build
analysis/        # EDA + success-indicator findings
model/           # training, evaluation, projections
site/            # static website (projection board + QB profiles)
docs/            # DATA_NEEDS.md — what's still missing and why it matters
```

## Data sources

| source | seasons | grain | status |
|---|---|---|---|
| PFF college QB (6 stat families) | 2014–2025 | player-season | ✅ in repo |
| CFBD player PPA | 2014–2025 | player-season | ✅ in repo |
| Combine + pro-day testing | 2016–2026 draft classes | player | ✅ in repo |
| ESPN QBR | — | player-season | ❌ all uploads 0 bytes (broken export) |
| NFL outcomes (labels) | 2015–2026 classes | player | ✅ hand-curated, see `data/labels/README.md` |
| 2027 QB watchlist | 2027 draft cycle | player | ✅ early projections, no draft capital yet |

## Quick start

```bash
pip install -r requirements.txt
python3 -m pipeline.build       # raw → processed tables
```
