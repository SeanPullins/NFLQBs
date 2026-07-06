# NFLQBs — College-to-Pro QB Projection

Model + website that projects incoming NFL QBs (recent draft classes) from
their college production, PFF charting data, and athletic testing — trained on
what actually separated NFL hits from busts in the 2015–2023 draft classes.

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
| PFF college QB (6 stat families) | 2014–2015 | player-season | ✅ in repo; **2016–2025 wanted** |
| CFBD player PPA | 2014–2025 | player-season | ✅ in repo |
| Combine + pro-day testing | 2016–2026 draft classes | player | ✅ in repo |
| ESPN QBR | — | player-season | ❌ all uploads 0 bytes (broken export) |
| NFL outcomes (labels) | 2015–2025 classes | player | ✅ hand-curated, see `data/labels/README.md` |

## Quick start

```bash
pip install -r requirements.txt
python3 -m pipeline.build       # raw → processed tables
```
