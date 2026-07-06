# NFL QB Outcome Labels

`nfl_qb_outcomes.csv` — one row per QB drafted 2015–2026, hand-curated as the
ground-truth "success" labels for the projection model.

## Provenance & caveats

Most outcome labels were **curated from model knowledge (knowledge cutoff:
January 2026)**. The 2026 drafted-QB rows were added from ESPN's draft position
tracker and NFL.com's prospect tracker on 2026-07-06, and intentionally leave NFL
outcome fields blank. These labels cover well-documented public facts (draft
slot, starting tenure, awards, contracts) and should be highly reliable for
famous players, but:

- **Verify against nflverse** (`draft_picks`, `player_stats`) when that data can
  be added to the repo. Pick numbers for late-round picks are the most likely
  place for small errors.
- `seasons_primary_starter` = approximate count of seasons with ~8+ starts.
- The **2026 draft class has draft facts only**. Do not use it for training
  until NFL outcome fields are mature enough to label.

## Columns

| column | meaning |
|---|---|
| `success_tier` | 0–4 career-value judgment (see below) |
| `hit` | 1 if tier ≥ 3 (became at least a quality long-term starter). Empty for projection targets |
| `seasons_primary_starter` | seasons with roughly 8+ starts (approx.) |
| `label_status` | `final` (2015–22, 3+ yrs observed), `provisional` (2023), `projection_target` (2024–26, too early to grade — **exclude from training**) |
| `label_confidence` | curator's confidence in the specific facts for the row |

## Success tiers

- **4 — Franchise/elite**: sustained top-of-league play (Mahomes, Allen, Jackson, Burrow, Herbert, Hurts, Goff, Prescott)
- **3 — Quality starter**: multi-year legitimate NFL starter; earned a second contract / starter money (Wentz, Watson, Mayfield, Darnold, Murray, Tua, Love, Lawrence, Purdy, Stroud)
- **2 — Fringe/bridge starter**: real starting tenure but never established long-term (Winston, Mariota, Trubisky, Brissett, Minshew, Fields, Mac Jones, D. Jones)
- **1 — Backup/journeyman**: stuck in the league without meaningful starting value
- **0 — Washout/bust**: little or no NFL value (including high picks who got starts but provided ~none: Rosen, Kizer, Zach Wilson, Lance)

Training target options for the model: binary `hit`, ordinal `success_tier`,
or continuous `seasons_primary_starter`.
