# Model Audit After Full PFF QB Import

Updated 2026-07-06 after importing `QB_2014.zip` through `QB_2025.zip`.

## Bottom line

Claude's first model was materially stale because it was built around the old
assumption that PFF QB data only covered 2014-2015 source seasons. After the full
PFF import, the raw PFF table grew from 1,204 to 7,599 player-seasons, and all
126 labeled drafted QBs now join to a profile.

The surprise: PFF helps, but only a little pre-draft, and it should not be used
in the default post-draft model once draft capital is known.

## Validation Results

Headline validation is now forward-by-draft-year on final labels only. For a
held-out year, the model trains only on earlier draft years. This avoids the old
leave-one-class-out issue where future draft classes helped predict the past.

| model | forward AUC | log loss | Brier | note |
|---|---:|---:|---:|---|
| Pick only | 0.8365 | 0.3947 | 0.1260 | draft slot baseline |
| Pre-draft, no PFF | 0.7734 | 0.4334 | 0.1396 | old feature family |
| Pre-draft, PFF-enabled | 0.7802 | 0.4443 | 0.1447 | small AUC lift, worse calibration |
| Post-draft, no PFF | 0.8709 | 0.3668 | 0.1172 | selected post-draft model |
| Post-draft, PFF-enabled | 0.8297 | 0.3834 | 0.1220 | audit only |

## What Changed Most

Biggest PFF bumps on the projection board:

| QB | class | PFF model delta |
|---|---:|---:|
| Kyle McCord | 2025 | +19.8 pts |
| Michael Penix Jr. | 2024 | +10.7 pts |
| Devin Leary | 2024 | +10.3 pts |
| Jalon Daniels | 2026 watchlist | +8.9 pts |
| Cam Ward | 2025 | +8.1 pts |
| Stetson Bennett | 2023 | +7.4 pts |
| Clayton Tune | 2023 | +7.0 pts |

Biggest PFF downgrades:

| QB | class | PFF model delta |
|---|---:|---:|
| Dillon Gabriel | 2025 | -28.8 pts |
| Cam Miller | 2025 | -21.6 pts |
| Bo Nix | 2024 | -16.7 pts |
| Carson Beck | 2026 watchlist | -9.6 pts |
| Jaxson Dart | 2025 | -9.4 pts |
| Cole Payton | 2026 watchlist | -8.9 pts |
| C.J. Stroud | 2023 | -8.5 pts |
| Jayden Daniels | 2024 | -7.9 pts |

## Caveats

- `allowed-pressure` appears redundant/suspect and is excluded from model
  features.
- Some PFF families are missing or empty by year. The model uses concept and
  pressure features because they cover the modern classes better than the
  top-level passing-grades file.
- PFF improves the pre-draft AUC only slightly and worsens post-draft validation.
  Treat the PFF delta as a useful audit signal, not as proof that the PFF model
  is automatically better for every player.
- 2026 rows are still a watchlist because the repo does not have a verified 2026
  drafted-QB label file yet.
