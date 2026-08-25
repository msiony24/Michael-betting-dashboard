# Tuesday findings: extending tennis_core_predictions.csv and re-running Phase 2

## What was reconstructed

The Phase 1 script that builds `tennis_core_predictions.csv` is still missing
from the repo (see PHASE1_REBUILD_BLUEPRINT.md), but its output could be
reverse-engineered well enough to test against, by fitting each sub-model
against the 6,809 existing rows and checking the fit:

- **p_rank**: `logistic(0.888 * log(rank_b / rank_a))` -- mean absolute
  residual 0.017 against the real stored values.
- **p_overall_elo / p_surface_elo**: standard Elo, K=24, starting at 1500,
  walked forward chronologically from the start of the 2024 data. K=24 was
  the best fit among {16, 20, 24, 28, 32} tested -- mean absolute residual
  ~0.015-0.016 against the real stored values.
- **p_v097_core**: a weighted blend, `0.151*p_rank + 0.440*p_overall_elo +
  0.409*p_surface_elo` (weights fit by least squares, sum to 0.99996) --
  mean absolute residual 0.0018, max 0.028 against the real stored values.

**This is a close approximation, not an exact reproduction.** The residuals
above are real and non-zero -- there's some additional detail in the true
Phase 1 formula this doesn't capture (possibly a different warm-up window,
a small additional context term, or margin-of-victory scaling). Good enough
to test directional conclusions on fresh data; not something to treat as
bit-exact ground truth.

## What was done

- Kept all 6,809 existing rows in `tennis_core_predictions.csv` completely
  untouched -- nothing about the already-validated data was touched.
- Walked the reconstructed Elo/rank models forward through all matches up to
  the existing cutoff (2026-08-21) to catch up state correctly, then
  computed the same columns for the 67 new matches from 2026-08-22 to
  2026-08-24 (mostly Winston-Salem, Cincinnati, and early US Open rounds).
- Saved the combined 6,876-row file as
  `audit/results/tennis_core_predictions_extended.csv` (a new file --
  the original is untouched).
- Ran a copy of `tennis_phase2_fast.py` (`audit/tennis_phase2_extended.py`,
  pointed at the extended file, output to
  `audit/results_phase2_extended/`) against it.

## Result: the original v0.98 decision holds on this fresh data

|         model | delta_log_loss (vs core) |
|----------------|---------------------------|
| p_transition   | **-0.000866** (best improvement) |
| p_surface_win  | -0.000228 |
| p_pressure     | -0.000180 |
| p_form         | -0.000169 |
| ...            | ... |
| p_fatigue      | **+0.002148** (clearly worse) |

`p_transition` and `p_pressure` -- the two features actually promoted into
production for v0.98 -- are still among the best-improving features on data
that includes matches from *after* the original Phase 4 report was written.
`p_fatigue` still looks bad, matching the original decision not to promote
it. **Directionally, nothing here contradicts the earlier validation.** That's
a real, if modest, positive signal -- not proof the model is great, but
evidence the earlier decision wasn't a fluke of the original sample.

## What this is not

This is 67 new matches against a 6,809-match base -- it barely moves the
needle statistically on its own, and it depends on an approximated core
formula rather than the real one. Treat this as "nothing alarming showed up,"
not "fully re-validated." The real fix is still finding or rebuilding the
exact Phase 1 script -- this was a fast way to get a directional read while
that's pending, not a replacement for it.

## Files in this delivery

- `audit/results/tennis_core_predictions_extended.csv` -- the extended
  6,876-row dataset (original 6,809 rows untouched + 67 new rows appended)
- `audit/tennis_phase2_extended.py` -- copy of the Phase 2 script pointed at
  the extended file
- `audit/results_phase2_extended/*.csv` -- the fresh output (summary,
  feature effects, full predictions)
