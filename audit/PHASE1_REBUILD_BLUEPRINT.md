# Rebuilding audit/results/tennis_core_predictions.csv ("Phase 1")

## Why this exists

The entire tennis validation pipeline (Phase 2 -> Phase 4: feature testing,
bootstrap confidence, calibration, release gates) reads its match list from
`audit/results/tennis_core_predictions.csv`. That file caps out at
2026-08-21 even though `data/atp_matches_*.csv` has real match results
through 2026-08-23 (23 new matches as of this check, including live US Open
rounds). The script that originally generated this file is not in the repo
-- only its output was ever committed. Confirmed via `grep -rl
"tennis_core_predictions" audit/*.py` -> only tennis_phase2_fast.py reads
it, nothing writes it.

Regenerating this file (extended through today, and re-run periodically
from here on) is the actual unblock for genuine walk-forward confirmation
of the v0.98 candidate on data nobody has looked at yet.

## Exact schema required (confirmed from the real file, 6,809 rows, zero
nulls in any column)

| column          | type    | meaning |
|-----------------|---------|---------|
| date            | date    | match date |
| surface         | str     | Hard / Clay / Grass |
| tournament      | str     | tourney_name from the ATP source data |
| round           | str     | R128/R64/.../SF/F etc. |
| player_a        | str     | winner_name (see below -- order matters) |
| player_b        | str     | loser_name |
| y               | float   | 1.0 if player_a won, 0.0 otherwise |
| sample_a        | int     | player_a's career match count *as of that date* (walk-forward, not final career total) |
| sample_b        | int     | same for player_b |
| p_rank          | float   | probability derived from ATP ranking differential only |
| p_overall_elo   | float   | probability from a plain, all-surface Elo model |
| p_surface_elo   | float   | probability from a surface-specific Elo model |
| p_elo_blend     | float   | some blend of overall_elo + surface_elo (exact weights unknown -- reverse-engineer from tennis_phase2_fast.py's `elo_prob`/`bprof`/`oppprof` helpers, which implement a live version of this same family of calculation) |
| p_v097_core     | float   | the actual production v0.97 core formula's output at that point in time |

Important: since row 0 has `player_a="Safiullin R."` beating `player_b="Shelton B."`
with `y=1.0`, **player_a is always the actual match winner** in this dataset,
not "the favorite" or a fixed home/away designation. Any regeneration script
must preserve that convention or y's meaning inverts silently.

## Walk-forward requirement (the part that's easy to get subtly wrong)

Every one of `sample_a`, `sample_b`, `p_rank`, `p_overall_elo`,
`p_surface_elo`, `p_elo_blend`, `p_v097_core` must be computed using **only
information available strictly before that match's date** -- Elo ratings,
rankings, and sample counts all update chronologically as the loop
processes matches in date order. `tennis_phase2_fast.py`'s `load()` function
already sorts by `['date','tourney_name','round']` and its `since()`/`bprof()`/
`oppprof()` helpers are a real, working reference implementation of exactly
this walk-forward pattern (Elo carried in a running `elo` dict, keyed by
`key(name)`, updated match-by-match as the loop advances) -- reuse that
pattern rather than reinventing it.

## Concrete next step

1. Reverse-engineer `p_v097_core`'s exact formula by comparing
   `tennis_phase2_fast.py`'s `p_core` computation (which is a *re-derivation*,
   not a copy) against a handful of known rows in the existing
   `tennis_core_predictions.csv`, to confirm the reverse-engineered formula
   reproduces the existing values before trusting it on new matches.
2. Write a small script that:
   - Loads all matches from `data/atp_matches_*.csv` (same as
     `tennis_phase2_fast.py`'s `load()`).
   - Walks forward in date order, maintaining running Elo/rank/sample state.
   - For every match already present in the existing
     `tennis_core_predictions.csv`, confirm the newly computed values match
     the existing ones (a correctness check, not just a vibe check).
   - For matches after 2026-08-21, compute and append new rows.
3. Overwrite `audit/results/tennis_core_predictions.csv` with the extended
   version.
4. Re-run `audit/tennis_phase2_fast.py` (already confirmed working) and then
   `audit/tennis_phase4_candidate_validation.py` (fix its `IN` path first --
   it currently looks for `audit/results_phase2/tennis_phase2_predictions.csv`,
   which doesn't exist; the real file Phase 2 produces is at
   `audit/results/tennis_phase2_predictions.csv`) against the extended data.
5. That finally gives a release-gate report against matches nobody has
   examined before -- the actual open item from Phase 4's own findings.

## Verified working today (2026-08-23), safe to build on directly

- `data/atp_matches_*.csv` has real match data through 2026-08-23 (23 rows
  past the 2026-08-21 cutoff already).
- `audit/tennis_phase2_fast.py` runs end-to-end without error against the
  current repo, in well under a minute.
- `audit/tennis_phase4_candidate_validation.py` has real, correct logic
  (bootstrap confidence, calibration tables, release gate) -- it just needs
  its `IN` path corrected and a Phase-1 file that actually extends past
  2026-08-21 to have something new to say.
