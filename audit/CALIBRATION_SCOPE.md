# Calibration scope — read before copying any number out of this folder

## The rule

**Constants fit in `audit/` calibrate the audit model. They are not production
constants.** Nothing in this folder may be pasted into `engine/` without first
being re-validated against production output or re-derived from measurement.

## What went wrong once already

`engine/nfl.py:spread_to_home_probability` used a logistic with divisor `12.0`.
That number came from the Phase 2 grid search
(`audit/nfl_phase2_calibration.py`, results in `audit/results_nfl_phase2/`),
which minimized log loss at 12.0 over 1,954 games.

The grid was run against `audit/nfl_walk_forward_backtest.py` — a standalone
Elo proxy with `k_factor=0.18`, `season_regression=0.55`,
`home_field_points=1.7`. It is not the production engine and shares no code
with it.

That proxy produced inflated margins. From
`audit/results_nfl_phase1/favorite_strength.csv`:

| bucket | avg model prob | actual win rate | gap |
|---|---|---|---|
| 70-75% | 0.722 | 0.667 | +0.056 |
| 75-80% | 0.774 | 0.698 | +0.077 |
| 80%+ | 0.866 | 0.769 | +0.097 |

Widening the divisor from 8.25 to 12.0 flattened the curve and cancelled that
inflation. Log loss improved (0.6515 → 0.6429) and the grid search recorded it
as the winner. It was the right constant *for that model*.

Production margins come from a different pipeline: team-state grades →
`power_points` → the matchup/scheme/weather/situational adjustment stack in
`engine/nfl.py:analyze`. They were never inflated. The wide divisor therefore
compressed every production probability toward 50%, worst on the biggest
favorites:

| projected margin | divisor 12.0 | empirical |
|---|---|---|
| 3 pts | 56.2% | ~58.8% |
| 7 pts | 64.2% | ~69.8% |
| 10 pts | 69.7% | ~77.1% |
| 14 pts | 76.3% | ~85.0% |

Observed symptom (2026 Week 2): Macabets projected DET 28-22, a 6.4-point
margin, essentially agreeing with a market that had the Lions around -7. The
mapping turned that agreement into 63% against a no-vig market price near 73%,
which reads as "the model likes the underdog." It didn't. The football model
and the market were half a point apart.

`summary.json` in `results_nfl_phase1/` already carried the warning, under
`interpretation_guardrail`. It was correct and it was not followed.

## Current state

`spread_to_home_probability` is now a normal CDF with a single measured
parameter, `NFL_MARGIN_SD`. It has no fitted degrees of freedom. Re-measure it
with `audit/measure_nfl_margin_sd.py`, which reads nflverse results and reports
the standard deviation of (actual margin − pregame spread).

`tests/test_nfl_probability_mapping.py` asserts the mapping against empirical
favorite win rates and will fail if the compressed curve returns.

## Before moving any future audit constant into `engine/`

1. Name the model the constant was fit against. If it is not the production
   engine, stop.
2. State what bias in that model the constant might be absorbing.
3. Re-derive from measurement where the quantity is a real-world property
   (margin variance, home-field advantage) rather than a model knob.
4. Add a test in `tests/` that pins the production behavior, not the fit.
