# NFL Fix #5 - Prevent current-season performance from being counted twice

## Finding

Macabets has two NFL layers that consume performance information:

1. `engine/nfl_rating_engine.py` already blends NFL performance into the automated player/unit ratings.
   - In preseason, 2025 evidence is capped at 20%.
   - During 2026, current-season performance progressively earns more influence.
   - QB/RB/WR/TE receive player-level performance.
   - OL/DL/secondary/special teams receive team-unit performance, with a smaller defense proxy for linebackers.

2. `engine/nfl_team_state.py` was then blending the same `team_snapshot.csv` performance fields into quarterback, offense, defense, offensive line, defensive line, secondary, and special teams again.

That creates a double-counting path once the 2026 season begins. A Week 1 unit could receive current-season evidence in the rating engine and then receive another live blend in the team-state layer.

## Fix

The team-state layer now treats the automated rating output as the complete baseline for all football components. It no longer re-blends team snapshot values for QB, offense, defense, OL, DL, secondary or special teams.

`recent_form` remains a separate current-season-only signal because it is not already embedded in the automated rating baseline. It still starts neutral in preseason and progressively earns weight during the season.

## Why this matters

This prevents Macabets from overreacting to one or two early-season games and keeps the intended transition intact:

- Preseason: Madden 27/current roster baseline + capped 2025 evidence.
- Early 2026: current-season evidence gradually enters once.
- Later 2026: current-season evidence gains influence without being stacked twice.

## Validation

Ran all NFL tests matching `tests/test_nfl*.py` after the change.

Result: **50 passed**.
