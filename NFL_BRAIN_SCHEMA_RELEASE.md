# Macabets NFL Brain Schema v0.2

## Purpose
Separates the football brain from the current ratings source. The exploit,
quarterback-gate, win-condition, failure-condition, and chain-reaction logic now
reads from a stable detailed team profile.

## Current behavior
The existing broad Macabets ratings are translated into the new schema through
an explicit provisional adapter. When Madden 27 or other upgraded data becomes
available, those sources can populate the same schema without rewriting the
brain.

## New brain outputs
- Quarterback gate for each team
- Opponent-specific exploit opportunities
- Primary win condition for each team
- Primary failure condition for each team
- Existing conflict and chain-reaction analysis
- Data-contract status and limitations

## Changed files
- app.py
- engine/nfl_brain.py
- engine/nfl_team_schema.py (new)
- tests/test_nfl_brain.py
- tests/test_nfl_team_schema.py (new)

## Verification
- All Python files compiled successfully.
- 12 focused NFL tests passed.
- Full suite: 43 passed, 1 pre-existing unrelated tennis failure.
- Existing tennis failure: test expects player-traits version 1.0, while the
  data file reports 1.5-top150.
