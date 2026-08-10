# Macabets NFL v0.67 — Preseason Weighting Audit

## What changed

- Reweighted the production NFL team-state score so independent personnel units drive the preseason baseline.
- Reduced aggregate offense/defense weights because QB/OL/skill and DL/secondary already contain overlapping information.
- Prior-season team snapshots are no longer blended a second time into team-state components.
- Prior-season performance remains available only through the already-capped personnel/unit layer.
- Prior-season recent form is reset to a neutral 67.5 preseason baseline.
- Current-season NFL evidence starts conservatively and earns more weight as games accumulate, capped at 85%.

## Production team-state weights

- Quarterback: 22%
- Offensive line: 14%
- Skill positions: 12%
- Defensive line: 14%
- Secondary: 13%
- Offense aggregate: 3%
- Defense aggregate: 4%
- Recent form: 4%
- Coaching: 7%
- Continuity: 4%
- Special teams: 3%

## Current-season evidence ramp

- Week 1: 20%
- +6 percentage points per additional week
- Maximum: 85%

## Guardrail

The 2025 Week 18 team snapshot cannot directly replace 2026 preseason personnel grades. It is not blended again at the team-state layer.
