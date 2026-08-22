# Macabets NFL Team Power Scale Audit

Date: 2026-08-22

## Finding

Production `team_power_score()` was applying a second compression layer:

- weighted team components are already compressed through player, unit, and team aggregation;
- the final weighted team rating was then divided by 2.5 again before being expressed as football edge points.

That left the current 32-team preseason power range at only about 3.96 points from highest to lowest and a standard deviation of about 0.98 points. With home field set to 1.7 points, home field could overwhelm a large share of meaningful team-strength differences.

## Current Week 1 sanity check

Using the current Macabets production pipeline and a 16-game Week 1 reference market snapshot:

- old scale mean absolute Macabets fair spread: 2.03 points
- reference market mean absolute spread: 4.00 points
- old maximum Macabets fair spread: 4.5 points
- reference maximum spread: 10.5 points
- old mean absolute difference versus reference market: 2.09 points

The old model also produced several near-pick'em games where the underlying personnel/team gap was larger.

## Historical calibration mismatch

The leakage-resistant NFL Phase 1 historical proxy generated 1,954 pregame predictions with:

- mean absolute predicted margin: 6.50 points
- predicted-margin standard deviation: 8.12 points

The 12.0 margin-to-win-probability denominator was calibrated on that much wider margin scale. Applying the same probability mapping to a current production margin distribution averaging only about 2 points creates an internal scale mismatch and pulls too many winner probabilities toward 50%.

## Change

Production now uses:

`power_points = raw - 67.5`

instead of:

`power_points = (raw - 67.5) / 2.5`

This is intentionally a simple 1:1 conversion rather than fitting an exact coefficient to current sportsbook prices.

## Post-change Week 1 sanity check

With all other model layers unchanged:

- mean absolute Macabets fair spread: 3.16 points
- maximum Macabets fair spread: 7.5 points
- mean absolute difference versus the same reference market: 1.47 points

The model still disagrees with the market on individual games, which is desirable for an independent model. The change corrects the systematic compression without forcing Macabets to copy sportsbook lines.

## Guardrail

This change does **not** alter:

- NFL component weights
- 2025 preseason performance cap
- home-field advantage
- matchup refinement caps
- injury/depth-chart logic
- 12.0 margin-to-probability mapping

Probability calibration should be audited separately after the team-strength scale is corrected.
