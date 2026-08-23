# NFL Fix #8 - Signed model-vs-market edge

## Finding

`engine.nfl.analyze()` computes a signed edge for the projected winner:

- positive = Macabets gives the projected winner a higher win probability than the no-vig market
- negative = the market gives the projected winner a higher win probability than Macabets

The shared `recommendation_from_edge()` helper then called `abs(edge)`. This erased the direction of the disagreement.

As a result, a projected winner with a materially **negative** model-vs-market edge could still receive `Lean` or even `Good Bet` if the absolute disagreement and confidence thresholds were large enough.

Example of the old behavior:

- signed projected-winner edge: -8.0 percentage points
- confidence: 78
- old recommendation: `Good Bet`

That is backwards for a moneyline betting recommendation. A negative edge means Macabets believes the price is worse than its own fair probability.

## Fix

Recommendation logic now preserves the sign.

- edge < +0.75 percentage points -> Pass
- positive edge meeting existing thresholds -> Lean / Good Bet / Strong Bet
- zero or negative edge -> always Pass

No team ratings, matchup weights, win probabilities, fair moneylines, spreads, or data pipelines were changed.

## Validation

`python -m pytest tests/test_nfl*.py -q`

55 passed.
