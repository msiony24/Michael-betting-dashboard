# NFL Fix #9 — Separate Winner Prediction From Bet Recommendation

## Finding
The main NFL recommendation card mixed two different markets:

- The verdict (`Strong Bet`, `Worth Betting`, `Lean`, `Pass`) came from the projected winner's moneyline price.
- The `RECOMMENDED PLAY` text came from the spread-value side.

That could produce a card where a moneyline verdict for the projected winner was paired with the opposite team's spread.

## Fix
The main recommendation is now aligned to the NFL engine's primary objective:

- Macabets always projects a winner.
- The main betting verdict evaluates that projected winner's moneyline price.
- If actionable, `RECOMMENDED PLAY` shows the projected winner's moneyline and current price.
- Spread disagreement remains visible separately as `Spread Value` and does not override the winner/moneyline verdict.

## Validation
- Full NFL test suite: 56 passed.
- No probability, team-rating, spread, injury, or calibration math changed.
