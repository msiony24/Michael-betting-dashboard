# Macabets Learning Architecture

## Core rule

**Settlement data is the source of truth; live model weights never self-edit directly from recent results.**

The learning stack is staged so one hot or bad week cannot rewrite Macabets.

## Layer 1 — Automatic Results & Settlement

Every prediction should ultimately contain:

- internal analysis ID
- provider event ID(s)
- sport and market type
- participants and event date
- model version
- prediction and predicted probability
- entry line/odds
- final result and score
- prediction correctness
- closing line / consensus close
- entry no-vig probability
- closing no-vig probability
- probability CLV
- model edge at close
- settlement source/version
- manual-override state

This layer must be stable before automated learning is trusted.

## Layer 2 — Performance Analysis

Break down settled predictions by, at minimum:

- sport
- surface
- model version
- predicted player/team
- opponent
- predicted-probability bucket
- confidence bucket
- entry-odds bucket
- closing-line bucket
- verdict
- price assessment
- matchup stability/volatility bucket
- event category / round
- favorite vs underdog
- core model factors present at analysis time

Required metrics:

- record / accuracy
- Brier score
- log loss
- calibration error
- flat-unit ROI
- CLV
- model edge at close
- sample size and confidence intervals

No segment should be treated as meaningful without a minimum sample threshold.

## Layer 3 — Calibration Engine

Calibration answers: **when Macabets says 70%, how often does that side actually win?**

Initial implementation should run in shadow mode:

1. Train only on settled historical predictions.
2. Preserve time order.
3. Use walk-forward/out-of-sample validation, never random leakage across future/past predictions.
4. Shrink small probability buckets toward the global baseline.
5. Compare raw vs calibrated Brier score and log loss.
6. Do not activate the calibration map unless it improves out-of-sample results by a predefined threshold across multiple windows.
7. Store the calibration version separately from the underlying sport model version.

Example diagnostic:

- Raw predictions in the 68–72% bucket: 70.1% average forecast
- Actual win rate: 64.2%
- Sufficient sample + persistent out-of-sample bias: calibrator may reduce future 70% outputs toward ~65–67%

The exact correction must come from the validated calibration model, not a one-week manual adjustment.

## Layer 4 — Weight-Learning Challenger

Weight learning should never directly edit the champion model.

For each prediction, save the factor values that existed **at prediction time**, such as:

- overall Elo
- surface Elo
- ranking
- recent form
- opponent strength
- serve/return matchup
- surface performance
- fatigue/workload
- H2H adjustment
- injury/availability
- matchup style
- situational context

Then train challenger models against final outcomes.

Safeguards:

- minimum sample per sport
- regularization / shrinkage
- capped weight movement per release
- no feature may learn from post-match information
- time-based train/validation/test splits
- compare prediction accuracy, Brier score, log loss, CLV, and stability
- test key subgroups so an overall gain is not hiding catastrophic failures on a surface/odds range

## Layer 5 — Model Registry & Promotion Gate

Every live change gets a version and immutable evaluation record.

Suggested lifecycle:

`draft → backtest → shadow → challenger → approved → champion → retired`

A challenger becomes champion only if it passes all gates, for example:

- sufficient out-of-sample sample size
- better Brier score than champion
- no unacceptable log-loss deterioration
- calibration not worse
- no severe degradation in major sport/surface/odds segments
- performance improvement persists across more than one time window
- manual approval before promotion

Never promote because of one week of ROI.

## Model rollback

The registry must retain:

- model version
- code/config hash
- factor weights
- calibration version
- training-data cutoff
- evaluation period
- out-of-sample metrics
- promotion decision/reason

That makes rollback deterministic if a new model behaves badly in production.
