# Macabets Tennis Audit - Phase 4 Candidate Validation

Baseline: clean v0.97 audit data. This phase changes no production files.

## Candidate tested

The safest v0.98 candidate preserves the existing v0.97 core probability and adds only the two secondary signals that were temporally stable in Phase 3:

- Surface transition: 2.125x the Phase-2 adjustment
- Event pressure: 0.975x the Phase-2 adjustment
- Combined transition + pressure adjustment capped at +/-4 percentage points
- No additional probability shrink toward 50%
- Fatigue is not promoted; it remains slated for redesign
- Serve/return is not promoted until historical coverage and surface baselines improve

The +/-4pp cap was added in Phase 4 after the uncapped candidate showed a 95th-percentile adjustment above 5pp. The cap keeps the change surgical while preserving most of the predictive gain.

## Head-to-head results

Across 6,809 walk-forward historical ATP matches from 2024 through Aug. 21, 2026:

| Model | Accuracy | Log loss | Brier |
|---|---:|---:|---:|
| v0.97 core | 64.650% | 0.623947 | 0.217811 |
| v0.98 candidate | 64.797% | 0.622728 | 0.217290 |
| v0.97 tested-stack proxy | 64.253% | 0.625574 | 0.218370 |

The v0.97 tested-stack proxy is the Phase-2 historical reconstruction of the tested secondary factors plus v0.97 shrinkage. It is not an exact replay of every live/manual context input in production and must not be treated as one.

### Temporal results

- 2025 core log loss: 0.626992
- 2025 candidate log loss: 0.625986
- 2026 core log loss: 0.625013
- 2026 candidate log loss: 0.623204

Candidate accuracy improved by about 0.13 percentage points in 2025 and 0.41 percentage points in 2026.

## Paired bootstrap confidence

Across all 6,809 matches, the candidate's paired log-loss improvement versus the core was -0.001219 with a 95% bootstrap interval of approximately [-0.002007, -0.000375]. About 99.9% of bootstrap samples favored the candidate on log loss.

For 2026, the paired log-loss improvement was -0.001809 with a 95% interval of approximately [-0.003255, -0.000377], with about 99.1% of bootstrap samples favoring the candidate.

For 2025, the point estimate favored the candidate but the 95% interval crossed zero. This means the improvement is directionally consistent but not equally strong in every individual year.

## Guardrails

The capped candidate passes the Phase-4 release gates:

1. 2025 log loss improves.
2. 2026 log loss improves.
3. Overall Brier score improves.
4. Overall accuracy is not degraded.
5. The selected secondary adjustment is capped at +/-4pp.

The mean absolute candidate adjustment is about 1.03pp; the median is about 0.57pp. The model is therefore still driven by the v0.97 core rather than by the new secondary layer.

## Segment caution

The candidate improved log loss on Clay, Grass, and Hard overall. Accuracy rose strongly on Clay and slightly on Grass, while Hard accuracy dipped modestly even though Hard log loss improved.

By round, most segments improved or were essentially flat on log loss, but semifinals were a notable weak spot. That is a reason to monitor event-pressure behavior rather than increasing pressure weight further.

## Important validation limitation

2026 was already examined during Phase 3, so Phase 4 must not describe 2026 as a completely untouched holdout. The cleanest future confirmation will be new matches after Aug. 21, 2026 and future seasons. The current result is best viewed as temporal stability plus paired historical evidence, not a final proof of permanent edge.

## Phase-4 recommendation

The candidate has earned the right to become the proposed v0.98 implementation, but implementation should remain narrowly scoped:

- preserve the v0.97 core formula;
- remove the aggressive final shrink toward 50%;
- recalibrate surface transition to about 2.125x its current Phase-2 shape;
- retain event pressure near current strength;
- cap the combined promoted transition + pressure contribution at +/-4pp;
- do not strengthen fatigue, serve/return, experience, deciding-set, form, opponent-strength, or surface-win layers based on this audit;
- continue future live validation before any further weight increases.

The next phase should create a production patch from v0.97 implementing only this narrow candidate, with tests and an easy rollback path.
