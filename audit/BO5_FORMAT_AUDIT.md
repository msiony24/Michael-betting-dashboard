# Macabets Tennis Best-of-5 / Format Audit

Date: 2026-08-22

## Scope

This audit is intentionally **not US Open-specific**. The target is a general ATP Best-of-5 format layer that can be used at any Grand Slam. The US Open is simply the next live deployment.

Historical validation used 1,276 leakage-safe ATP Grand Slam main-draw Best-of-5 predictions from 2024-2026 (through Wimbledon 2026).

## 1. Current Best-of-5 transform is useful

The audited v0.98 probability before the format transform versus the existing Best-of-5 transform:

| Model | Accuracy | Log loss | Brier | Mean favorite probability |
|---|---:|---:|---:|---:|
| Raw v0.98 probability | 72.02% | 0.54963 | 0.18533 | 67.31% |
| Existing BO5 transform (`0.72`) | 72.02% | **0.54130** | **0.18270** | 71.08% |

The Best-of-5 transform improves probability quality without changing which player is selected.

### By season with current `0.72` transform

| Season | N | Accuracy | Log loss | Brier |
|---|---:|---:|---:|---:|
| 2024 | 462 | 74.03% | 0.51440 | 0.17181 |
| 2025 | 457 | 71.99% | 0.55828 | 0.18963 |
| 2026 | 357 | 69.47% | 0.55436 | 0.18792 |

A grid search finds a slightly better full-sample fit around 0.76-0.77, but the best setting moves materially by season. The current 0.72 setting is therefore retained as the more conservative, robust choice rather than tuning to the historical sample.

## 2. Best-of-5 calibration

Using the current `0.72` transform:

| Predicted favorite bucket | N | Mean predicted | Actual win rate |
|---|---:|---:|---:|
| 50-55% | 164 | 52.52% | 50.61% |
| 55-60% | 164 | 57.46% | 59.76% |
| 60-65% | 151 | 62.60% | 65.56% |
| 65-70% | 154 | 67.41% | 62.99% |
| 70-75% | 133 | 72.37% | 76.69% |
| 75-80% | 141 | 77.50% | 77.30% |
| 80-85% | 132 | 82.46% | 84.09% |
| 85-90% | 116 | 87.36% | 90.52% |
| 90-95% | 92 | 92.24% | 94.57% |
| 95%+ | 29 | 95.72% | 96.55% |

The 65-70% bucket is the main weak spot, but a single bucket is not enough evidence to add a special calibration rule. Overall calibration is healthy enough to keep the current transform.

## 3. Tournament diagnostics

These are diagnostics only; the model is **not** tuned separately by tournament.

| Tournament | N | Accuracy | Log loss | Brier |
|---|---:|---:|---:|---:|
| Australian Open | 335 | 75.52% | 0.48978 | 0.16160 |
| French Open | 352 | 71.59% | 0.55904 | 0.18901 |
| Wimbledon | 348 | 70.69% | 0.55637 | 0.18904 |
| US Open | 241 | 69.71% | 0.56522 | 0.19367 |

## 4. Dedicated BO5 features tested

Leakage-safe walk-forward experiments tested:

- historical Best-of-5 win rate
- explicit five-set record
- 4+/long-match record
- comeback performance after losing Set 1
- Grand Slam / BO5 experience
- current-event workload and sets played
- age
- combinations of the above

**None were robust enough across both 2025 and 2026 to promote into production.**

The most promising candidate was current-event workload. It improved 2025 log loss by about 0.00234 and accuracy by about 0.66 percentage points, but worsened 2026 log loss/Brier. Five-set record and long-match record generally worsened probability quality. Large combined feature stacks were unstable across seasons.

Decision: **do not add speculative BO5 weights.** Keep the proven format transform and continue collecting evidence.

## 5. Production probability fix

The prior `simulate_matches()` implementation used 20,000 random Monte Carlo trials even though the model already had a per-set probability. That made the final displayed probability move slightly from run to run for the same matchup.

It has been replaced with the exact closed-form match probability for both Best-of-3 and Best-of-5 while preserving the existing per-set mapping:

`set_p = 0.5 + (model_p - 0.5) * 0.72`

For Best-of-5, the exact winning score-line probabilities are calculated for 3-0, 3-1 and 3-2 (and the equivalent losing score lines). There is now no Monte Carlo noise in the displayed win probability.

## 6. Historical deciding-set bug fixed

The old historical profile logic treated every 3-set Grand Slam score as a deciding match. In Best-of-5, a 3-0 result is not a deciding-set match.

The engine now treats:

- Grand Slam main-draw BO5: deciding only when the match reaches 5 sets.
- Generic BO3: deciding when the match reaches 3 sets.

This corrects deciding-match and fatigue diagnostics without forcing a new unvalidated probability weight.

## 7. Major data-integrity issues found and fixed

The most important discovery in this audit was not a missing BO5 feature. It was data labeling and duplication in the live Tennis refresh.

### Tournament level normalization

API rows were frequently passed tournament names into `normalize_level()`, but the old code mainly recognized category strings such as `Grand Slam` or `Masters 1000`. This caused events such as Wimbledon, Montreal and Cincinnati to be stored as generic level `A`.

The normalizer now recognizes Grand Slam and Masters event-name aliases directly.

### Qualifiers separated from main draw

Provider qualifying rounds such as `QF`, `SF` and `F` were mixed under the same event name as the main draw. That could make qualifying matches look like late-round Grand Slam or Masters matches.

The refresh now identifies the main-draw start and converts earlier qualifying knockout rows to:

- round `Q`
- generic level `A`

Example, 2026 Wimbledon after repair:

- 111 qualifier matches: `A / Q`
- exactly 127 main-draw matches: `G`
  - R128: 64
  - R64: 32
  - R32: 16
  - R16: 8
  - QF: 4
  - SF: 2
  - F: 1

### Provider duplicates removed

Recent baseline and API feeds sometimes represented the same match with:

- a one-day date difference
- tournament aliases such as Montreal vs Canadian Open
- abbreviated vs compound surnames
- slightly different score formatting

A conservative near-date duplicate pass now keeps the richer API-enriched row only when the match evidence is strong enough to identify the same contest.

Current 2026 file:

- before cleanup: 2,291 rows
- after cleanup: **2,241 rows**
- duplicates removed: **50**
- matches with full serve/return coverage retained: **896**

This matters because duplicates were actively double-counting recent form and fatigue.

## 8. Regression / release validation

Targeted Tennis suite after the changes:

**35 passed**

The existing Phase 4 release gate was rerun after the data cleanup and still passes every criterion:

- 2025 log loss improves: PASS
- 2026 log loss improves: PASS
- overall Brier improves: PASS
- accuracy protection: PASS
- candidate adjustment remains surgical: PASS

Post-cleanup v0.98 walk-forward sample: 6,762 matches.

## Production recommendation

1. Keep the existing BO5 transform at `0.72`.
2. Use exact deterministic BO3/BO5 probability instead of simulation noise.
3. Do not add extra five-set / experience / age / fatigue weights yet.
4. Keep the corrected event-level, qualifying and duplicate cleanup in the daily Tennis refresh.
5. Continue collecting Best-of-5 results and retest workload/durability features after a larger sample exists.

This is a **general Best-of-5 and Tennis data-integrity release**, not a US Open-specific model.
