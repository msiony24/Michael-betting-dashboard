# Macabets NFL Rating Backtest

Season replayed: **2025** | Madden baseline: **Madden 26 2-week-1 (downloaded)** | Weeks scored: 2-18 | Games: 255

Lower log loss and Brier = better probabilities. Differences of ~0.005 in log loss are small;
the 95% ranges below show whether a difference is distinguishable from luck.

## Main comparison

| Version | Log loss | Brier | Accuracy | Early weeks LL | Late weeks LL |
|---|---|---|---|---|---|
| v1_4_legacy | 0.6249 | 0.2177 | 63.5% | 0.6428 | 0.6058 |
| v1_5_current | 0.6476 | 0.2280 | 62.7% | 0.6459 | 0.6494 |
| madden_only | 0.6559 | 0.2319 | 60.4% | 0.6486 | 0.6637 |

## Head-to-head (negative = first version is better)

- **v1_5_current vs madden_only:** -0.0083 (95% range -0.0141 to -0.0026; better in 100% of resamples)
- **v1_5_current vs v1_4_legacy:** +0.0226 (95% range +0.0053 to +0.0388; better in 0% of resamples)
- **best grid (grid_pm0.5_tc0.8_tk3) vs v1_5_current:** -0.0032 (95% range -0.0061 to -0.0002; better in 98% of resamples)

## Versus the betting market (same games)

| Source | Log loss | Brier | Accuracy |
|---|---|---|---|
| market_closing_moneyline | 0.6125 | 0.2137 | 64.7% |
| madden_only_same_games | 0.6559 | 0.2319 | 60.4% |
| v1_4_legacy_same_games | 0.6249 | 0.2177 | 63.5% |
| v1_5_current_same_games | 0.6476 | 0.2280 | 62.7% |

## Calibration (favorite's predicted vs actual win rate)

- **madden_only:** 50-60%: 55% pred / 54% actual (n=159), 60-70%: 64% pred / 70% actual (n=82), 70-80%: 73% pred / 77% actual (n=13), 80-100%: 87% pred / 100% actual (n=1)
- **v1_4_legacy:** 50-60%: 55% pred / 54% actual (n=141), 60-70%: 64% pred / 71% actual (n=69), 70-80%: 74% pred / 82% actual (n=40), 80-100%: 85% pred / 80% actual (n=5)
- **v1_5_current:** 50-60%: 55% pred / 55% actual (n=166), 60-70%: 64% pred / 74% actual (n=78), 70-80%: 72% pred / 90% actual (n=10), 80-100%: 86% pred / 100% actual (n=1)

## Best settings from the grid (in-sample; treat as a hint, not proof)

| Variant | Log loss | Early LL | Late LL | Player stability x | Team cap | Team ramp K |
|---|---|---|---|---|---|---|
| grid_pm0.5_tc0.8_tk3 | 0.6444 | 0.6451 | 0.6437 | 0.5 | 0.8 | 3 |
| grid_pm1_tc0.8_tk3 | 0.6450 | 0.6458 | 0.6442 | 1 | 0.8 | 3 |
| grid_pm0.5_tc0.8_tk6 | 0.6454 | 0.6450 | 0.6459 | 0.5 | 0.8 | 6 |
| grid_pm2_tc0.8_tk3 | 0.6459 | 0.6466 | 0.6453 | 2 | 0.8 | 3 |
| grid_pm1_tc0.8_tk6 | 0.6461 | 0.6458 | 0.6464 | 1 | 0.8 | 6 |
| grid_pm0.5_tc0.6_tk3 | 0.6461 | 0.6451 | 0.6472 | 0.5 | 0.6 | 3 |
| grid_pm1_tc0.6_tk3 | 0.6467 | 0.6458 | 0.6477 | 1 | 0.6 | 3 |
| grid_pm0.5_tc0.8_tk12 | 0.6469 | 0.6452 | 0.6487 | 0.5 | 0.8 | 12 |

## Scale check

`best_margin_scale` in summary.csv is the in-sample multiplier on projected margins that would
minimize log loss (1.0 = production). Well above 1 means projections are too timid; below 1 too bold.

## What this does and does not test

Only the personnel/performance rating core is replayed. Coaching, continuity, recent form,
schedule, weather, scheme and matchup layers are held neutral; depth charts and Sleeper injuries
are replaced by each week's active roster. All versions get identical inputs.

Generated 2026-09-15T18:53:53+00:00.
