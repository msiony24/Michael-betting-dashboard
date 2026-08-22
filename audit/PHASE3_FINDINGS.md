# Macabets Tennis Audit - Phase 3 Findings

Baseline: v0.97. Audit only; no production model files changed.

## Main conclusions

- The v0.97 core probability remains strong and should not be rebuilt.
- The existing probability shrink toward 50% is not supported by temporal validation. The core is already close to calibrated; 2026 favored only a tiny 1.02 expansion, not shrinkage.
- Surface transition is the clearest stable secondary factor. Its current shape is useful, but historical validation suggests roughly 2x the present Phase-2 adjustment strength.
- Event pressure is a smaller but stable positive factor. Approximately current strength is reasonable.
- Fatigue is mis-specified. The best fitted multiplier is negative, meaning the current adjustment often points the wrong direction. Do not simply strengthen it; redesign the signal before production use.
- Surface win rate, recent form, opponent strength, deciding-set history, and experience were not stable across both temporal validations. They should not be increased until redesigned/retested.
- Serve/return data coverage is the limiting issue. There was no usable Phase-2 serve/return coverage in 2024-25 and only 440/1,943 matches (22.6%) in 2026. On 2026 Hard and Grass it helped, while Clay performed poorly. This layer needs more historical stat coverage and surface-specific treatment before calibration is trustworthy.

## Temporal validation

A conservative transition + pressure candidate improved log loss in both 2025 and 2026 without touching the core architecture:

- 2025 core log loss: 0.626992
- 2025 transition(2x) + pressure(1x): 0.625810
- 2026 core log loss: 0.625013
- 2026 transition(2x) + pressure(1x): 0.623077

Adding an inverted fatigue term improved those numbers further, but fatigue should be redesigned rather than flipped blindly.

## Recommended next production step

For v0.98 implementation, the safest first production candidate is:

1. Preserve the v0.97 core formula.
2. Remove the current confidence/probability shrink toward 50%.
3. Recalibrate surface transition toward about 2x its current Phase-2 adjustment strength.
4. Keep event pressure around current strength.
5. Neutralize or redesign fatigue before allowing it to move predictions.
6. Leave unstable factors at current/neutral influence until each is redesigned and revalidated.
7. Expand historical serve/return coverage before making that layer a major probability mover.

All details are in `results_phase3/`.
