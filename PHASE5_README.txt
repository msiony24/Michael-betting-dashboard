MACABETS TENNIS v0.98 - PHASE 5 PRODUCTION PATCH

UPLOAD / REPLACE ONLY:
1) app.py -> repository root
2) engine/tennis.py -> engine folder

DO NOT upload the ROLLBACK_v097 folder into production unless you need to restore v0.97.
It contains exact pre-patch copies of the two changed files.

WHAT v0.98 CHANGES
- Preserves the audited overall Elo + surface Elo + ranking core.
- Promotes only the two Phase-4 validated secondary signals:
  * Surface transition: 2.125x current audited adjustment shape.
  * Event pressure: 0.975x current audited adjustment shape.
- Caps the combined promoted adjustment at +/-4 percentage points.
- Removes the old 0.76 / 0.82 / 0.88 probability shrink toward 50%.
- Keeps other existing factors calculated for diagnostics/explanations, but they no longer move the v0.98 production probability until separately validated.
- Saves Tennis analyses with the engine-reported v0.98 model version instead of the stale v0.47 label.

NOT CHANGED IN THIS PATCH
- NFL
- UFC
- Supabase / persistence architecture
- Daily Slate
- Tennis identity resolver / H2H resolver
- Serve-return formula
- Fatigue formula

VALIDATION PERFORMED
- Python syntax compilation: PASS for app.py and engine/tennis.py.
- Targeted Tennis regression tests: 10 passed.
- One cache-freshness test could not be collected in this container because Streamlit is not installed here; this is an environment limitation, not a test failure.

ROLLBACK
If production behaves unexpectedly, replace app.py and engine/tennis.py with the copies inside ROLLBACK_v097.
