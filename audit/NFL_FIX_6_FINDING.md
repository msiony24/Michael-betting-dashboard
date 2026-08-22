# NFL Fix #6 - Sleeper hard-status availability

## Finding
Sleeper can place roster-list designations such as `IR`, `PUP`, and `Sus` in the `injury_status` field while `status` remains `Active` or `Inactive`.

The prior Macabets classifier treated only injury_status=`Out` as a hard injury. It checked IR/PUP/suspension language only in `roster_status`. As a result, current IR/PUP/suspended players could remain `Active` with `definitively_unavailable=False`, so an unavailable starter could remain in the starting unit instead of promoting the next depth-chart player.

## Fix
- Recognize exact hard injury-status designations: Out, IR, PUP, Sus/Suspended, NFI, COV and long-form reserve equivalents.
- Preserve Questionable and Doubtful as uncertainty, not automatic benching.
- Reclassify availability whenever a cached CSV is loaded, so a stale snapshot cannot preserve an old bad classification after code is fixed.

## Validation
- 51/51 NFL tests passed.
- Same-day local Sleeper snapshot diagnostic: cached logic had 19 definitive unavailable players; corrected load classified 162, including 143 hard-status rows previously missed.
- In that snapshot, depth-chart substitutions increased from 1 starter promotion to 18 across 14 teams. This is a diagnostic on that snapshot, not a claim that all 18 will remain unavailable on Week 1 game day.
