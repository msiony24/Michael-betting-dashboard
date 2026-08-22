# NFL Fix #7 — Stable Player Identity

## Finding
The production rating engine aggregated and joined weekly NFL performance by normalized player name only.
That is unsafe because normalized NFL names are not globally unique, and the normalizer intentionally removes
suffixes such as Jr/II/III.

The current Madden 27 roster contains seven normalized names shared by players on different teams:
- Byron Murphy
- Byron Young
- Chris Paul
- Jaylon Jones
- Justin Jefferson
- Marcus Harris
- Michael Carter

All seven also appear in the 2025 nflverse performance file.

A concrete reproduction was Justin Jefferson:
- Minnesota WR Justin Jefferson (GSIS 00-0036322) has 2025 receiving performance.
- Cleveland LB Justin Jefferson (GSIS 00-0041075) is a different player.
- Before the fix, both rows received the Minnesota receiver's performance grade because both normalized to `justinjefferson`.

Michael Carter had a similar suffix collision: `Michael Carter II` and `Michael Carter` both normalized to `michaelcarter`.

## Fix
- Preserve nflverse `player_id` / GSIS identity while aggregating weekly performance.
- Aggregate season performance by stable player ID rather than player name whenever the ID exists.
- Resolve each current Madden player to the nflverse roster GSIS ID before joining performance.
- Preserve performance across team changes because GSIS stays with the player.
- Use name fallback only when the normalized name maps to one unambiguous stats identity.
- Canonicalize nflverse roster `AZ` to Macabets `ARI` (plus LAR->LA and OAK->LV) so Cardinals identities resolve correctly.

## Validation
After the fix:
- Minnesota WR Justin Jefferson retains his 2025 performance prior.
- Cleveland LB Justin Jefferson receives no Minnesota WR performance.
- Tennessee RB Michael Carter retains his correct prior.
- Philadelphia DB Michael Carter II no longer receives the RB performance prior.
- Existing performance-bearing player count changed from 485 to 483 because the two false same-name matches were removed.
- 53/53 NFL-specific tests passed in the audit tree.

This is an identity/data-integrity correction, not a new football feature or a subjective weighting change.
