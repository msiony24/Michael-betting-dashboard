# NFL Fix #11 - Player weekly stats were not part of the production workflow

## Finding

The daily NFL workflow refreshed the broad nflverse foundation, Sleeper availability, and automated ratings, but it did not run the repository's dedicated `update_nfl_player_stats.py` refresh.

During the 2026 preseason, nflverse's 2026 weekly player-stat release is not published yet. The broad foundation correctly reports that dataset as unavailable, while Macabets continues to use the existing 2025 player-weekly file as its capped preseason prior. However, because the dedicated refresher was not invoked, the fallback file and its metadata were not guaranteed to be refreshed on each workflow run.

That creates two transition risks:

1. A successful daily workflow can leave the player-performance prior stale while every other NFL data source has refreshed.
2. When 2026 weekly stats first become available, the production workflow should explicitly switch the player feed from the capped 2025 prior to current-season data and rewrite the metadata in the same run that rebuilds ratings.

## Fix

The workflow now runs `update_nfl_player_stats.py` after the NFL foundation refresh and before Sleeper availability / automated rating rebuilds.

The dedicated refresher already has the correct behavior:

- 2026 unavailable -> load 2025, mark it as fallback, cap performance at 20%.
- 2026 available -> use 2026, mark it current-season, allow the normal 80% maximum as sample size grows.

## Validation

Added two regression tests covering both sides of the season transition.

Full NFL-specific suite: 60 passed.
