# Macabets v0.23 — NFL Data Pipeline

Macabets now supports a real NFL performance-data pipeline while preserving the existing tennis application.

## What changed

- `engine/nfl_fetch.py` downloads public play-by-play data through `nflreadpy` / nflverse.
- `engine/nfl_ratings.py` loads the saved team snapshot into Macabets.
- `update_nfl_data.py` creates `data/nfl/team_snapshot.csv`.
- `.github/workflows/update-nfl-data.yml` refreshes the snapshot weekly or manually.
- The NFL page clearly states whether it is using an nflverse snapshot or starter-prior fallback.

## NFL rating inputs

- Offense: EPA/play, success rate, explosive-play rate and turnover rate.
- Defense: EPA allowed, success rate allowed, explosive plays allowed and takeaway rate.
- Quarterback: passing EPA/dropback, passing success rate and CPOE.
- Strength of schedule: opponent net EPA faced.
- Special teams: special-teams EPA.
- Coaching: transparent manual prior for now.

## First real-data refresh

After uploading this release to GitHub:

1. Open the repository's **Actions** tab.
2. Select **Update Macabets NFL Data**.
3. Click **Run workflow**.
4. Enter a completed season such as `2025` during the 2026 offseason, or leave it blank once the current season has usable games.
5. Wait for the workflow to commit `data/nfl/team_snapshot.csv`.
6. Streamlit will redeploy and the NFL page will show **Real-data mode**.

The app continues to run safely with starter priors if the network refresh fails or no snapshot exists.

## Data attribution

NFL statistical data is obtained from nflverse through the `nflreadpy` package. Most nflverse data is provided under CC-BY 4.0; retain attribution when distributing derived outputs.
