# Macabets — Version 4

Macabets Version 4 adds an automatic ATP Tennis Matchup Lab.

## Tennis workflow

The user selects:

- Player A
- Player B
- Tournament
- Round
- Surface
- Event date
- A simple 1–10 form rating for each player

Macabets automatically calculates:

- Overall Elo
- Surface Elo
- Ranking-based strength
- Statistical base probability
- Recent database form
- Serve and return performance
- Surface history
- Rest and workload
- Recorded retirement signals
- Advanced-round and major-event performance
- Deciding-match performance
- Head-to-head context
- Final probability and fair American line
- Edge and Green Light / Lean / Pass when sportsbook odds are supplied

## Data

The app downloads public ATP historical match files at runtime and caches them for six hours.
Internet access is required when the cache refreshes.

## Deploy

Upload and replace these three files in the GitHub repository connected to Streamlit:

- `app.py`
- `tennis_engine.py`
- `requirements.txt`

Commit the changes. Streamlit Community Cloud should rebuild automatically.

## Storage

Saved bets and matchup analyses remain session-based. Download CSV backups for information that must be retained.
