# Macabets Tennis v0.6

This build connects the Streamlit interface to the automatic ATP analytics engine.

## Repository structure

- `app.py`
- `engine/data.py`
- `engine/tennis.py`
- `update_tennis_data.py`
- `requirements.txt`
- `.github/workflows/update-tennis-data.yml`
- `data/` populated by the updater

## First setup

1. Upload the entire package structure to GitHub.
2. In GitHub, open **Actions**.
3. Run **Update Macabets Tennis Data**.
4. Wait for the action to commit the `data/atp_matches_*.csv` files.
5. Reboot the Streamlit application.

The Automatic Match Analyzer will then calculate Elo, surface Elo, form, serve/return profile,
fatigue, rest, event-pressure adjustments, fair odds, confidence, and simulated set outcomes.
