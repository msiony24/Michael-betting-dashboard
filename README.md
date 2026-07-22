# Macabets — Version 3

A Streamlit betting intelligence dashboard with:

- Bet tracking and settlement
- Target-profit stake calculations
- Bankroll, ROI, win rate and exposure monitoring
- Monte Carlo bankroll simulations
- Performance analysis
- CSV backup and restore
- Tennis Matchup Lab beta
- No-vig market probability
- Macabets fair-line pricing
- Context-based probability adjustments
- Green Light / Lean / Pass recommendations
- Preferred entry and maximum playable price
- Matchup-analysis CSV export

## Deploying the update

1. Open the GitHub repository connected to Streamlit.
2. Replace the current `app.py` with `app_macabets_v3.py`.
3. Rename `app_macabets_v3.py` to `app.py` in GitHub, or paste its contents into the existing `app.py`.
4. Keep the included `requirements.txt`.
5. Commit the changes.
6. Streamlit Community Cloud will rebuild automatically.

## Important storage note

Streamlit session data is temporary. Download the bets CSV and matchup-analysis CSV after updates you want to keep.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
