
# Michael Betting Dashboard — Version 2

A Streamlit dashboard for:

- Tracking pending and settled bets
- Calculating the stake required to win a target amount
- Monitoring bankroll, ROI, win rate and exposure
- Running Monte Carlo bankroll simulations
- Recording sport-specific matchup research
- Exporting and restoring betting records by CSV

## Deploying the update

1. Open the GitHub repository used by Streamlit.
2. Replace the existing `app.py` with this version.
3. Replace `requirements.txt`.
4. Commit the changes.
5. Streamlit Community Cloud will automatically rebuild the website.

The main file path remains:

`app.py`

## Important storage note

Streamlit Community Cloud does not guarantee permanent local storage. Download the CSV backup after entering or settling bets. Use the sidebar uploader to restore it.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
