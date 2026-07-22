# Macabets Version 5

Macabets V5 is rebuilt around one workflow:

1. Select a sport.
2. Search two players.
3. Select the tournament and round.
4. Press **Run Macabets**.
5. Read the simulation, fair line, market edge, reasons and risks.

There are no manual model probabilities, context sliders or personal form ratings.

## Files

- `app.py` — streamlined interface
- `engine/data.py` — ATP data loading and caching
- `engine/tennis.py` — ratings, context and Monte Carlo simulation
- `requirements.txt`

## Deployment

Upload all files and folders to the GitHub repository connected to Streamlit. Preserve the `engine` folder structure.

Streamlit will rebuild after the commit.

## Data attribution

ATP historical rankings, results and statistics are sourced from Jeff Sackmann / Tennis Abstract and are licensed CC BY-NC-SA 4.0. Attribution is required and commercial use is not permitted without separate permission.

## Data loading

The app first uses certificate-verified HTTPS. If the Streamlit environment has an incomplete certificate chain, it retries the GitHub raw request without local certificate verification and caches the downloaded CSV files.
