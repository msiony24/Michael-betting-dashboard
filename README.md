# Macabets — Tennis Fair Line Engine v1

This build adds the first working Macabets decision-engine feature to the existing Streamlit dashboard.

## New in this build

- Tennis Fair Line Engine
- Structured weighted matchup scorecard
- Macabets win probability
- Macabets fair American line
- Sportsbook implied probability comparison
- Estimated ROI at the offered price
- Factor-contribution table
- Confidence-based probability shrinkage to reduce false precision
- Risk Simulator renamed to Outcome Simulator

## Current limitation

The first version uses manual 0–10 matchup ratings. The next development stage will replace those inputs with automatically collected rankings, Elo, recent form, surface performance, workload, scheduling and matchup data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
