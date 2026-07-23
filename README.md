# Macabets v0.21 — NFL Foundation

Macabets now contains two analysis workspaces:

- **Tennis Analysis** — the existing ATP model and workflow remain intact.
- **NFL Foundation** — a new NFL matchup workspace with all 32 teams, market inputs, game context, projected score, fair-line report shell, confidence, upset risk, and game-script output.

## Important limitation

NFL v0.1 is intentionally market-derived. It does **not** claim an independent betting edge yet. The interface and report contract are now in place so independent team-quality, quarterback, injury, matchup, coaching, situational, and game-script engines can be added without rebuilding the application structure.

## Repository structure

- `app.py`
- `engine/data.py`
- `engine/tennis.py`
- `engine/nfl.py`
- `engine/nfl_data.py`
- `engine/confidence.py`
- `update_tennis_data.py`
- `requirements.txt`
- `.github/workflows/update-tennis-data.yml`
- `data/`

## Deployment

Upload the **contents of this folder** directly to the root of the existing GitHub repository. Do not upload the ZIP itself and do not place these files inside a new nested folder.

After GitHub commits the changes, Streamlit should redeploy automatically. The app should display **Macabets v0.21 — NFL Foundation** in the upper-right corner, and the Analysis Engine should contain **Tennis Analysis**, **NFL Foundation**, and **Outcome Simulator** tabs.
