"""Walk-forward NFL backtest against completed seasons.

WHY THIS EXISTS: the live model's ratings are built on this season's Madden
data and roster, which by definition has no games played against it yet (the
2026 season kicks off 2026-09-09 -- checked schedules.csv directly: 272
games, 0 with a final score, as of this script being written). That means
there is currently no way to test whether *this season's* predictions are
accurate. What this script tests instead is the underlying signal the
model's "refinement layers" (scheme, situational, opponent-adjustment, etc.)
all read from: team offensive/defensive EPA. If EPA differential reliably
predicts real outcomes in seasons that have already been played, that's
real evidence the signal those layers work from is sound -- not a test of
the exact Madden-blended production formula, since Madden ratings don't
exist historically.

WALK-FORWARD DISCIPLINE: for the week-N game between team A and team B, team
A and B's EPA snapshots are built using ONLY plays from weeks 1..N-1 of that
season (plus, optionally, prior completed seasons for early-season teams
with little current-season sample). Nothing from week N or later ever
touches the prediction for week N.

HOW TO RUN THIS: requires `nflreadpy` and real network access to fetch
play-by-play data -- neither is available in the sandbox this was written
in, only in your Codespace / GitHub Actions environment (the same place
update_nfl_data.py already runs successfully). Run:

    python audit/nfl_walk_forward_backtest.py --seasons 2023 2024 2025

Output: a CSV of every prediction vs. actual outcome, plus a printed summary
(accuracy, Brier score, log loss, calibration table) -- same methodology
used for the Tennis Phase 2/4 backtests earlier this week.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nfl_fetch import build_team_snapshot, _final_game_rows  # noqa: E402


def _spread_to_home_probability(home_margin: float, divisor: float = 12.0) -> float:
    """Same calibrated logistic mapping used in engine/nfl.py, so this
    backtest tests the same probability-conversion the live model uses."""
    return 1.0 / (1.0 + math.exp(-float(home_margin) / divisor))


def _fetch_pbp(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    return nfl.load_pbp(seasons).to_pandas() if hasattr(nfl.load_pbp(seasons), "to_pandas") else nfl.load_pbp(seasons)


def run_backtest(seasons: list[int], min_week_sample: int = 2) -> pd.DataFrame:
    """Predict every REG-season game in `seasons`, using only prior weeks'
    data for each prediction. Returns one row per game with the predicted
    home win probability and the real outcome."""
    pbp = _fetch_pbp(seasons)
    pbp = pbp[pbp["season_type"] == "REG"]

    rows = []
    for season in seasons:
        season_pbp = pbp[pbp["season"] == season]
        weeks = sorted(season_pbp["week"].dropna().unique())
        for week in weeks:
            if week <= min_week_sample:
                continue  # not enough same-season sample yet; skip rather than guess

            prior_pbp = season_pbp[season_pbp["week"] < week]
            if prior_pbp.empty:
                continue
            try:
                snapshot = build_team_snapshot(prior_pbp, season).set_index("team_abbr")
            except ValueError:
                continue  # no usable plays yet for this cutoff -- skip, don't fake it

            this_week_pbp = season_pbp[season_pbp["week"] == week]
            games = _final_game_rows(this_week_pbp)
            for _, game in games.iterrows():
                home, away = game["home_team"], game["away_team"]
                if home not in snapshot.index or away not in snapshot.index:
                    continue  # a team with zero prior data this cutoff -- skip, don't guess
                home_epa = snapshot.loc[home, "offense_epa_per_play"] - snapshot.loc[home, "defense_epa_allowed"]
                away_epa = snapshot.loc[away, "offense_epa_per_play"] - snapshot.loc[away, "defense_epa_allowed"]
                epa_margin = (home_epa - away_epa) * 25.0  # rough points-per-EPA-differential scaling
                predicted_prob = _spread_to_home_probability(epa_margin)
                home_won = float(game["home_score"] > game["away_score"])
                rows.append({
                    "season": season, "week": week, "home_team": home, "away_team": away,
                    "predicted_home_probability": predicted_prob, "home_won": home_won,
                    "home_score": game["home_score"], "away_score": game["away_score"],
                })

    return pd.DataFrame(rows)


def score(predictions: pd.DataFrame) -> dict:
    p = predictions["predicted_home_probability"].to_numpy()
    y = predictions["home_won"].to_numpy()
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)

    accuracy = float(((p >= 0.5).astype(float) == y).mean())
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))

    p_favorite = np.maximum(p, 1 - p)
    favorite_won = np.where(p >= 0.5, y == 1, y == 0)
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.00]
    calibration = pd.DataFrame({"p_favorite": p_favorite, "favorite_won": favorite_won})
    calibration["bucket"] = pd.cut(calibration["p_favorite"], bins=bins, include_lowest=True)
    calibration_table = calibration.groupby("bucket", observed=True).agg(
        n=("favorite_won", "size"), predicted_avg=("p_favorite", "mean"), actual_win_rate=("favorite_won", "mean"),
    )

    return {"n_games": len(predictions), "accuracy": accuracy, "brier_score": brier,
            "log_loss": log_loss, "calibration_table": calibration_table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--out", type=str, default="audit/results/nfl_walk_forward_predictions.csv")
    args = parser.parse_args()

    predictions = run_backtest(args.seasons)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)

    results = score(predictions)
    print(f"\nGames scored: {results['n_games']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Brier score: {results['brier_score']:.4f}  (lower is better, 0.25 = always guessing 50/50)")
    print(f"Log loss: {results['log_loss']:.4f}")
    print("\nCalibration by confidence bucket:")
    print(results["calibration_table"].to_string())
    print(f"\nFull predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
