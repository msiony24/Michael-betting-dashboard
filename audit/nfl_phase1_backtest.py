"""Audit-only NFL Phase 1 backtest for Macabets.

This script does NOT import or modify the production NFL prediction engine.
It reconstructs a deliberately simple, leakage-resistant weekly strength model
from historical completed games, then audits the production engine's current
margin-to-win-probability mapping against out-of-sample NFL results.

Why this exists:
- The production repo does not contain historical weekly snapshots of every
  current Macabets personnel/matchup layer.
- Replaying today's ratings against old games would leak future information.
- So Phase 1 first tests the calibration assumption (margin -> win probability)
  honestly, using only information available before each historical game.

Outputs are artifacts only. Nothing in production is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2019, 2026))
PRODUCTION_LOGISTIC_DENOMINATOR = 8.25
PRODUCTION_HOME_FIELD = 1.7
K_FACTOR = 0.18
SEASON_REGRESSION = 0.55


def sigmoid_margin(margin: float, denominator: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(margin) / float(denominator)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p.astype(float) - y.astype(float)) ** 2))


def download_games() -> pd.DataFrame:
    with urllib.request.urlopen(GAMES_URL, timeout=60) as response:
        raw = response.read()
    tmp = Path("/tmp/nflverse_games.csv")
    tmp.write_bytes(raw)
    frame = pd.read_csv(tmp)
    return frame


def prepare_games(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"nflverse games.csv missing columns: {sorted(missing)}")
    games = frame.copy()
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games[
        games["season"].isin(SEASONS)
        & games["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])
        & games["home_score"].notna()
        & games["away_score"].notna()
    ].copy()
    games["actual_home_margin"] = games["home_score"] - games["away_score"]
    games["home_win"] = (games["actual_home_margin"] > 0).astype(int)
    # Ties cannot train a binary winner model honestly; keep them out of calibration.
    games = games[games["actual_home_margin"] != 0].copy()
    sort_cols = [c for c in ["season", "week", "gameday", "gametime"] if c in games.columns]
    return games.sort_values(sort_cols).reset_index(drop=True)


def preseason_prior(team: str, prior_season_end: dict[str, float]) -> float:
    old = float(prior_season_end.get(team, 0.0))
    return old * SEASON_REGRESSION


def build_walk_forward_predictions(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    previous_end: dict[str, float] = {}

    for season in sorted(games["season"].dropna().astype(int).unique()):
        season_games = games[games["season"] == season].copy()
        teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
        ratings = {team: preseason_prior(team, previous_end) for team in teams}

        for _, game in season_games.iterrows():
            home = str(game["home_team"])
            away = str(game["away_team"])
            neutral = bool(game.get("location") == "Neutral") if "location" in game else False
            hfa = 0.0 if neutral else PRODUCTION_HOME_FIELD
            predicted_margin = ratings.get(home, 0.0) - ratings.get(away, 0.0) + hfa

            if season in TEST_SEASONS:
                rows.append({
                    "season": int(season),
                    "week": int(game["week"]),
                    "home_team": home,
                    "away_team": away,
                    "predicted_home_margin": float(predicted_margin),
                    "actual_home_margin": float(game["actual_home_margin"]),
                    "home_win": int(game["home_win"]),
                    "neutral": neutral,
                })

            # Update only AFTER recording the prediction: no future leakage.
            residual = float(game["actual_home_margin"] - predicted_margin)
            update = K_FACTOR * residual
            ratings[home] = ratings.get(home, 0.0) + update
            ratings[away] = ratings.get(away, 0.0) - update

        previous_end = dict(ratings)

    return pd.DataFrame(rows)


def evaluate_denominator(preds: pd.DataFrame, denominator: float) -> dict:
    probs = preds["predicted_home_margin"].map(lambda x: sigmoid_margin(x, denominator)).to_numpy()
    y = preds["home_win"].to_numpy()
    winner = (probs >= 0.5).astype(int)
    return {
        "denominator": round(float(denominator), 3),
        "games": int(len(preds)),
        "accuracy": round(float(np.mean(winner == y)), 5),
        "log_loss": round(log_loss(y, probs), 6),
        "brier": round(brier(y, probs), 6),
        "mean_probability": round(float(np.mean(probs)), 5),
        "actual_home_win_rate": round(float(np.mean(y)), 5),
    }


def calibration_table(preds: pd.DataFrame, denominator: float) -> pd.DataFrame:
    out = preds.copy()
    out["prob"] = out["predicted_home_margin"].map(lambda x: sigmoid_margin(x, denominator))
    bins = [0, .45, .50, .55, .60, .65, .70, .75, .80, 1.001]
    labels = ["<45%", "45-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80%+"]
    out["bucket"] = pd.cut(out["prob"], bins=bins, labels=labels, include_lowest=True, right=False)
    table = out.groupby("bucket", observed=False).agg(
        games=("home_win", "size"),
        avg_predicted_probability=("prob", "mean"),
        actual_home_win_rate=("home_win", "mean"),
    ).reset_index()
    table["calibration_gap"] = table["avg_predicted_probability"] - table["actual_home_win_rate"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="audit/results_nfl_phase1")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    games = prepare_games(download_games())
    preds = build_walk_forward_predictions(games)
    if preds.empty:
        raise RuntimeError("No historical predictions were generated.")

    grid = np.arange(4.0, 14.01, 0.25)
    evaluations = pd.DataFrame([evaluate_denominator(preds, d) for d in grid])
    best = evaluations.sort_values(["log_loss", "brier"]).iloc[0].to_dict()
    production = evaluate_denominator(preds, PRODUCTION_LOGISTIC_DENOMINATOR)

    # Also report performance by projected favorite strength using production mapping.
    detail = preds.copy()
    detail["production_home_probability"] = detail["predicted_home_margin"].map(
        lambda x: sigmoid_margin(x, PRODUCTION_LOGISTIC_DENOMINATOR)
    )
    detail["projected_winner_probability"] = np.where(
        detail["production_home_probability"] >= 0.5,
        detail["production_home_probability"],
        1.0 - detail["production_home_probability"],
    )
    detail["projected_winner_correct"] = np.where(
        detail["production_home_probability"] >= 0.5,
        detail["home_win"] == 1,
        detail["home_win"] == 0,
    ).astype(int)
    strength_bins = [0.50, .55, .60, .65, .70, .75, .80, 1.001]
    strength_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80%+"]
    detail["winner_prob_bucket"] = pd.cut(
        detail["projected_winner_probability"], strength_bins, labels=strength_labels,
        include_lowest=True, right=False,
    )
    strength = detail.groupby("winner_prob_bucket", observed=False).agg(
        games=("projected_winner_correct", "size"),
        avg_model_probability=("projected_winner_probability", "mean"),
        actual_win_rate=("projected_winner_correct", "mean"),
    ).reset_index()
    strength["calibration_gap"] = strength["avg_model_probability"] - strength["actual_win_rate"]

    calibration = calibration_table(preds, PRODUCTION_LOGISTIC_DENOMINATOR)
    season_summary = detail.groupby("season").agg(
        games=("home_win", "size"),
        winner_accuracy=("projected_winner_correct", "mean"),
    ).reset_index()

    summary = {
        "purpose": "audit-only; production code untouched",
        "historical_test_seasons": TEST_SEASONS,
        "games_tested": int(len(preds)),
        "walk_forward_model": {
            "description": "pregame-only team strength updated after each completed game",
            "k_factor": K_FACTOR,
            "season_regression": SEASON_REGRESSION,
            "home_field_points": PRODUCTION_HOME_FIELD,
        },
        "production_probability_mapping": production,
        "best_grid_probability_mapping": best,
        "interpretation_guardrail": (
            "This calibrates the margin-to-probability assumption on a leakage-resistant historical proxy. "
            "It is not a claim that the full current Macabets NFL engine had these historical predictions, "
            "because historical snapshots of every current engine layer do not exist in the repo."
        ),
    }

    detail.to_csv(output / "predictions.csv", index=False)
    evaluations.to_csv(output / "denominator_grid.csv", index=False)
    calibration.to_csv(output / "production_calibration.csv", index=False)
    strength.to_csv(output / "favorite_strength.csv", index=False)
    season_summary.to_csv(output / "season_summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nProduction probability calibration by projected winner strength:")
    print(strength.to_string(index=False))
    print("\nSeason summary:")
    print(season_summary.to_string(index=False))


if __name__ == "__main__":
    main()
