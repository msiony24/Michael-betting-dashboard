"""Walk-forward NFL backtest against completed seasons, plus a simple-baseline comparison.

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

This script also runs a genuinely simple baseline -- standard win/loss Elo,
no play-by-play, no EPA -- on the exact same games, so the two numbers are
directly comparable. This answers the same question asked for Tennis
earlier this week: is the additional intelligence (EPA-based signal, and
everything built on top of it) actually earning its place over something
much simpler?

WALK-FORWARD DISCIPLINE: for the week-N game between team A and team B, team
A and B's EPA snapshots are built using ONLY plays from weeks 1..N-1 of that
season. Nothing from week N or later ever touches the prediction for week N.
The simple Elo baseline is equally walk-forward: each team's rating only
reflects games actually played before the one being predicted.

HOW TO RUN THIS: requires `nflreadpy` and real network access to fetch
play-by-play data -- neither is available in the sandbox this was written
in, only in your Codespace / GitHub Actions environment (the same place
update_nfl_data.py already runs successfully). Run:

    python audit/nfl_walk_forward_backtest.py --seasons 2023 2024 2025

Output: prints accuracy/Brier/log loss/calibration for both the EPA signal
and the simple Elo baseline side by side, plus saves both prediction sets
to CSV.
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
    data for each prediction. Returns one row per game with the raw
    (unscaled) EPA margin and the real outcome -- fit_scaling_constant()
    finds the real points-per-EPA multiplier from this."""
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
                epa_margin_raw = home_epa - away_epa
                home_won = float(game["home_score"] > game["away_score"])
                rows.append({
                    "season": season, "week": week, "home_team": home, "away_team": away,
                    "epa_margin_raw": epa_margin_raw, "home_won": home_won,
                    "home_score": game["home_score"], "away_score": game["away_score"],
                })

    return pd.DataFrame(rows)


def fit_scaling_constant(predictions: pd.DataFrame) -> float:
    """Fit the points-per-EPA-differential multiplier by maximum likelihood
    instead of guessing one. Dependency-free coarse-to-fine grid search (no
    scipy -- it isn't in requirements.txt, so this environment can't assume
    it's installed)."""
    x = predictions["epa_margin_raw"].to_numpy()
    y = predictions["home_won"].to_numpy()

    def neg_log_likelihood(scale: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-(x * scale) / 12.0)), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    low, high = 0.1, 200.0
    best = low
    for _ in range(40):
        candidates = np.linspace(low, high, 25)
        losses = [neg_log_likelihood(c) for c in candidates]
        best_idx = int(np.argmin(losses))
        best = float(candidates[best_idx])
        step = (high - low) / 24
        low = max(0.1, best - 2 * step)
        high = min(200.0, best + 2 * step)
        if high - low < 1e-6:
            break
    return best


def apply_scaling(predictions: pd.DataFrame, scale: float) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["predicted_home_probability"] = predictions["epa_margin_raw"].apply(
        lambda m: _spread_to_home_probability(m * scale)
    )
    return predictions


def run_simple_elo_baseline(seasons: list[int], k: float = 20.0, home_field: float = 65.0) -> pd.DataFrame:
    """A genuinely simple baseline: standard win/loss Elo (no play-by-play,
    no EPA, no margin-of-victory scaling), walked forward across the same
    seasons, for direct comparison against the EPA-based signal on
    identical games. This is the NFL analog of the "basic single-factor
    Elo" baseline used in the Tennis benchmark earlier this week."""
    pbp = _fetch_pbp(seasons)
    pbp = pbp[pbp["season_type"] == "REG"]

    ratings: dict[str, float] = {}
    rows = []
    for season in seasons:
        season_pbp = pbp[pbp["season"] == season]
        weeks = sorted(season_pbp["week"].dropna().unique())
        for week in weeks:
            games = _final_game_rows(season_pbp[season_pbp["week"] == week])
            for _, game in games.iterrows():
                home, away = game["home_team"], game["away_team"]
                home_rating = ratings.get(home, 1500.0)
                away_rating = ratings.get(away, 1500.0)
                expected_home = 1.0 / (1.0 + 10.0 ** (-((home_rating + home_field) - away_rating) / 400.0))
                home_won = float(game["home_score"] > game["away_score"])
                rows.append({
                    "season": season, "week": week, "home_team": home, "away_team": away,
                    "predicted_home_probability": expected_home, "home_won": home_won,
                })
                ratings[home] = home_rating + k * (home_won - expected_home)
                ratings[away] = away_rating + k * ((1 - home_won) - (1 - expected_home))
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
    parser.add_argument("--out-dir", type=str, default="audit/results")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EPA-BASED SIGNAL (the signal Macabets' refinement layers read from)")
    print("=" * 70)
    epa_predictions = run_backtest(args.seasons)
    fitted_scale = fit_scaling_constant(epa_predictions)
    epa_predictions = apply_scaling(epa_predictions, fitted_scale)
    epa_predictions.to_csv(out_dir / "nfl_walk_forward_predictions.csv", index=False)
    epa_results = score(epa_predictions)
    print(f"Fitted scaling constant (points per unit of EPA differential): {fitted_scale:.3f}")
    print(f"Games scored: {epa_results['n_games']}")
    print(f"Accuracy: {epa_results['accuracy']:.4f}")
    print(f"Brier score: {epa_results['brier_score']:.4f}")
    print(f"Log loss: {epa_results['log_loss']:.4f}")

    print()
    print("=" * 70)
    print("SIMPLE BASELINE (standard win/loss Elo, no play-by-play at all)")
    print("=" * 70)
    elo_predictions = run_simple_elo_baseline(args.seasons)
    elo_predictions.to_csv(out_dir / "nfl_simple_elo_baseline_predictions.csv", index=False)
    elo_results = score(elo_predictions)
    print(f"Games scored: {elo_results['n_games']}")
    print(f"Accuracy: {elo_results['accuracy']:.4f}")
    print(f"Brier score: {elo_results['brier_score']:.4f}")
    print(f"Log loss: {elo_results['log_loss']:.4f}")

    print()
    print("=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"{'Model':<45}{'Accuracy':<12}{'Brier':<10}{'Log Loss'}")
    print(f"{'Simple Elo baseline':<45}{elo_results['accuracy']:<12.4f}{elo_results['brier_score']:<10.4f}{elo_results['log_loss']:.4f}")
    print(f"{'EPA-based signal (Macabets foundation)':<45}{epa_results['accuracy']:<12.4f}{epa_results['brier_score']:<10.4f}{epa_results['log_loss']:.4f}")

    print("\nCalibration by confidence bucket (EPA-based signal):")
    print(epa_results["calibration_table"].to_string())
    print(f"\nFull predictions saved to: {out_dir}")


if __name__ == "__main__":
    main()
