"""Measure NFL_MARGIN_SD from real results instead of trusting a constant.

Computes the standard deviation of (actual final margin - pregame spread)
across historical NFL seasons. That number is what engine/nfl.py should use in
NFL_MARGIN_SD, and it is the only input to the margin-to-probability mapping.

This is a MEASUREMENT script. It does not fit anything to the Macabets model
and its output must never be tuned to make the model agree with the market.

Run:
    python audit/measure_nfl_margin_sd.py
    python audit/measure_nfl_margin_sd.py --start 2015 --end 2025

Output:
    audit/results_nfl_margin_sd/margin_sd.csv       per-season and pooled SD
    audit/results_nfl_margin_sd/spread_buckets.csv  actual win rate by spread
    audit/results_nfl_margin_sd/summary.json

Requires network access to nflverse (the GitHub Actions runner has it; the
Streamlit Cloud app does not need to run this).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "results_nfl_margin_sd"
SCHEDULE_URL = (
    "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
)


def load_schedules(start: int, end: int) -> pd.DataFrame:
    df = pd.read_csv(SCHEDULE_URL, low_memory=False)
    df = df[(df["season"] >= start) & (df["season"] <= end)]

    # Regular season and playoffs both count; preseason does not.
    if "game_type" in df.columns:
        df = df[df["game_type"] != "PRE"]

    needed = ["season", "home_score", "away_score", "spread_line"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"nflverse schedule is missing expected columns: {missing}. "
            f"Available: {sorted(df.columns)}"
        )

    df = df.dropna(subset=needed).copy()
    df["home_margin"] = df["home_score"] - df["away_score"]

    # nflverse spread_line is stated from the HOME team's perspective, positive
    # when the home team is favored. Verify that before trusting it: the
    # correlation with actual home margin must be clearly positive.
    corr = df["spread_line"].corr(df["home_margin"])
    if corr < 0.2:
        raise SystemExit(
            f"spread_line does not correlate with home margin (r={corr:.3f}). "
            "The sign convention may have changed upstream. Inspect before use."
        )

    df["residual"] = df["home_margin"] - df["spread_line"]
    return df


def per_season_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, chunk in df.groupby("season"):
        rows.append(
            {
                "season": int(season),
                "games": int(len(chunk)),
                "residual_mean": round(float(chunk["residual"].mean()), 3),
                "residual_sd": round(float(chunk["residual"].std(ddof=1)), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("season")


def spread_bucket_table(df: pd.DataFrame, sd: float) -> pd.DataFrame:
    """Actual favorite win rate by spread size, next to the model's prediction.

    This is the table to eyeball. If the fitted SD is right, predicted and
    actual should track closely across every bucket, especially the big ones.
    """
    fav = df.copy()
    fav["fav_spread"] = fav["spread_line"].abs()
    fav["fav_won"] = (
        ((fav["spread_line"] > 0) & (fav["home_margin"] > 0))
        | ((fav["spread_line"] < 0) & (fav["home_margin"] < 0))
    )
    # Pushes on the moneyline (ties) are rare; drop them rather than credit them.
    fav = fav[fav["home_margin"] != 0]
    fav = fav[fav["fav_spread"] > 0]

    edges = [0, 1.5, 3.5, 6.5, 9.5, 13.5, 60]
    labels = ["0.5-1.5", "2-3.5", "4-6.5", "7-9.5", "10-13.5", "14+"]
    fav["bucket"] = pd.cut(fav["fav_spread"], bins=edges, labels=labels)

    dist = NormalDist(0.0, sd)
    rows = []
    for bucket, chunk in fav.groupby("bucket", observed=True):
        if chunk.empty:
            continue
        mean_spread = float(chunk["fav_spread"].mean())
        rows.append(
            {
                "spread_bucket": str(bucket),
                "games": int(len(chunk)),
                "avg_spread": round(mean_spread, 2),
                "model_win_prob": round(dist.cdf(mean_spread), 4),
                "actual_win_rate": round(float(chunk["fav_won"].mean()), 4),
                "gap": round(
                    dist.cdf(mean_spread) - float(chunk["fav_won"].mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2006)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()

    df = load_schedules(args.start, args.end)
    if len(df) < 500:
        raise SystemExit(f"Only {len(df)} usable games; widen the season range.")

    pooled_sd = float(df["residual"].std(ddof=1))
    pooled_mean = float(df["residual"].mean())

    seasons = per_season_table(df)
    buckets = spread_bucket_table(df, pooled_sd)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seasons.to_csv(OUT_DIR / "margin_sd.csv", index=False)
    buckets.to_csv(OUT_DIR / "spread_buckets.csv", index=False)

    summary = {
        "purpose": "measure NFL_MARGIN_SD for engine/nfl.py; audit-only",
        "seasons": [args.start, args.end],
        "games": int(len(df)),
        "residual_mean": round(pooled_mean, 4),
        "pooled_margin_sd": round(pooled_sd, 3),
        "recommended_NFL_MARGIN_SD": round(pooled_sd, 1),
        "season_sd_min": float(seasons["residual_sd"].min()),
        "season_sd_max": float(seasons["residual_sd"].max()),
        "note": (
            "residual_mean should be near zero. A large non-zero mean means the "
            "spread sign convention is flipped or the sample is biased, and the "
            "SD should not be used until that is explained."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Games: {len(df)}  seasons {args.start}-{args.end}")
    print(f"Residual mean: {pooled_mean:+.3f} points (want ~0)")
    print(f"Pooled margin SD: {pooled_sd:.3f}")
    print(f"Per-season SD range: {seasons['residual_sd'].min():.2f} - "
          f"{seasons['residual_sd'].max():.2f}")
    print()
    print("Favorite win rate by spread bucket:")
    print(buckets.to_string(index=False))
    print()
    print(f"--> Set NFL_MARGIN_SD = {pooled_sd:.1f} in engine/nfl.py")
    if not math.isclose(pooled_sd, 13.5, abs_tol=1.0):
        print("    (This differs from the current 13.5 by more than a point. "
              "Check the bucket table above before changing it.)")


if __name__ == "__main__":
    main()
