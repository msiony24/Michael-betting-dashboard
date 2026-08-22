"""NFL Phase 2 calibration audit.

Uses Phase 1 leakage-resistant historical predictions and evaluates a finer
margin-to-win-probability calibration grid around the Phase 1 optimum.
Production code is not imported or modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("audit/results_nfl_phase1/predictions.csv")
OUTPUT = Path("audit/results_nfl_phase2")
CURRENT_DENOMINATOR = 8.25


def sigmoid_margin(margin: float, denominator: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(margin) / float(denominator)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p.astype(float) - y.astype(float)) ** 2))


def evaluate(frame: pd.DataFrame, denominator: float) -> dict:
    probs = frame["predicted_home_margin"].map(lambda x: sigmoid_margin(x, denominator)).to_numpy()
    y = frame["home_win"].astype(int).to_numpy()
    return {
        "denominator": float(denominator),
        "games": int(len(frame)),
        "log_loss": log_loss(y, probs),
        "brier": brier(y, probs),
        "mean_probability": float(np.mean(probs)),
        "actual_home_win_rate": float(np.mean(y)),
    }


def calibration_bins(frame: pd.DataFrame, denominator: float) -> pd.DataFrame:
    out = frame.copy()
    out["home_probability"] = out["predicted_home_margin"].map(lambda x: sigmoid_margin(x, denominator))
    out["winner_probability"] = np.where(out["home_probability"] >= 0.5, out["home_probability"], 1 - out["home_probability"])
    out["winner_correct"] = np.where(out["home_probability"] >= 0.5, out["home_win"] == 1, out["home_win"] == 0).astype(int)
    bins = [0.50, .55, .60, .65, .70, .75, .80, .85, .90, 1.001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85-90%", "90%+"]
    out["bucket"] = pd.cut(out["winner_probability"], bins=bins, labels=labels, include_lowest=True, right=False)
    table = out.groupby("bucket", observed=False).agg(
        games=("winner_correct", "size"),
        avg_model_probability=("winner_probability", "mean"),
        actual_win_rate=("winner_correct", "mean"),
    ).reset_index()
    table["calibration_gap"] = table["avg_model_probability"] - table["actual_win_rate"]
    return table


def season_stability(frame: pd.DataFrame, denominator: float) -> pd.DataFrame:
    rows = []
    for season, group in frame.groupby("season"):
        metric = evaluate(group, denominator)
        rows.append({"season": int(season), **metric})
    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT.exists():
        raise RuntimeError(f"Missing Phase 1 input: {INPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)

    fine_grid = np.arange(10.5, 13.501, 0.05)
    grid = pd.DataFrame([evaluate(frame, d) for d in fine_grid])
    best_log = grid.sort_values(["log_loss", "brier"]).iloc[0].to_dict()
    best_brier = grid.sort_values(["brier", "log_loss"]).iloc[0].to_dict()

    candidates = sorted({
        CURRENT_DENOMINATOR,
        11.5, 11.75, 12.0, 12.25, 12.5,
        round(float(best_log["denominator"]), 2),
        round(float(best_brier["denominator"]), 2),
    })

    candidate_rows = [evaluate(frame, d) for d in candidates]
    candidates_df = pd.DataFrame(candidate_rows)

    for d in candidates:
        calibration_bins(frame, d).to_csv(OUTPUT / f"calibration_{str(d).replace('.', '_')}.csv", index=False)
        season_stability(frame, d).to_csv(OUTPUT / f"seasons_{str(d).replace('.', '_')}.csv", index=False)

    current = evaluate(frame, CURRENT_DENOMINATOR)
    promoted = min(candidate_rows, key=lambda r: (r["log_loss"], r["brier"]))
    improvement = {
        "log_loss_absolute": current["log_loss"] - promoted["log_loss"],
        "log_loss_relative_pct": 100.0 * (current["log_loss"] - promoted["log_loss"]) / current["log_loss"],
        "brier_absolute": current["brier"] - promoted["brier"],
        "brier_relative_pct": 100.0 * (current["brier"] - promoted["brier"]) / current["brier"],
    }

    summary = {
        "purpose": "audit-only probability calibration; production untouched",
        "games": int(len(frame)),
        "current_denominator": CURRENT_DENOMINATOR,
        "best_log_loss_denominator": best_log,
        "best_brier_denominator": best_brier,
        "recommended_candidate": promoted,
        "improvement_vs_current": improvement,
        "promotion_rule": "Prefer a rounded denominator near the broad optimum, not a fragile decimal optimum. Promote only if season-level calibration remains stable.",
    }

    grid.to_csv(OUTPUT / "fine_grid.csv", index=False)
    candidates_df.to_csv(OUTPUT / "candidate_comparison.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nCandidates:")
    print(candidates_df.to_string(index=False))


if __name__ == "__main__":
    main()
