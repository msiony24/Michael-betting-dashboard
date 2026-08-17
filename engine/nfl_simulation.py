"""Monte Carlo game-script simulation for Macabets NFL.

The simulator does not add a new football edge. It expresses the already-built
Macabets margin and total as outcome distributions so the app can reason about
volatility, upset frequency, score ranges, spreads and totals without double
counting personnel/scheme/LOS/situational inputs.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

DEFAULT_SIMULATIONS = 20000
NFL_MARGIN_SD = 13.86
NFL_TOTAL_SD = 12.75


def _seed_for_matchup(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _script_label(*, favorite: str, favorite_prob: float, fair_total: float, margin_iqr: float) -> str:
    if favorite_prob < 0.56:
        return "Toss-up / high-variance game"
    if fair_total >= 49.5 and margin_iqr >= 17.0:
        return "Volatile scoring game"
    if fair_total <= 42.0 and favorite_prob < 0.64:
        return "Low-scoring, one-possession type game"
    if favorite_prob >= 0.70:
        return f"{favorite} control script"
    return f"{favorite} lean in a competitive game"


def simulate_game(
    *,
    away_team: str,
    home_team: str,
    projected_home_margin: float,
    fair_total: float,
    market_spread_home: float | None = None,
    market_total: float | None = None,
    simulations: int = DEFAULT_SIMULATIONS,
    seed_context: Any = None,
) -> dict:
    """Simulate NFL score/margin outcomes around the model's existing centers.

    projected_home_margin and fair_total already include Macabets' football
    intelligence. The simulation only adds realistic game-to-game variance.
    """
    n = max(2000, int(simulations or DEFAULT_SIMULATIONS))
    seed = _seed_for_matchup(
        away_team,
        home_team,
        round(float(projected_home_margin), 3),
        round(float(fair_total), 3),
        seed_context,
        n,
    )
    rng = np.random.default_rng(seed)

    # Margin and total are intentionally modeled as separate uncertainty axes.
    # A small positive correlation captures that higher-scoring environments
    # tend to create a little more margin dispersion without turning the market
    # total into a second side signal.
    z_margin = rng.normal(0.0, 1.0, n)
    z_total_independent = rng.normal(0.0, 1.0, n)
    rho = 0.12
    z_total = rho * z_margin + np.sqrt(1.0 - rho**2) * z_total_independent

    margin = float(projected_home_margin) + NFL_MARGIN_SD * z_margin
    total = float(fair_total) + NFL_TOTAL_SD * z_total
    total = np.clip(total, 10.0, 90.0)

    home_score = (total + margin) / 2.0
    away_score = (total - margin) / 2.0
    home_score = np.clip(home_score, 0.0, None)
    away_score = np.clip(away_score, 0.0, None)

    # Reconstruct realized totals/margins after non-negative score clipping.
    realized_total = home_score + away_score
    realized_margin = home_score - away_score

    home_win_probability = float(np.mean(realized_margin > 0.0))
    away_win_probability = 1.0 - home_win_probability
    favorite = home_team if home_win_probability >= 0.5 else away_team
    favorite_probability = max(home_win_probability, away_win_probability)

    home_margin_q25 = _pct(realized_margin, 25)
    home_margin_q75 = _pct(realized_margin, 75)
    margin_iqr = home_margin_q75 - home_margin_q25
    total_q25 = _pct(realized_total, 25)
    total_q75 = _pct(realized_total, 75)

    if market_spread_home is None:
        home_cover_probability = None
    else:
        # Home bet covers when home scoring margin + listed home spread > 0.
        home_cover_probability = float(np.mean(realized_margin + float(market_spread_home) > 0.0))

    if market_total is None:
        over_probability = None
    else:
        over_probability = float(np.mean(realized_total > float(market_total)))

    volatility = "High" if margin_iqr >= 19.0 else "Moderate" if margin_iqr >= 15.0 else "Low"
    one_score_probability = float(np.mean(np.abs(realized_margin) <= 8.0))
    close_game_probability = float(np.mean(np.abs(realized_margin) <= 3.0))

    return {
        "available": True,
        "simulations": n,
        "seed": int(seed),
        "home_win_probability": home_win_probability,
        "away_win_probability": away_win_probability,
        "favorite": favorite,
        "favorite_win_probability": favorite_probability,
        "upset_probability": 1.0 - favorite_probability,
        "home_cover_probability": home_cover_probability,
        "away_cover_probability": None if home_cover_probability is None else 1.0 - home_cover_probability,
        "over_probability": over_probability,
        "under_probability": None if over_probability is None else 1.0 - over_probability,
        "mean_home_score": float(np.mean(home_score)),
        "mean_away_score": float(np.mean(away_score)),
        "home_score_range_50": [_pct(home_score, 25), _pct(home_score, 75)],
        "away_score_range_50": [_pct(away_score, 25), _pct(away_score, 75)],
        "home_score_range_80": [_pct(home_score, 10), _pct(home_score, 90)],
        "away_score_range_80": [_pct(away_score, 10), _pct(away_score, 90)],
        "margin_median": _pct(realized_margin, 50),
        "margin_range_50": [home_margin_q25, home_margin_q75],
        "margin_range_80": [_pct(realized_margin, 10), _pct(realized_margin, 90)],
        "total_median": _pct(realized_total, 50),
        "total_range_50": [total_q25, total_q75],
        "total_range_80": [_pct(realized_total, 10), _pct(realized_total, 90)],
        "one_score_probability": one_score_probability,
        "close_game_probability": close_game_probability,
        "volatility": volatility,
        "game_script": _script_label(
            favorite=favorite,
            favorite_prob=favorite_probability,
            fair_total=float(fair_total),
            margin_iqr=margin_iqr,
        ),
        "method_note": (
            "Monte Carlo distribution centered on the existing Macabets fair margin and fair total. "
            "It adds outcome variance only and does not create another matchup adjustment."
        ),
    }
