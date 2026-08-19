from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import numpy as np
import pandas as pd

from engine.ufc import analyze, load_ufc_ratings


CAP_AUDIT_VERSION = "Macabets UFC Matchup-Cap Audit v0.1 — Structural Stress Test"


@dataclass(frozen=True)
class UFCCapAuditConfig:
    max_pairs: int = 60
    rounds: int = 3
    seed: int = 2417


CAPS = {
    "performance": 0.05,
    "style": 0.03,
    "cardio_3r": 0.0075,
    "cardio_5r": 0.015,
    "damage": 0.0075,
    "context": 0.02,
    "correlated": 0.085,
    "total_non_rating": 0.10,
}


def _active_ratings() -> pd.DataFrame:
    frame = load_ufc_ratings()
    if "active_pool" in frame:
        mask = frame["active_pool"]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[mask]
    frame = frame.copy()
    frame["macabets_rating"] = pd.to_numeric(frame.get("macabets_rating"), errors="coerce")
    return frame.dropna(subset=["macabets_rating"])


def _pair_candidates(ratings: pd.DataFrame, max_pairs: int, seed: int) -> list[tuple[str, str, str]]:
    """Deterministic blend of close, medium and deliberately extreme same-division pairs."""
    rng = random.Random(int(seed))
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, division: str) -> None:
        if not a or not b or a == b:
            return
        key = tuple(sorted((a.casefold(), b.casefold())))
        if key in seen:
            return
        seen.add(key)
        pairs.append((a, b, division))

    groups: list[tuple[str, list[str]]] = []
    for division, group in ratings.groupby("division", sort=True):
        g = group.sort_values("macabets_rating", ascending=False)
        names = g["fighter"].astype(str).tolist()
        if len(names) < 2:
            continue
        groups.append((str(division), names))

        # Strong-vs-strong and neighboring quality.
        for i in range(min(4, len(names) - 1)):
            add(names[i], names[i + 1], str(division))

        # Stress-test strong-vs-middle / strong-vs-lower active fighters.
        for i in range(min(2, len(names))):
            for j in {len(names) // 2, len(names) - 1}:
                if 0 <= j < len(names) and i != j:
                    add(names[i], names[j], str(division))

    attempts = 0
    while len(pairs) < max_pairs and groups and attempts < max_pairs * 30:
        division, names = rng.choice(groups)
        a, b = rng.sample(names, 2)
        add(a, b, division)
        attempts += 1
    return pairs[:max_pairs]


def _summary(values: list[float], cap: float, raw_values: list[float] | None = None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    raw = np.asarray(raw_values if raw_values is not None else values, dtype=float)
    if not len(arr):
        return {"sample": 0, "cap": float(cap)}
    abs_arr = np.abs(arr)
    abs_raw = np.abs(raw)
    hits = abs_raw >= float(cap) - 1e-9
    return {
        "sample": int(len(arr)),
        "cap": float(cap),
        "mean_abs": float(abs_arr.mean()),
        "p95_abs": float(np.percentile(abs_arr, 95)),
        "max_abs": float(abs_arr.max()),
        "max_raw_abs": float(abs_raw.max()),
        "cap_hits": int(hits.sum()),
        "cap_hit_rate": float(hits.mean()),
        "p95_cap_utilization": float(np.percentile(abs_raw, 95) / cap) if cap else 0.0,
        "max_cap_utilization": float(abs_raw.max() / cap) if cap else 0.0,
    }


def run_matchup_cap_audit(config: UFCCapAuditConfig | None = None) -> dict[str, Any]:
    """Measure whether current UFC matchup guardrails actually clip live full-stack signals.

    This deliberately does not infer an 'optimal' cap from historical outcomes because the
    repository does not preserve historical snapshots of the modern full-stack matchup features.
    It answers the structural question first: are the live caps even binding often enough to be
    a plausible source of fair-line compression?
    """
    config = config or UFCCapAuditConfig()
    rounds = 5 if int(config.rounds) == 5 else 3
    ratings = _active_ratings()
    pairs = _pair_candidates(ratings, int(config.max_pairs), int(config.seed))
    rows: list[dict[str, Any]] = []

    for fighter_a, fighter_b, division in pairs:
        try:
            result = analyze(fighter_a, fighter_b, rounds=rounds)
            dec = result.get("probability_decomposition") or {}
            rows.append({
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "division": division,
                "rating_probability_a": float(dec.get("rating_probability_a", 0.5) or 0.5),
                "performance_adjustment_a": float(dec.get("performance_adjustment_a", 0.0) or 0.0),
                "style_adjustment_a": float(dec.get("style_adjustment_a", 0.0) or 0.0),
                "cardio_adjustment_a": float(dec.get("cardio_adjustment_a", 0.0) or 0.0),
                "damage_adjustment_a": float(dec.get("damage_adjustment_a", 0.0) or 0.0),
                "context_adjustment_a": float(dec.get("context_adjustment_a", 0.0) or 0.0),
                "correlated_raw_a": float(dec.get("correlated_matchup_sum_before_cap_a", 0.0) or 0.0),
                "correlated_capped_a": float(dec.get("correlated_matchup_after_cap_a", 0.0) or 0.0),
                "total_capped_a": float(dec.get("total_adjustment_after_cap_a", 0.0) or 0.0),
                "final_probability_a": float(dec.get("final_probability_a", 0.5) or 0.5),
            })
        except Exception:
            continue

    if not rows:
        return {"available": False, "version": CAP_AUDIT_VERSION, "reason": "No active UFC matchup pairs could be audited."}

    frame = pd.DataFrame(rows)
    frame["correlated_removed_a"] = frame["correlated_raw_a"] - frame["correlated_capped_a"]
    frame["total_raw_a"] = frame["correlated_capped_a"] + frame["context_adjustment_a"]
    frame["total_removed_a"] = frame["total_raw_a"] - frame["total_capped_a"]
    cardio_cap = CAPS["cardio_5r"] if rounds == 5 else CAPS["cardio_3r"]

    summaries = {
        "performance": _summary(frame["performance_adjustment_a"].tolist(), CAPS["performance"]),
        "style": _summary(frame["style_adjustment_a"].tolist(), CAPS["style"]),
        "cardio": _summary(frame["cardio_adjustment_a"].tolist(), cardio_cap),
        "damage": _summary(frame["damage_adjustment_a"].tolist(), CAPS["damage"]),
        "context": _summary(frame["context_adjustment_a"].tolist(), CAPS["context"]),
        "correlated": _summary(frame["correlated_capped_a"].tolist(), CAPS["correlated"], frame["correlated_raw_a"].tolist()),
        "total_non_rating": _summary(frame["total_capped_a"].tolist(), CAPS["total_non_rating"], frame["total_raw_a"].tolist()),
    }

    correlated_hits = int(summaries["correlated"].get("cap_hits", 0) or 0)
    total_hits = int(summaries["total_non_rating"].get("cap_hits", 0) or 0)
    style_hits = int(summaries["style"].get("cap_hits", 0) or 0)

    if correlated_hits == 0 and total_hits == 0:
        recommendation = (
            "Keep the ±8.5% correlated and ±10% total non-rating caps unchanged. In this broad live-stack stress sample they do not bind, "
            "so raising them would not solve probability compression. Any remaining compression is more likely inside component scaling or the rating backbone."
        )
    else:
        recommendation = (
            "Aggregate caps do bind in this live-stack sample, but structural evidence alone is not enough to raise them. "
            "Historical full-stack feature snapshots are required before changing production caps."
        )

    extremes = frame.assign(abs_total=frame["total_raw_a"].abs()).sort_values("abs_total", ascending=False).head(12)
    return {
        "available": True,
        "version": CAP_AUDIT_VERSION,
        "sample": int(len(frame)),
        "rounds": rounds,
        "component_summaries": summaries,
        "style_cap_hits": style_hits,
        "correlated_cap_hits": correlated_hits,
        "total_cap_hits": total_hits,
        "correlated_adjustment_removed_total_abs": float(frame["correlated_removed_a"].abs().sum()),
        "total_adjustment_removed_total_abs": float(frame["total_removed_a"].abs().sum()),
        "recommendation": recommendation,
        "extreme_matchups": extremes.drop(columns=["abs_total"]).to_dict(orient="records"),
        "limitations": [
            "This is a current live-stack structural stress test across active same-division pairings, not a historical outcome backtest.",
            "Macabets does not currently store historical snapshots of every modern Performance/Style/Cardio/Damage feature table, so a leakage-safe historical cap optimization cannot yet be performed exactly.",
            "This audit is intended to determine whether caps are clipping information now. Component scaling should only be changed with leakage-safe outcome evidence.",
        ],
    }
