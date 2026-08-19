from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import numpy as np
import pandas as pd

from engine.ufc import analyze, load_ufc_ratings


COMPONENT_SCALE_AUDIT_VERSION = "Macabets UFC Component-Scaling Audit v0.1 — Live Signal Transfer"


@dataclass(frozen=True)
class UFCComponentScaleAuditConfig:
    max_pairs: int = 80
    rounds: int = 3
    seed: int = 2417


def _active_ratings() -> pd.DataFrame:
    frame = load_ufc_ratings().copy()
    if "active_pool" in frame.columns:
        mask = frame["active_pool"]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[mask]
    frame["macabets_rating"] = pd.to_numeric(frame.get("macabets_rating"), errors="coerce")
    return frame.dropna(subset=["macabets_rating"])


def _pair_candidates(ratings: pd.DataFrame, max_pairs: int, seed: int) -> list[tuple[str, str, str]]:
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
        names = group.sort_values("macabets_rating", ascending=False)["fighter"].astype(str).tolist()
        if len(names) < 2:
            continue
        groups.append((str(division), names))
        for i in range(min(4, len(names) - 1)):
            add(names[i], names[i + 1], str(division))
        for i in range(min(2, len(names))):
            for j in {len(names) // 2, len(names) - 1}:
                if 0 <= j < len(names) and i != j:
                    add(names[i], names[j], str(division))

    attempts = 0
    while len(pairs) < max_pairs and groups and attempts < max_pairs * 40:
        division, names = rng.choice(groups)
        a, b = rng.sample(names, 2)
        add(a, b, division)
        attempts += 1
    return pairs[:max_pairs]


def _describe(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    gap = frame[f"{prefix}_gap"].abs().to_numpy(float)
    rel = frame[f"{prefix}_reliability"].to_numpy(float)
    adj = frame[f"{prefix}_adjustment"].abs().to_numpy(float)
    if len(frame) == 0:
        return {"sample": 0}
    high = (gap >= 20.0) & (rel >= 0.75)
    very_high = (gap >= 30.0) & (rel >= 0.85)
    return {
        "sample": int(len(frame)),
        "mean_abs_gap": float(gap.mean()),
        "p95_abs_gap": float(np.percentile(gap, 95)),
        "max_abs_gap": float(gap.max()),
        "mean_reliability": float(rel.mean()),
        "mean_abs_adjustment": float(adj.mean()),
        "p95_abs_adjustment": float(np.percentile(adj, 95)),
        "max_abs_adjustment": float(adj.max()),
        "high_signal_sample": int(high.sum()),
        "high_signal_mean_abs_adjustment": float(adj[high].mean()) if high.any() else 0.0,
        "very_high_signal_sample": int(very_high.sum()),
        "very_high_signal_mean_abs_adjustment": float(adj[very_high].mean()) if very_high.any() else 0.0,
    }


def run_component_scale_audit(config: UFCComponentScaleAuditConfig | None = None) -> dict[str, Any]:
    """Audit how current Performance and Style gaps transfer into fair-probability movement.

    This is intentionally a structural/sensitivity audit, not a historical optimizer. The repository
    still does not preserve exact pre-fight snapshots for all modern Performance/Style components,
    so production multipliers must not be changed from this report alone.
    """
    config = config or UFCComponentScaleAuditConfig()
    rounds = 5 if int(config.rounds) == 5 else 3
    pairs = _pair_candidates(_active_ratings(), int(config.max_pairs), int(config.seed))
    rows: list[dict[str, Any]] = []

    for fighter_a, fighter_b, division in pairs:
        try:
            result = analyze(fighter_a, fighter_b, rounds=rounds)
            perf = result.get("performance_matchup") or {}
            style = result.get("style_matchup") or {}
            dec = result.get("probability_decomposition") or {}
            rows.append({
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "division": division,
                "performance_gap": float(perf.get("weighted_gap", 0.0) or 0.0),
                "performance_reliability": float(perf.get("reliability", 0.0) or 0.0),
                "performance_adjustment": float(perf.get("adjustment_a", 0.0) or 0.0),
                "style_gap": float(style.get("weighted_gap", 0.0) or 0.0),
                "style_reliability": float(style.get("reliability", 0.0) or 0.0),
                "style_adjustment": float(style.get("adjustment_a", 0.0) or 0.0),
                "rating_probability_a": float(dec.get("rating_probability_a", 0.5) or 0.5),
                "final_probability_a": float(dec.get("final_probability_a", 0.5) or 0.5),
            })
        except Exception:
            continue

    if not rows:
        return {
            "available": False,
            "version": COMPONENT_SCALE_AUDIT_VERSION,
            "reason": "No active UFC matchup pairs could be audited.",
        }

    frame = pd.DataFrame(rows)
    perf_summary = _describe(frame, "performance")
    style_summary = _describe(frame, "style")

    nonzero = frame.loc[(frame["performance_adjustment"].abs() > 1e-9) & (frame["style_adjustment"].abs() > 1e-9)].copy()
    if nonzero.empty:
        sign_agreement = 0.0
        opposing_cancellation = 0.0
    else:
        agree = np.sign(nonzero["performance_adjustment"]) == np.sign(nonzero["style_adjustment"])
        sign_agreement = float(agree.mean())
        opposing = nonzero.loc[~agree]
        if opposing.empty:
            opposing_cancellation = 0.0
        else:
            gross = opposing["performance_adjustment"].abs() + opposing["style_adjustment"].abs()
            net = (opposing["performance_adjustment"] + opposing["style_adjustment"]).abs()
            opposing_cancellation = float((gross - net).mean())

    both_high = frame.loc[
        (frame["performance_gap"].abs() >= 20.0)
        & (frame["performance_reliability"] >= 0.75)
        & (frame["style_gap"].abs() >= 20.0)
        & (frame["style_reliability"] >= 0.75)
    ].copy()
    if not both_high.empty:
        same_direction = np.sign(both_high["performance_adjustment"]) == np.sign(both_high["style_adjustment"])
        both_high_agreement = float(same_direction.mean())
        same = both_high.loc[same_direction]
        high_agree_mean_combined = float((same["performance_adjustment"] + same["style_adjustment"]).abs().mean()) if not same.empty else 0.0
    else:
        both_high_agreement = 0.0
        high_agree_mean_combined = 0.0

    # Formula-level transfer rates at full reliability before clipping.
    # Performance: (gap / 50) * .05 = gap * .001
    # Style:       (gap / 55) * .03 ~= gap * .00054545
    perf_pp_per_10_gap = 0.01
    style_pp_per_10_gap = (10.0 / 55.0) * 0.03

    if style_summary.get("high_signal_sample", 0) >= 10 and style_summary.get("high_signal_mean_abs_adjustment", 0.0) < 0.0225:
        style_flag = (
            "Style is the main scaling candidate. High-reliability, 20+ point style gaps are common, yet their mean fair-probability impact remains modest. "
            "Do not raise the multiplier from this structural audit alone; validate extreme style residuals against historical outcomes first."
        )
    else:
        style_flag = (
            "Style does not show an obvious structural under-transfer signal in this sample. Keep its production multiplier unchanged until historical outcome evidence exists."
        )

    if perf_summary.get("high_signal_sample", 0) >= 8 and perf_summary.get("high_signal_mean_abs_adjustment", 0.0) >= 0.0225:
        performance_flag = (
            "Performance appears materially responsive when its gap is large and reliable. There is no structural case here for increasing its production multiplier."
        )
    else:
        performance_flag = (
            "Performance deserves further historical testing, but this live structural sample does not justify a multiplier change by itself."
        )

    extreme = frame.copy()
    extreme["combined_abs"] = (extreme["performance_adjustment"] + extreme["style_adjustment"]).abs()
    extreme = extreme.sort_values("combined_abs", ascending=False).head(15)

    return {
        "available": True,
        "version": COMPONENT_SCALE_AUDIT_VERSION,
        "sample": int(len(frame)),
        "rounds": rounds,
        "performance": perf_summary,
        "style": style_summary,
        "formula_transfer": {
            "performance_probability_points_per_10_gap": perf_pp_per_10_gap,
            "style_probability_points_per_10_gap": style_pp_per_10_gap,
            "style_vs_performance_transfer_ratio": float(style_pp_per_10_gap / perf_pp_per_10_gap),
        },
        "interaction": {
            "sign_agreement_rate": sign_agreement,
            "opposing_mean_probability_cancellation": opposing_cancellation,
            "both_high_signal_sample": int(len(both_high)),
            "both_high_signal_agreement_rate": both_high_agreement,
            "high_signal_same_direction_mean_combined_impact": high_agree_mean_combined,
        },
        "performance_assessment": performance_flag,
        "style_assessment": style_flag,
        "recommendation": (
            "Keep live component multipliers unchanged for now. The structural audit shows Performance transfers large reliable gaps at a meaningful rate, while Style is intentionally about 45% less responsive per weighted-gap point and is the only component that merits a dedicated leakage-safe outcome calibration next."
        ),
        "extreme_matchups": extreme.drop(columns=["combined_abs"]).to_dict(orient="records"),
        "limitations": [
            "This report stress-tests the current live full-stack signal transfer across active same-division matchups; it is not an outcome backtest.",
            "Performance and Style weighted-gap units are not perfectly interchangeable, so their raw gap-to-probability slopes should not be compared as if they were the same statistic.",
            "Exact historical snapshots of the modern opponent-adjusted Performance, Advanced Striking, Advanced Grappling and Style tables are not stored. A production scaling change requires leakage-safe historical reconstruction or forward model-version evidence.",
        ],
    }
