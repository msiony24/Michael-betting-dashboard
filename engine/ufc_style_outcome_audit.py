from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np
import pandas as pd

from engine.ufc_strength_audit import UFCStrengthAuditConfig, run_strength_backbone_audit


STYLE_OUTCOME_AUDIT_VERSION = "Macabets UFC Style Outcome Audit v0.1 — Leakage-Safe Residual Test"


@dataclass(frozen=True)
class UFCStyleOutcomeAuditConfig:
    start_date: str = "2018-01-01"
    max_bouts: int = 1400
    recent_fights: int = 8
    min_prior_fights: int = 3
    holdout_fraction: float = 0.30
    min_group_sample: int = 30


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _fight_minutes(row: pd.Series) -> float:
    try:
        rnd = max(1, int(float(row.get("round", 1))))
    except (TypeError, ValueError):
        rnd = 1
    seconds = 0
    text = str(row.get("time", "") or "")
    if ":" in text:
        try:
            minute, second = text.split(":", 1)
            seconds = int(minute) * 60 + int(second)
        except (TypeError, ValueError):
            seconds = 0
    return max(1.0 / 60.0, ((rnd - 1) * 300 + seconds) / 60.0)


def _prepare_rows(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    frame = frame.dropna(subset=["event_date", "fight_url", "fighter", "opponent"]).copy()
    stat_cols = [
        c for c in (
            "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
            "kd", "sub_att", "control_seconds",
        ) if c in frame.columns
    ]
    opp = frame[["fight_url", "fighter"] + stat_cols].copy()
    opp = opp.rename(columns={"fighter": "opponent", **{c: f"opponent_{c}" for c in stat_cols}})
    frame = frame.merge(opp, on=["fight_url", "opponent"], how="left")
    frame["_fighter_key"] = frame["fighter"].astype(str).str.casefold()
    return frame.sort_values(["event_date", "fight_url", "fighter"]).reset_index(drop=True)


def _profile(rows: pd.DataFrame) -> dict[str, float | int | None]:
    if rows.empty:
        return {"sample": 0, "completeness": 0.0}
    mins = np.array([_fight_minutes(row) for _, row in rows.iterrows()], dtype=float)
    total_minutes = float(mins.sum())

    def total(col: str) -> tuple[float, bool]:
        if col not in rows.columns:
            return 0.0, False
        values = pd.to_numeric(rows[col], errors="coerce")
        return float(values.fillna(0).sum()), bool(values.notna().any())

    sig_l, sig_l_ok = total("sig_str_landed")
    sig_a, sig_a_ok = total("sig_str_attempted")
    opp_sig_l, opp_sig_l_ok = total("opponent_sig_str_landed")
    opp_sig_a, opp_sig_a_ok = total("opponent_sig_str_attempted")
    td_l, td_l_ok = total("td_landed")
    td_a, td_a_ok = total("td_attempted")
    opp_td_l, opp_td_l_ok = total("opponent_td_landed")
    opp_td_a, opp_td_a_ok = total("opponent_td_attempted")
    kd, kd_ok = total("kd")
    opp_kd, opp_kd_ok = total("opponent_kd")
    subs, sub_ok = total("sub_att")
    ctl, ctl_ok = total("control_seconds")
    opp_ctl, opp_ctl_ok = total("opponent_control_seconds")

    accuracy = _safe_div(sig_l, sig_a) if sig_a_ok else None
    defense = (1.0 - opp_sig_l / opp_sig_a) if opp_sig_a_ok and opp_sig_a > 0 else None
    td_accuracy = _safe_div(td_l, td_a) if td_a_ok else None
    td_defense = (1.0 - opp_td_l / opp_td_a) if opp_td_a_ok and opp_td_a > 0 else None
    control_share = _safe_div(ctl, ctl + opp_ctl) if (ctl_ok or opp_ctl_ok) else None

    details = [sig_a_ok, opp_sig_a_ok, td_a_ok, opp_td_a_ok, ctl_ok or opp_ctl_ok]
    return {
        "sample": int(len(rows)),
        "completeness": float(sum(details) / len(details)),
        "sig_accuracy": accuracy,
        "sig_defense": defense,
        "kd_per15": float(kd / total_minutes * 15.0) if kd_ok and total_minutes > 0 else None,
        "kd_abs_per15": float(opp_kd / total_minutes * 15.0) if opp_kd_ok and total_minutes > 0 else None,
        "td_per15": float(td_l / total_minutes * 15.0) if td_l_ok and total_minutes > 0 else None,
        "td_accuracy": td_accuracy,
        "td_defense": td_defense,
        "sub_per15": float(subs / total_minutes * 15.0) if sub_ok and total_minutes > 0 else None,
        "control_share": control_share,
        "pace": float(sig_a / total_minutes) if sig_a_ok and total_minutes > 0 else None,
    }


def _robust_scaler(profiles: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    metrics = [
        "sig_accuracy", "sig_defense", "kd_per15", "kd_abs_per15", "td_per15",
        "td_accuracy", "td_defense", "sub_per15", "control_share", "pace",
    ]
    out: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = np.array([
            float(p[metric]) for p in profiles
            if p.get(metric) is not None and np.isfinite(float(p[metric]))
        ], dtype=float)
        if len(values) < 20:
            out[metric] = (0.0, 1.0)
            continue
        med = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        scale = float(max((q75 - q25) / 1.349, np.std(values) * 0.35, 1e-6))
        out[metric] = (med, scale)
    return out


def _z(profile: dict[str, Any], metric: str, scaler: dict[str, tuple[float, float]], invert: bool = False) -> float | None:
    value = profile.get(metric)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    center, scale = scaler.get(metric, (0.0, 1.0))
    score = (number - center) / max(scale, 1e-6)
    return float(-score if invert else score)


def _avg(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def _style_proxy(a: dict[str, Any], b: dict[str, Any], scaler: dict[str, tuple[float, float]]) -> tuple[float, float, int]:
    # Fixed pre-2018 robust scaling makes this a leakage-safe historical proxy for the
    # current attack-vs-defense Style architecture. It intentionally does not claim to
    # recreate today's advanced percentile tables exactly.
    a_str_attack = _avg([_z(a, "sig_accuracy", scaler), _z(a, "kd_per15", scaler)])
    b_str_attack = _avg([_z(b, "sig_accuracy", scaler), _z(b, "kd_per15", scaler)])
    a_str_def = _avg([_z(a, "sig_defense", scaler), _z(a, "kd_abs_per15", scaler, invert=True)])
    b_str_def = _avg([_z(b, "sig_defense", scaler), _z(b, "kd_abs_per15", scaler, invert=True)])

    a_w_attack = _avg([_z(a, "td_per15", scaler), _z(a, "td_accuracy", scaler), _z(a, "control_share", scaler)])
    b_w_attack = _avg([_z(b, "td_per15", scaler), _z(b, "td_accuracy", scaler), _z(b, "control_share", scaler)])
    a_w_def = _z(a, "td_defense", scaler)
    b_w_def = _z(b, "td_defense", scaler)

    a_g_attack = _avg([_z(a, "sub_per15", scaler), _z(a, "control_share", scaler)])
    b_g_attack = _avg([_z(b, "sub_per15", scaler), _z(b, "control_share", scaler)])
    a_dur = _avg([_z(a, "sig_defense", scaler), _z(a, "kd_abs_per15", scaler, invert=True)])
    b_dur = _avg([_z(b, "sig_defense", scaler), _z(b, "kd_abs_per15", scaler, invert=True)])
    a_g_def = _avg([_z(a, "td_defense", scaler), a_dur])
    b_g_def = _avg([_z(b, "td_defense", scaler), b_dur])

    a_pace = _z(a, "pace", scaler)
    b_pace = _z(b, "pace", scaler)
    a_attr_def = _avg([a_dur, a_pace])
    b_attr_def = _avg([b_dur, b_pace])

    interactions: list[tuple[float, float]] = []
    specs = [
        (a_str_attack, b_str_def, b_str_attack, a_str_def, 0.36),
        (a_w_attack, b_w_def, b_w_attack, a_w_def, 0.30),
        (a_g_attack, b_g_def, b_g_attack, a_g_def, 0.18),
        (a_pace, b_attr_def, b_pace, a_attr_def, 0.16),
    ]
    for aa, bd, ba, ad, weight in specs:
        if aa is None or bd is None or ba is None or ad is None:
            continue
        interactions.append((((aa - bd) - (ba - ad)), float(weight)))
    if not interactions:
        return 0.0, 0.0, 0
    weight_sum = sum(weight for _, weight in interactions)
    gap = sum(value * weight for value, weight in interactions) / weight_sum
    sample_rel = min(1.0, min(int(a.get("sample", 0)), int(b.get("sample", 0))) / 6.0)
    completeness = min(float(a.get("completeness", 0.0)), float(b.get("completeness", 0.0)))
    coverage = len(interactions) / 4.0
    reliability = sample_rel * (0.40 + 0.60 * completeness) * coverage
    return float(gap), float(reliability), int(len(interactions))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) if len(p) else float("nan")


def _fit_beta(p: np.ndarray, signal: np.ndarray, y: np.ndarray) -> float:
    base_logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    best_beta = 0.0
    best_loss = float("inf")
    for beta in np.linspace(-0.50, 0.50, 201):
        pred = 1.0 / (1.0 + np.exp(-(base_logit + beta * signal)))
        loss = _log_loss(pred, y)
        if loss < best_loss:
            best_loss = loss
            best_beta = float(beta)
    return best_beta


def _apply_beta(p: np.ndarray, signal: np.ndarray, beta: float) -> np.ndarray:
    base_logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return 1.0 / (1.0 + np.exp(-(base_logit + float(beta) * signal)))


def run_style_outcome_audit(fights: pd.DataFrame, *, config: UFCStyleOutcomeAuditConfig | None = None) -> dict[str, Any]:
    config = config or UFCStyleOutcomeAuditConfig()
    frame = _prepare_rows(fights)
    if frame.empty:
        return {"available": False, "version": STYLE_OUTCOME_AUDIT_VERSION, "reason": "No UFC fight history."}

    strength = run_strength_backbone_audit(
        fights,
        config=UFCStrengthAuditConfig(
            start_date=config.start_date,
            min_prior_fights=config.min_prior_fights,
            max_bouts=0,
            holdout_fraction=config.holdout_fraction,
            min_group_sample=config.min_group_sample,
        ),
    )
    if not strength.get("available"):
        return {"available": False, "version": STYLE_OUTCOME_AUDIT_VERSION, "reason": "Strength backbone audit was unavailable."}
    strength_records = pd.DataFrame(strength.get("records", []))
    if strength_records.empty:
        return {"available": False, "version": STYLE_OUTCOME_AUDIT_VERSION, "reason": "No eligible Strength records."}

    # Build fighter histories once. Slicing by date guarantees only pre-fight rows enter a profile.
    histories: dict[str, pd.DataFrame] = {
        key: group.sort_values("event_date").reset_index(drop=True)
        for key, group in frame.groupby("_fighter_key", sort=False)
    }

    start = pd.Timestamp(config.start_date)
    seed_profiles: list[dict[str, Any]] = []
    for group in histories.values():
        prior = group.loc[group["event_date"] < start].tail(config.recent_fights)
        if len(prior) >= config.min_prior_fights:
            seed_profiles.append(_profile(prior))
    scaler = _robust_scaler(seed_profiles)

    out: list[dict[str, Any]] = []
    for row in strength_records.to_dict(orient="records"):
        date = pd.Timestamp(row["event_date"])
        a = str(row["fighter_a"])
        b = str(row["fighter_b"])
        ha = histories.get(a.casefold())
        hb = histories.get(b.casefold())
        if ha is None or hb is None:
            continue
        pa_rows = ha.loc[ha["event_date"] < date].tail(config.recent_fights)
        pb_rows = hb.loc[hb["event_date"] < date].tail(config.recent_fights)
        if len(pa_rows) < config.min_prior_fights or len(pb_rows) < config.min_prior_fights:
            continue
        pa = _profile(pa_rows)
        pb = _profile(pb_rows)
        gap, reliability, coverage = _style_proxy(pa, pb, scaler)
        if reliability <= 0 or coverage < 2:
            continue
        out.append({
            **row,
            "style_proxy_gap": gap,
            "style_reliability": reliability,
            "style_signal": gap * reliability,
            "style_coverage": coverage,
        })

    result = pd.DataFrame(out)
    if config.max_bouts and len(result) > int(config.max_bouts):
        result = result.tail(int(config.max_bouts)).reset_index(drop=True)
    if len(result) < 100:
        return {"available": False, "version": STYLE_OUTCOME_AUDIT_VERSION, "reason": f"Only {len(result)} eligible style-proxy bouts were available."}

    split = max(1, min(len(result) - 1, int(round(len(result) * (1.0 - config.holdout_fraction)))))
    train = result.iloc[:split].copy()
    holdout = result.iloc[split:].copy()
    signal_scale = float(train["style_signal"].std(ddof=0) or 1.0)
    train_signal = train["style_signal"].to_numpy(float) / signal_scale
    hold_signal = holdout["style_signal"].to_numpy(float) / signal_scale
    beta = _fit_beta(train["p_strength"].to_numpy(float), train_signal, train["a_win"].to_numpy(float))
    holdout["p_style_augmented"] = _apply_beta(holdout["p_strength"].to_numpy(float), hold_signal, beta)

    y = holdout["a_win"].to_numpy(float)
    p0 = holdout["p_strength"].to_numpy(float)
    p1 = holdout["p_style_augmented"].to_numpy(float)
    baseline = {"brier": _brier(p0, y), "log_loss": _log_loss(p0, y)}
    augmented = {"brier": _brier(p1, y), "log_loss": _log_loss(p1, y)}

    holdout["abs_style_signal"] = holdout["style_signal"].abs()
    quantiles = holdout["abs_style_signal"].quantile([0.50, 0.75, 0.90]).to_dict()
    groups: list[dict[str, Any]] = []
    for label, threshold in [
        ("Top 50% style magnitude", quantiles.get(0.50, 0.0)),
        ("Top 25% style magnitude", quantiles.get(0.75, 0.0)),
        ("Top 10% style magnitude", quantiles.get(0.90, 0.0)),
    ]:
        part = holdout.loc[holdout["abs_style_signal"] >= float(threshold)].copy()
        if len(part) < 10:
            continue
        style_a = part["style_signal"] >= 0
        style_side_win = np.where(style_a, part["a_win"].to_numpy(int), 1 - part["a_win"].to_numpy(int))
        style_side_base_p = np.where(style_a, part["p_strength"].to_numpy(float), 1 - part["p_strength"].to_numpy(float))
        groups.append({
            "group": label,
            "sample": int(len(part)),
            "style_side_win_rate": float(np.mean(style_side_win)),
            "baseline_expected_win_rate": float(np.mean(style_side_base_p)),
            "outperformance": float(np.mean(style_side_win) - np.mean(style_side_base_p)),
            "mean_abs_signal": float(part["abs_style_signal"].mean()),
        })

    aligned = np.sign(holdout["style_signal"].to_numpy(float)) == np.sign(holdout["p_strength"].to_numpy(float) - 0.5)
    interaction_groups: list[dict[str, Any]] = []
    for label, mask in [("Style agrees with Strength", aligned), ("Style opposes Strength", ~aligned)]:
        part = holdout.loc[mask].copy()
        if len(part) < 20:
            continue
        base_correct = ((part["p_strength"] >= 0.5).to_numpy() == (part["a_win"] == 1).to_numpy()).mean()
        interaction_groups.append({"group": label, "sample": int(len(part)), "strength_winner_accuracy": float(base_correct), "mean_abs_style_signal": float(part["style_signal"].abs().mean())})

    brier_gain = float(baseline["brier"] - augmented["brier"])
    log_gain = float(baseline["log_loss"] - augmented["log_loss"])
    if beta <= 0.01 or brier_gain <= 0 or log_gain <= 0:
        recommendation = "Do not increase the live Style multiplier. This leakage-safe proxy does not show reliable incremental holdout value beyond Strength."
    elif beta < 0.08 or brier_gain < 0.0005:
        recommendation = "Style shows positive incremental signal, but the holdout gain is small. Keep the live multiplier unchanged and gather more forward evidence."
    else:
        recommendation = "Style shows meaningful incremental holdout value. A small controlled multiplier trial is defensible, but should still be validated on additional time windows before production promotion."

    return {
        "available": True,
        "version": STYLE_OUTCOME_AUDIT_VERSION,
        "sample": int(len(result)),
        "train_sample": int(len(train)),
        "holdout_sample": int(len(holdout)),
        "date_start": str(result["event_date"].min()),
        "date_end": str(result["event_date"].max()),
        "fitted_beta": float(beta),
        "signal_scale": signal_scale,
        "holdout_baseline": baseline,
        "holdout_style_augmented": augmented,
        "brier_gain": brier_gain,
        "log_loss_gain": log_gain,
        "magnitude_groups": groups,
        "interaction_groups": interaction_groups,
        "recommendation": recommendation,
        "limitations": [
            "This is leakage-safe, but it is a historical proxy for the current Style engine rather than an exact replay of today's Advanced Striking and Advanced Grappling percentile tables.",
            "All style features use only each fighter's bouts completed before the target fight. Scaling constants are fixed from pre-2018 fighter histories and are not learned from the audited outcomes.",
            "The fitted Style coefficient is trained only on the earlier chronological segment and evaluated on the later holdout. It is diagnostic and is never applied to the live model automatically.",
            "Historical sportsbook prices are not included, so this measures incremental predictive value versus the Strength backbone, not betting ROI or CLV.",
        ],
        "records": holdout.tail(250).to_dict(orient="records"),
    }
