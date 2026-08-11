"""Opponent-specific NFL trait matchup layer.

This module uses Madden 27 *traits* to describe compatibility between the actual
starters selected by the Footballguys/depth-chart personnel layer. It is a
refinement layer only: raw talent is already represented in team/unit grades, so
trait compatibility is centered near zero and tightly capped before it can move
the projected margin.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from engine.nfl_depth_chart import normalize_player_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MADDEN_PLAYERS = PROJECT_ROOT / "data" / "madden_27_players.csv"

STYLE_ADJUSTMENT_CAP = 0.75


def _num(value: Any, default: float = 67.5) -> float:
    try:
        value = float(value)
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return float(default)


@lru_cache(maxsize=4)
def _load_madden(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "player_name" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["_name_key"] = frame["player_name"].map(normalize_player_name)
    return frame.drop_duplicates("_name_key", keep="first").set_index("_name_key", drop=False)


def _starter_rows(team_data: dict[str, Any], unit: str, raw: pd.DataFrame) -> pd.DataFrame:
    unit_data = ((team_data or {}).get("units") or {}).get(unit) or {}
    players = [p for p in unit_data.get("top_players", []) or [] if isinstance(p, dict) and p.get("starter")]
    rows = []
    for player in players:
        key = normalize_player_name(player.get("name"))
        if key and key in raw.index:
            row = raw.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=raw.columns)


def _mean(frame: pd.DataFrame, columns: list[str], default: float = 67.5) -> float:
    values = []
    for column in columns:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not numeric.empty:
            values.extend(numeric.astype(float).tolist())
    return sum(values) / len(values) if values else float(default)


def _top_mean(frame: pd.DataFrame, column: str, n: int = 3, default: float = 67.5) -> float:
    if column not in frame.columns or frame.empty:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna().sort_values(ascending=False).head(n)
    return float(values.mean()) if not values.empty else float(default)


def _strength(edge: float) -> str:
    magnitude = abs(float(edge))
    if magnitude < 2.0:
        return "Even"
    if magnitude < 4.5:
        return "Slight"
    if magnitude < 7.5:
        return "Clear"
    return "Strong"


def _style_row(label: str, offense: str, defense: str, score: float, reason: str) -> dict[str, Any]:
    score = float(score)
    strength = _strength(score)
    return {
        "Matchup": f"{offense} {label}",
        "Advantage": "Even" if strength == "Even" else (offense if score > 0 else defense),
        "Strength": strength,
        "Edge": round(abs(score), 1),
        "Signed Edge": round(score, 2),
        "Source": "Madden 27 starter traits + Footballguys depth chart",
        "Why": reason,
    }


def _passing_style(offense: str, defense: str, offense_data: dict[str, Any], defense_data: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any]:
    qb = _starter_rows(offense_data, "quarterback", raw)
    rec = _starter_rows(offense_data, "receiving_weapons", raw)
    db = _starter_rows(defense_data, "secondary", raw)
    if qb.empty or rec.empty or db.empty:
        return _style_row("receiver/QB traits vs coverage", offense, defense, 0.0, "Insufficient matched starter traits for a style adjustment.")

    wr = rec[rec.get("position", pd.Series(dtype=str)).astype(str).str.upper().eq("WR")]
    targets = wr if not wr.empty else rec
    speed_edge = _top_mean(targets, "speed", 3) - _top_mean(db, "speed", 3)
    route = _mean(targets, ["short_route_running", "medium_route_running", "deep_route_running"])
    coverage = _mean(db, ["man_coverage", "zone_coverage"])
    route_edge = route - coverage
    release_edge = _mean(targets, ["release"]) - _mean(db, ["press"])
    deep_qb_edge = _mean(qb, ["throw_accuracy_deep", "throw_power"]) - _mean(db, ["play_recognition", "zone_coverage"])

    score = speed_edge * 0.28 + route_edge * 0.32 + release_edge * 0.18 + deep_qb_edge * 0.22
    reason = (
        f"Starter trait edges: receiver speed {speed_edge:+.1f}, route-vs-coverage {route_edge:+.1f}, "
        f"release-vs-press {release_edge:+.1f}, deep-pass-vs-coverage {deep_qb_edge:+.1f}."
    )
    return _style_row("receiver/QB traits vs coverage", offense, defense, score, reason)


def _pass_rush_style(offense: str, defense: str, offense_data: dict[str, Any], defense_data: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any]:
    ol = _starter_rows(offense_data, "offensive_line", raw)
    front = _starter_rows(defense_data, "defensive_front", raw)
    if ol.empty or front.empty:
        return _style_row("OL technique vs pass rush", offense, defense, 0.0, "Insufficient matched line starter traits for a style adjustment.")

    edge = front[front.get("position", pd.Series(dtype=str)).astype(str).str.upper().isin({"DE", "LE", "RE", "EDGE", "LEDG", "REDG", "OLB", "LOLB", "ROLB"})]
    rush = edge if not edge.empty else front
    finesse_edge = _top_mean(ol, "pass_block_finesse", 5) - _top_mean(rush, "finesse_moves", 3)
    power_edge = _top_mean(ol, "pass_block_power", 5) - _top_mean(rush, "power_moves", 3)
    base_edge = _mean(ol, ["pass_block", "awareness"]) - _mean(rush, ["acceleration", "block_shedding", "play_recognition"])
    score = finesse_edge * 0.36 + power_edge * 0.36 + base_edge * 0.28
    reason = (
        f"Trench trait edges: pass-block finesse vs finesse rush {finesse_edge:+.1f}, "
        f"pass-block power vs power rush {power_edge:+.1f}, protection recognition vs disruption {base_edge:+.1f}."
    )
    return _style_row("OL technique vs pass rush", offense, defense, score, reason)


def _run_style(offense: str, defense: str, offense_data: dict[str, Any], defense_data: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any]:
    rb = _starter_rows(offense_data, "running_backs", raw)
    ol = _starter_rows(offense_data, "offensive_line", raw)
    front = pd.concat([
        _starter_rows(defense_data, "defensive_front", raw),
        _starter_rows(defense_data, "linebackers", raw),
    ], ignore_index=True)
    if rb.empty or ol.empty or front.empty:
        return _style_row("run style vs front seven", offense, defense, 0.0, "Insufficient matched starter traits for a run-style adjustment.")

    blocking_edge = _mean(ol, ["run_block", "run_block_power", "run_block_finesse", "impact_blocking"]) - _mean(front, ["block_shedding", "strength", "play_recognition"])
    runner_edge = _mean(rb, ["break_tackle", "bc_vision", "change_of_direction", "trucking"]) - _mean(front, ["tackle", "pursuit", "play_recognition"])
    speed_edge = _mean(rb, ["speed", "acceleration"]) - _mean(front, ["speed", "acceleration", "pursuit"])
    score = blocking_edge * 0.48 + runner_edge * 0.34 + speed_edge * 0.18
    reason = (
        f"Run-game trait edges: blocking vs shedding {blocking_edge:+.1f}, runner creation vs tackling {runner_edge:+.1f}, "
        f"backfield speed vs pursuit {speed_edge:+.1f}."
    )
    return _style_row("run style vs front seven", offense, defense, score, reason)


def _qb_mobility_style(offense: str, defense: str, offense_data: dict[str, Any], defense_data: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any]:
    qb = _starter_rows(offense_data, "quarterback", raw)
    front = pd.concat([
        _starter_rows(defense_data, "defensive_front", raw),
        _starter_rows(defense_data, "linebackers", raw),
    ], ignore_index=True)
    if qb.empty or front.empty:
        return _style_row("QB movement vs contain", offense, defense, 0.0, "Insufficient matched starter traits for a QB-mobility adjustment.")

    mobility = _mean(qb, ["speed", "acceleration", "agility", "throw_on_the_run", "break_sack"])
    contain = _mean(front, ["speed", "acceleration", "pursuit", "tackle", "play_recognition"])
    score = mobility - contain
    reason = f"QB movement/escape profile {mobility:.1f} vs front-seven pursuit/contain profile {contain:.1f}."
    return _style_row("QB movement vs contain", offense, defense, score, reason)


def build_style_matchup_context(
    *,
    away_team: str,
    home_team: str,
    away_data: dict[str, Any],
    home_data: dict[str, Any],
    madden_players_path: Path | str = DEFAULT_MADDEN_PLAYERS,
) -> dict[str, Any]:
    raw = _load_madden(str(Path(madden_players_path).resolve()))
    if raw.empty:
        return {"available": False, "home_margin_adjustment": 0.0, "matchups": [], "summary": "Madden starter traits unavailable."}

    rows = [
        _passing_style(away_team, home_team, away_data, home_data, raw),
        _pass_rush_style(away_team, home_team, away_data, home_data, raw),
        _run_style(away_team, home_team, away_data, home_data, raw),
        _qb_mobility_style(away_team, home_team, away_data, home_data, raw),
        _passing_style(home_team, away_team, home_data, away_data, raw),
        _pass_rush_style(home_team, away_team, home_data, away_data, raw),
        _run_style(home_team, away_team, home_data, away_data, raw),
        _qb_mobility_style(home_team, away_team, home_data, away_data, raw),
    ]

    # Convert trait compatibility into only a small margin refinement. These Madden
    # traits already help form player grades, so this layer can describe *fit* but
    # must never re-award the full talent advantage a second time.
    weights = {"receiver/QB traits": 0.36, "OL technique": 0.30, "run style": 0.22, "QB movement": 0.12}

    def side_score(team: str) -> float:
        total = 0.0
        for row in rows:
            if not str(row.get("Matchup", "")).startswith(team + " "):
                continue
            label = str(row.get("Matchup", ""))[len(team) + 1:]
            key = next((key for key in weights if label.startswith(key)), None)
            if key:
                total += _num(row.get("Signed Edge"), 0.0) * weights[key]
        return total

    away_score = side_score(away_team)
    home_score = side_score(home_team)
    net_home = home_score - away_score
    adjustment = max(-STYLE_ADJUSTMENT_CAP, min(STYLE_ADJUSTMENT_CAP, net_home * 0.035))
    strongest = max(rows, key=lambda row: _num(row.get("Edge"), 0.0)) if rows else None

    # Overall style/LOS verdict. This is a presentation summary of the SAME weighted
    # trait compatibility already used above; it does not add another model adjustment.
    magnitude = abs(net_home)
    if magnitude < 1.5:
        overall_advantage = "Even"
        overall_strength = "Even"
    elif magnitude < 4.0:
        overall_advantage = home_team if net_home > 0 else away_team
        overall_strength = "Slight"
    elif magnitude < 7.0:
        overall_advantage = home_team if net_home > 0 else away_team
        overall_strength = "Clear"
    else:
        overall_advantage = home_team if net_home > 0 else away_team
        overall_strength = "Strong"

    # Rank the individual weighted matchup contributions in home-margin terms so the
    # UI can explain what drove the overall conclusion without simply counting rows.
    weighted_drivers = []
    for row in rows:
        matchup = str(row.get("Matchup", ""))
        offense_team = home_team if matchup.startswith(home_team + " ") else away_team
        label = matchup[len(offense_team) + 1:] if matchup.startswith(offense_team + " ") else matchup
        key = next((key for key in weights if label.startswith(key)), None)
        if not key:
            continue
        signed_edge = _num(row.get("Signed Edge"), 0.0)
        home_contribution = signed_edge * weights[key] if offense_team == home_team else -signed_edge * weights[key]
        beneficiary = home_team if home_contribution > 0 else away_team if home_contribution < 0 else "Even"
        weighted_drivers.append({
            "matchup": matchup,
            "beneficiary": beneficiary,
            "weighted_edge": round(abs(home_contribution), 2),
            "home_contribution": round(home_contribution, 3),
        })

    if overall_advantage == "Even":
        overall_why = (
            "The combined player-style and line-of-scrimmage matchups are too close to give either team a meaningful overall edge."
        )
    else:
        winner_drivers = sorted(
            [d for d in weighted_drivers if d.get("beneficiary") == overall_advantage],
            key=lambda d: d.get("weighted_edge", 0.0),
            reverse=True,
        )[:2]
        driver_labels = []
        for driver in winner_drivers:
            label = str(driver.get("matchup", ""))
            if label.startswith(overall_advantage + " "):
                label = label[len(overall_advantage) + 1:]
            driver_labels.append(label)
        if len(driver_labels) >= 2:
            driver_text = f"{driver_labels[0]} and {driver_labels[1]}"
        elif driver_labels:
            driver_text = driver_labels[0]
        else:
            driver_text = "the combined starter-trait matchups"
        overall_why = (
            f"{overall_advantage} has the {overall_strength.lower()} overall style/LOS advantage after all eight matchups are weighted together. "
            f"The biggest drivers are {driver_text}."
        )

    return {
        "available": True,
        "home_margin_adjustment": round(adjustment, 2),
        "adjustment_cap": STYLE_ADJUSTMENT_CAP,
        "matchups": rows,
        "strongest_edge": strongest,
        "overall_advantage": overall_advantage,
        "overall_strength": overall_strength,
        "overall_edge": round(magnitude, 2),
        "overall_why": overall_why,
        "overall_weighted_drivers": weighted_drivers,
        "summary": "Madden 27 starter traits refine matchup compatibility with a ±0.75-point cap to avoid double counting base talent.",
    }
