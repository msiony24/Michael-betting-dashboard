from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_STATS_METADATA = PROJECT_ROOT / "data" / "nfl" / "player_weekly_stats_metadata.json"

CATEGORY_WEIGHTS = {
    "Quarterback": 0.22,
    "Passing matchup": 0.18,
    "Pass protection": 0.14,
    "Run matchup": 0.12,
    "Overall defense": 0.10,
    "Coaching": 0.08,
    "Special teams": 0.06,
    "Roster continuity": 0.04,
}


def _num(value: Any, default: float = 67.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _strength(gap: float) -> str:
    gap = abs(float(gap))
    if gap < 4.0:
        return "Even"
    if gap < 7.0:
        return "Slight"
    if gap < 11.0:
        return "Clear"
    return "Major"


def _verdict(away_team: str, home_team: str, away_grade: float, home_grade: float) -> tuple[str, str, float]:
    gap = float(away_grade) - float(home_grade)
    strength = _strength(gap)
    if strength == "Even":
        return "Even", strength, abs(gap)
    return (away_team if gap > 0 else home_team), strength, abs(gap)


def _load_metadata() -> dict[str, Any]:
    try:
        payload = json.loads(PLAYER_STATS_METADATA.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _data_mode(personnel_context: Mapping[str, Any]) -> tuple[str, str]:
    meta = _load_metadata()
    fallback = bool(meta.get("fallback_prior"))
    requested = meta.get("requested_season")
    active = meta.get("active_season")
    cap = _num(meta.get("performance_cap"), 0.0)
    if fallback:
        return (
            "Preseason — Madden-heavy",
            f"Madden 27 is the primary personnel prior; {active or 'prior-season'} NFL performance is capped at {cap:.0%} while {requested or 'current-season'} data is unavailable.",
        )
    if cap > 0:
        return (
            "Current-season blend",
            f"Current NFL performance is active and can supply up to {cap:.0%} of player-level evaluation where the sample is available.",
        )
    return (
        str(personnel_context.get("data_mode") or "Madden-heavy personnel baseline"),
        "Madden 27 supplies the roster baseline until current NFL performance data is available.",
    )


def _personnel_row(personnel_context: Mapping[str, Any], startswith: str) -> dict[str, Any] | None:
    for row in personnel_context.get("matchups", []) or []:
        if str(row.get("Matchup", "")).startswith(startswith):
            return row
    return None


def _category_row(category: str, away_team: str, home_team: str, away_grade: float, home_grade: float, source: str, note: str = "") -> dict[str, Any]:
    leader, strength, gap = _verdict(away_team, home_team, away_grade, home_grade)
    return {
        "Category": category,
        "Advantage": leader,
        "Strength": strength,
        "Rating Gap": round(gap, 1),
        away_team: round(float(away_grade), 1),
        home_team: round(float(home_grade), 1),
        "Source": source,
        "Why": note,
    }


def _side_matchup_grade(team: str, opponent: str, personnel_context: Mapping[str, Any], phrase: str) -> tuple[float, float, dict[str, Any] | None]:
    row = _personnel_row(personnel_context, f"{team} {phrase}")
    if not row:
        return 67.5, 67.5, None
    return _num(row.get("Attack Grade")), _num(row.get("Defense Grade")), row


def _answer_label(margin: float, *, reverse: bool = False) -> str:
    value = -margin if reverse else margin
    if value >= 8:
        return "Yes — strong edge"
    if value >= 4:
        return "Lean yes"
    if value <= -8:
        return "No — major concern"
    if value <= -4:
        return "Lean no"
    return "Mixed / close"


def _question_answers(
    *,
    away_team: str,
    home_team: str,
    away_components: Mapping[str, float],
    home_components: Mapping[str, float],
    personnel_context: Mapping[str, Any],
    overall_leader: str,
    football_edge_points: float,
    data_mode: str,
) -> list[dict[str, Any]]:
    teams = (away_team, home_team)
    opponent = {away_team: home_team, home_team: away_team}
    comps = {away_team: away_components, home_team: home_components}

    def matchup_margin(team: str, phrase: str) -> float:
        attack, defense, _ = _side_matchup_grade(team, opponent[team], personnel_context, phrase)
        return attack - defense

    questions: list[dict[str, Any]] = []

    q1 = {"number": 1, "key": "qb_trust", "question": "Do I trust this quarterback in this matchup?", "answers_by_team": {}}
    for team in teams:
        opp = opponent[team]
        qb = _num(comps[team].get("quarterback"))
        pass_margin = matchup_margin(team, "passing attack vs secondary")
        composite = (qb - _num(comps[opp].get("quarterback"))) * 0.45 + pass_margin * 0.55
        qb_gap = qb - _num(comps[opp].get("quarterback"))
        if qb_gap >= 4:
            qb_read = "has the stronger starting quarterback"
        elif qb_gap <= -4:
            qb_read = "is at a quarterback disadvantage"
        else:
            qb_read = "is close enough at quarterback that this is not a major talent mismatch"
        if pass_margin >= 4:
            matchup_read = "The passing matchup is favorable against this secondary."
        elif pass_margin <= -4:
            matchup_read = "The opposing secondary creates a difficult passing matchup."
        else:
            matchup_read = "The passing matchup itself is fairly balanced."
        q1["answers_by_team"][team] = {
            "answer": _answer_label(composite),
            "score": round(composite, 1),
            "reason": f"{team} {qb_read}. {matchup_read}",
        }
    questions.append(q1)

    q2 = {"number": 2, "key": "offensive_line", "question": "Can the offensive line protect and create running lanes?", "answers_by_team": {}}
    for team in teams:
        protection = matchup_margin(team, "pass protection vs defensive front")
        run = matchup_margin(team, "run game vs front seven")
        composite = protection * 0.60 + run * 0.40
        if protection >= 4:
            protection_read = "The offensive line should hold up well in pass protection."
        elif protection <= -4:
            protection_read = "Pass protection is a real concern against this front."
        else:
            protection_read = "Pass protection looks competitive rather than one-sided."
        if run >= 4:
            run_read = "There should also be room to create rushing lanes."
        elif run <= -4:
            run_read = "Creating consistent rushing lanes could be difficult."
        else:
            run_read = "The run-blocking matchup is close."
        q2["answers_by_team"][team] = {
            "answer": _answer_label(composite), "score": round(composite, 1),
            "reason": f"{protection_read} {run_read}",
        }
    questions.append(q2)

    q3 = {"number": 3, "key": "offensive_capability", "question": "Can this offense consistently move the football and score?", "answers_by_team": {}}
    for team in teams:
        pass_m = matchup_margin(team, "passing attack vs secondary")
        run_m = matchup_margin(team, "run game vs front seven")
        offense = _num(comps[team].get("offense")) - _num(comps[opponent[team]].get("defense"))
        composite = pass_m * 0.45 + run_m * 0.25 + offense * 0.30
        positives = sum(v >= 4 for v in (pass_m, run_m, offense))
        negatives = sum(v <= -4 for v in (pass_m, run_m, offense))
        if positives >= 2:
            read = "The offense has multiple workable ways to move the ball rather than depending on one matchup."
        elif negatives >= 2:
            read = "The offense is facing resistance in more than one area, so sustaining drives may be difficult."
        else:
            read = "The offense has a path to success, but the matchup is mixed and execution will matter."
        q3["answers_by_team"][team] = {
            "answer": _answer_label(composite), "score": round(composite, 1),
            "reason": read,
        }
    questions.append(q3)

    q4 = {"number": 4, "key": "run_defense", "question": "Can this defense stop the run?", "answers_by_team": {}}
    for team in teams:
        opp_run = matchup_margin(opponent[team], "run game vs front seven")
        if opp_run >= 4:
            read = f"{opponent[team]} has a meaningful rushing matchup advantage, so {team}'s run defense has a real challenge."
        elif opp_run <= -4:
            read = f"{team}'s front seven has the better matchup and should be able to make the run game work for its yards."
        else:
            read = "The run matchup is close; neither side has an obvious trench advantage."
        q4["answers_by_team"][team] = {
            "answer": _answer_label(opp_run, reverse=True), "score": round(-opp_run, 1),
            "reason": read,
        }
    questions.append(q4)

    q5 = {"number": 5, "key": "pass_defense", "question": "Can this defense stop the pass?", "answers_by_team": {}}
    for team in teams:
        opp_pass = matchup_margin(opponent[team], "passing attack vs secondary")
        opp_prot = matchup_margin(opponent[team], "pass protection vs defensive front")
        composite = -(opp_pass * 0.65 + opp_prot * 0.35)
        if composite >= 4:
            read = "The coverage and pass-rush matchup gives this defense a credible way to disrupt the passing game."
        elif composite <= -4:
            read = "The opponent has too many favorable pieces in the passing matchup for this to grade as a defensive strength."
        else:
            read = "The pass-defense matchup is fairly balanced; pressure and coverage execution should decide it."
        q5["answers_by_team"][team] = {
            "answer": _answer_label(composite), "score": round(composite, 1),
            "reason": read,
        }
    questions.append(q5)

    if overall_leader == "Even":
        better_answer = "No clear overall edge"
    else:
        better_answer = f"{overall_leader} — {abs(football_edge_points):.1f}-point football edge"
    questions.append({
        "number": 6, "key": "better_team", "question": "Which is the better overall football team?",
        "answer": better_answer,
        "reason": "This combines baseline team strength, opponent-specific personnel matchups, home field, schedule, weather, and game-quality context once each.",
    })

    exploit_rows = list(personnel_context.get("matchups", []) or []) + list(personnel_context.get("style_matchups", []) or [])
    ranked = sorted(
        [r for r in exploit_rows if r.get("Advantage") not in {None, "Even"}],
        key=lambda r: _num(r.get("Edge"), 0.0), reverse=True,
    )
    exploit_text = "; ".join(
        f"{r.get('Advantage')}: {r.get('Matchup')} ({str(r.get('Strength','')).lower()})" for r in ranked[:3]
    ) or "No large personnel mismatch is currently identified."
    questions.append({
        "number": 7, "key": "exploits", "question": "What can each team exploit?",
        "answer": exploit_text,
        "reason": "Exploit paths come from direct offense-versus-defense personnel matchups rather than generic team-grade counting.",
    })

    critical_team = overall_leader if overall_leader != "Even" else "No clear side"
    questions.append({
        "number": 8, "key": "critical_plays", "question": "Which team is more likely to execute the critical plays?",
        "answer": critical_team,
        "reason": (
            f"The current {data_mode.lower()} profile gives {critical_team if critical_team != 'No clear side' else 'neither team'} the stronger combined QB, matchup and team-context position. "
            "Late-down/red-zone data will gain influence as current-season samples become available."
        ),
    })
    return questions


def build_matchup_intelligence(
    *,
    away_team: str,
    home_team: str,
    away_components: Mapping[str, float],
    home_components: Mapping[str, float],
    away_power: float,
    home_power: float,
    home_field_points: float,
    weather_home_adjustment: float = 0.0,
    schedule_home_adjustment: float = 0.0,
    game_quality_home_adjustment: float = 0.0,
    scheme_home_adjustment: float = 0.0,
    los_home_adjustment: float = 0.0,
    situational_home_adjustment: float = 0.0,
    opponent_adjusted_home_adjustment: float = 0.0,
    personnel_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    personnel = dict(personnel_context or {})
    matchup_adjustment = _num(personnel.get("home_margin_adjustment"), 0.0)

    # One coherent football edge. Base power already contains team-level performance;
    # only the opponent-specific personnel mismatch adjustment is added on top.
    base_team_edge = float(home_power) - float(away_power)
    context_edge = (
        float(home_field_points)
        + float(weather_home_adjustment)
        + float(schedule_home_adjustment)
        + float(game_quality_home_adjustment)
        + float(scheme_home_adjustment)
        + float(los_home_adjustment)
        + float(situational_home_adjustment)
        + float(opponent_adjusted_home_adjustment)
    )
    football_home_edge = base_team_edge + context_edge + matchup_adjustment

    if abs(football_home_edge) < 0.5:
        overall_leader = "Even"
        overall_strength = "Toss-up"
    else:
        overall_leader = home_team if football_home_edge > 0 else away_team
        mag = abs(football_home_edge)
        overall_strength = "Slight" if mag < 2.0 else "Moderate" if mag < 4.5 else "Strong"

    source_mode, source_note = _data_mode(personnel)
    categories: list[dict[str, Any]] = []

    # Direct team-level categories.
    categories.append(_category_row(
        "Quarterback", away_team, home_team,
        _num(away_components.get("quarterback")), _num(home_components.get("quarterback")),
        source_mode, "Quarterback is weighted more heavily than secondary categories."
    ))
    categories.append(_category_row(
        "Overall Defense", away_team, home_team,
        _num(away_components.get("defense")), _num(home_components.get("defense")),
        "NFL team-state + roster blend"
    ))
    categories.append(_category_row(
        "Coaching", away_team, home_team,
        _num(away_components.get("coaching")), _num(home_components.get("coaching")),
        "2026 sourced head-coach model"
    ))
    categories.append(_category_row(
        "Special Teams", away_team, home_team,
        _num(away_components.get("special_teams")), _num(home_components.get("special_teams")),
        "NFL team-state / special-teams baseline"
    ))
    categories.append(_category_row(
        "Roster Continuity", away_team, home_team,
        _num(away_components.get("continuity")), _num(home_components.get("continuity")),
        "Automated prior-season starter retention"
    ))

    # Opponent-specific conflicts from the personnel engine. These replace the old
    # generic running/receiving/OL/DL category comparisons.
    for row in personnel.get("matchups", []) or []:
        categories.append({
            "Category": str(row.get("Matchup", "Personnel matchup")),
            "Advantage": row.get("Advantage", "Even"),
            "Strength": row.get("Strength", "Even"),
            "Rating Gap": round(_num(row.get("Edge"), 0.0), 1),
            away_team: "—",
            home_team: "—",
            "Source": row.get("Source", source_mode),
            "Why": f"Attack {row.get('Attack Grade', '—')} vs defense {row.get('Defense Grade', '—')}",
        })

    # Starter-level trait compatibility is reported separately from the broad
    # unit mismatch rows. Its scoreboard influence is already contained in the
    # tightly capped personnel adjustment, so it is not counted again below.
    for row in personnel.get("style_matchups", []) or []:
        categories.append({
            "Category": str(row.get("Matchup", "Style matchup")),
            "Advantage": row.get("Advantage", "Even"),
            "Strength": row.get("Strength", "Even"),
            "Rating Gap": round(_num(row.get("Edge"), 0.0), 1),
            away_team: "—",
            home_team: "—",
            "Source": row.get("Source", "Madden 27 starter traits"),
            "Why": row.get("Why", "Opponent-specific starter trait compatibility."),
        })

    # Weighted directional drivers. No category-count scoreboard.
    drivers = []
    qb_gap = _num(home_components.get("quarterback")) - _num(away_components.get("quarterback"))
    drivers.append(("Quarterback", home_team if qb_gap > 0 else away_team, abs(qb_gap) * CATEGORY_WEIGHTS["Quarterback"], qb_gap))
    for row in personnel.get("matchups", []) or []:
        if row.get("Advantage") == "Even":
            continue
        label = str(row.get("Matchup", ""))
        if "passing attack" in label:
            weight = CATEGORY_WEIGHTS["Passing matchup"]
        elif "pass protection" in label:
            weight = CATEGORY_WEIGHTS["Pass protection"]
        else:
            weight = CATEGORY_WEIGHTS["Run matchup"]
        drivers.append((label, str(row.get("Advantage")), _num(row.get("Edge"), 0.0) * weight, _num(row.get("Edge"), 0.0)))
    for key, label, weight in (
        ("defense", "Overall defense", CATEGORY_WEIGHTS["Overall defense"]),
        ("coaching", "Coaching", CATEGORY_WEIGHTS["Coaching"]),
        ("special_teams", "Special teams", CATEGORY_WEIGHTS["Special teams"]),
        ("continuity", "Roster continuity", CATEGORY_WEIGHTS["Roster continuity"]),
    ):
        gap = _num(home_components.get(key)) - _num(away_components.get(key))
        if abs(gap) >= 4.0:
            drivers.append((label, home_team if gap > 0 else away_team, abs(gap) * weight, abs(gap)))
    drivers.sort(key=lambda item: item[2], reverse=True)
    top_drivers = [
        {"factor": d[0], "leader": d[1], "weighted_importance": round(d[2], 2), "raw_gap": round(abs(d[3]), 1)}
        for d in drivers[:5]
    ]

    questions = _question_answers(
        away_team=away_team, home_team=home_team,
        away_components=away_components, home_components=home_components,
        personnel_context=personnel, overall_leader=overall_leader,
        football_edge_points=football_home_edge, data_mode=source_mode,
    )

    return {
        "version": "Unified NFL Matchup Intelligence v2 — Starter Trait Compatibility",
        "available": True,
        "overall_leader": overall_leader,
        "overall_strength": overall_strength,
        "football_home_edge_points": round(football_home_edge, 2),
        "football_edge_points": round(abs(football_home_edge), 2),
        "base_team_edge_home": round(base_team_edge, 2),
        "context_edge_home": round(context_edge, 2),
        "matchup_adjustment_home": round(matchup_adjustment, 2),
        "scheme_adjustment_home": round(float(scheme_home_adjustment), 2),
        "los_adjustment_home": round(float(los_home_adjustment), 2),
        "base_personnel_adjustment_home": round(_num(personnel.get("base_home_margin_adjustment"), 0.0), 2),
        "style_adjustment_home": round(_num(personnel.get("style_home_margin_adjustment"), 0.0), 2),
        "style_matchups": list(personnel.get("style_matchups", []) or []),
        "overall_style_advantage": personnel.get("overall_style_advantage"),
        "overall_style_strength": personnel.get("overall_style_strength"),
        "overall_style_edge": personnel.get("overall_style_edge"),
        "overall_style_why": personnel.get("overall_style_why"),
        "overall_style_weighted_drivers": list(personnel.get("overall_style_weighted_drivers", []) or []),
        "data_mode": source_mode,
        "data_note": source_note,
        "categories": categories,
        "top_drivers": top_drivers,
        "questions": questions,
        "summary": (
            f"{overall_leader if overall_leader != 'Even' else 'Neither team'} holds the {overall_strength.lower()} overall football edge "
            f"after baseline team strength, opponent-specific personnel matchups, scheme compatibility, real line-of-scrimmage interaction, opponent-adjusted performance and game context are combined without category-count scoring."
        ),
        "guardrail": "Team-level performance enters base power once. Unit mismatches, starter-trait compatibility, scheme compatibility, real line-of-scrimmage interaction and opponent-quality correction are small refinements with hard caps; the same talent or performance signal is never re-awarded at full strength.",
    }
