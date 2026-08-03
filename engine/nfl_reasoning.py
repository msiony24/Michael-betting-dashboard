"""Evaluate a user's NFL betting thesis against Macabets matchup inputs.

The first version is deliberately deterministic. It does not pretend to know
facts that are absent from the current data snapshot. It extracts football
claims from the user's explanation, checks them against matchup interactions,
and distinguishes supported points from assumptions that still need evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable


TOPIC_KEYWORDS = {
    "quarterback_vs_coverage": (
        "qb", "quarterback", "passing", "pass game", "throw", "coverage"
    ),
    "protection_vs_pass_rush": (
        "offensive line", "o-line", "oline", "protection", "protect", "pressure",
        "pass rush", "sack", "trenches"
    ),
    "receivers_vs_secondary": (
        "receiver", "receivers", "wideout", "wr", "tight end", "te", "secondary",
        "corner", "corners", "coverage"
    ),
    "run_structure_vs_front": (
        "run game", "running game", "rushing", "rush offense", "running back",
        "rb", "ground game", "front seven", "run defense"
    ),
    "defense_vs_offense": (
        "defense", "defensive", "stop", "disrupt", "turnover", "takeaway"
    ),
    "coaching": (
        "coach", "coaching", "scheme", "adjustment", "game plan", "play calling"
    ),
    "home_field": (
        "home", "home field", "crowd", "road", "travel"
    ),
}

UNVERIFIED_KEYWORDS = {
    "injuries": ("injury", "injuries", "injured", "hurt", "healthy", "ruled out", "inactive"),
    "recent form": ("recent form", "last game", "last few", "momentum", "hot", "cold"),
    "weather": ("weather", "wind", "rain", "snow", "temperature"),
    "turnover expectation": ("turnover", "turnovers", "interception", "fumble"),
    "specific scheme tendency": (
        "man coverage", "zone coverage", "cover 2", "cover two", "cover 3", "cover three",
        "blitz rate", "play action", "outside zone", "inside zone", "gap scheme"
    ),
}


@dataclass(frozen=True)
class ReasoningPoint:
    topic: str
    label: str
    status: str
    edge: float
    evidence: str


def _value(profile: dict, key: str, default: float = 50.0) -> float:
    try:
        return float(profile.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)


def build_conflict_scores(
    selected_team: str,
    opponent: str,
    selected_profile: dict,
    opponent_profile: dict,
    *,
    selected_is_home: bool,
    home_field_points: float,
) -> dict[str, dict]:
    """Create opponent-specific matchup interactions from existing team grades."""
    selected_qb = _value(selected_profile, "quarterback")
    selected_ol = _value(selected_profile, "offensive_line")
    selected_weapons = _value(selected_profile, "skill_positions")
    selected_offense = _value(selected_profile, "offense")
    selected_defense = _value(selected_profile, "defense")
    selected_dl = _value(selected_profile, "defensive_line")
    selected_coaching = _value(selected_profile, "coaching")

    opponent_secondary = _value(opponent_profile, "secondary")
    opponent_dl = _value(opponent_profile, "defensive_line")
    opponent_ol = _value(opponent_profile, "offensive_line")
    opponent_defense = _value(opponent_profile, "defense")
    opponent_offense = _value(opponent_profile, "offense")
    opponent_coaching = _value(opponent_profile, "coaching")

    run_attack = selected_offense * 0.45 + selected_ol * 0.55
    run_resistance = opponent_defense * 0.55 + opponent_dl * 0.45

    home_edge = float(home_field_points) if selected_is_home else -float(home_field_points)

    return {
        "quarterback_vs_coverage": {
            "label": "Quarterback vs opposing coverage",
            "edge": selected_qb - opponent_secondary,
            "selected": selected_qb,
            "opponent": opponent_secondary,
        },
        "protection_vs_pass_rush": {
            "label": "Pass protection vs opposing pass rush",
            "edge": selected_ol - opponent_dl,
            "selected": selected_ol,
            "opponent": opponent_dl,
        },
        "receivers_vs_secondary": {
            "label": "Receiving weapons vs opposing secondary",
            "edge": selected_weapons - opponent_secondary,
            "selected": selected_weapons,
            "opponent": opponent_secondary,
        },
        "run_structure_vs_front": {
            "label": "Run structure vs opposing front",
            "edge": run_attack - run_resistance,
            "selected": run_attack,
            "opponent": run_resistance,
        },
        "defense_vs_offense": {
            "label": "Defensive disruption vs opposing offense",
            "edge": selected_defense - opponent_offense,
            "selected": selected_defense,
            "opponent": opponent_offense,
        },
        "pass_rush_vs_protection": {
            "label": "Pass rush vs opposing protection",
            "edge": selected_dl - opponent_ol,
            "selected": selected_dl,
            "opponent": opponent_ol,
        },
        "coaching": {
            "label": "Coaching and adjustment profile",
            "edge": selected_coaching - opponent_coaching,
            "selected": selected_coaching,
            "opponent": opponent_coaching,
        },
        "home_field": {
            "label": "Venue position",
            "edge": home_edge,
            "selected": home_edge,
            "opponent": 0.0,
        },
    }


def _status_for_edge(edge: float) -> str:
    if edge >= 4.0:
        return "Confirmed"
    if edge >= 1.5:
        return "Mostly confirmed"
    if edge > -1.5:
        return "Mixed"
    return "Pushback"


def _evidence_sentence(
    selected_team: str,
    opponent: str,
    conflict: dict,
    status: str,
) -> str:
    label = conflict["label"]
    edge = float(conflict["edge"])
    selected = float(conflict["selected"])
    opposing = float(conflict["opponent"])

    if status == "Confirmed":
        return (
            f"{selected_team} grades {selected:.1f} in this area against {opponent}'s "
            f"{opposing:.1f} resistance, creating a clear {edge:+.1f} matchup edge in {label.lower()}."
        )
    if status == "Mostly confirmed":
        return (
            f"The matchup supports your point, but only modestly: {selected_team} holds a "
            f"{edge:+.1f} edge in {label.lower()} ({selected:.1f} vs {opposing:.1f})."
        )
    if status == "Mixed":
        return (
            f"Macabets sees this battle as close. {selected_team} is at {selected:.1f} versus "
            f"{opponent} at {opposing:.1f}, a difference of only {edge:+.1f}."
        )
    return (
        f"This is where Macabets pushes back: {selected_team}'s {selected:.1f} profile is below "
        f"{opponent}'s {opposing:.1f} opposing unit, a {edge:+.1f} disadvantage in {label.lower()}."
    )


def challenge_reasoning(
    *,
    reasoning: str,
    selected_team: str,
    opponent: str,
    selected_profile: dict,
    opponent_profile: dict,
    projected_winner: str,
    selected_is_home: bool,
    home_field_points: float = 1.7,
) -> dict:
    """Challenge a betting thesis without changing the underlying prediction."""
    clean_reasoning = " ".join(str(reasoning or "").lower().split())
    if len(clean_reasoning) < 12:
        raise ValueError("Explain your reasoning in at least one complete sentence.")

    conflicts = build_conflict_scores(
        selected_team,
        opponent,
        selected_profile,
        opponent_profile,
        selected_is_home=selected_is_home,
        home_field_points=home_field_points,
    )

    recognized_topics: list[str] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if _contains_any(clean_reasoning, keywords):
            recognized_topics.append(topic)

    # A protection/pressure statement often describes both sides of the trench battle.
    if "protection_vs_pass_rush" in recognized_topics and "pass_rush_vs_protection" not in recognized_topics:
        if any(word in clean_reasoning for word in ("their pass rush", "our pass rush", "get pressure", "pressure them")):
            recognized_topics.append("pass_rush_vs_protection")

    points: list[ReasoningPoint] = []
    for topic in recognized_topics:
        conflict = conflicts[topic]
        status = _status_for_edge(float(conflict["edge"]))
        points.append(
            ReasoningPoint(
                topic=topic,
                label=str(conflict["label"]),
                status=status,
                edge=round(float(conflict["edge"]), 1),
                evidence=_evidence_sentence(selected_team, opponent, conflict, status),
            )
        )

    assumptions = []
    for label, keywords in UNVERIFIED_KEYWORDS.items():
        if _contains_any(clean_reasoning, keywords):
            assumptions.append(
                f"Your {label} point needs a current external data check; the present Macabets team profile does not verify it automatically."
            )

    mentioned = set(recognized_topics)
    missing_candidates = [
        (topic, data)
        for topic, data in conflicts.items()
        if topic not in mentioned and topic != "home_field"
    ]
    missing_candidates.sort(key=lambda item: abs(float(item[1]["edge"])), reverse=True)
    missing_factors = []
    for _, conflict in missing_candidates[:2]:
        edge = float(conflict["edge"])
        direction = selected_team if edge > 0 else opponent
        missing_factors.append(
            f"You did not address {conflict['label'].lower()}, where the current ratings favor "
            f"{direction} by {abs(edge):.1f} points."
        )

    if points:
        support_score = sum(max(-6.0, min(6.0, point.edge)) for point in points) / len(points)
    else:
        support_score = 0.0

    model_alignment = selected_team == projected_winner
    confidence_adjustment = round(max(-5.0, min(5.0, support_score * 0.65)), 1)
    if not model_alignment and confidence_adjustment > 0:
        confidence_adjustment = round(confidence_adjustment * 0.5, 1)
    if assumptions:
        confidence_adjustment = round(confidence_adjustment - min(2.0, 0.5 * len(assumptions)), 1)

    confirmed = sum(point.status in {"Confirmed", "Mostly confirmed"} for point in points)
    pushback = sum(point.status == "Pushback" for point in points)

    if not points:
        verdict = "Reasonable, but not testable yet"
        bottom_line = (
            "Macabets could not connect your main claim to one of its current matchup inputs. "
            "The thesis may still be valid, but it needs a more specific football mechanism or additional data."
        )
    elif pushback > confirmed:
        verdict = "Macabets disagrees"
        bottom_line = (
            f"Your case for {selected_team} leans on battles that currently favor {opponent}. "
            "The bet needs a stronger counterargument before Macabets would confirm it."
        )
    elif pushback and confirmed:
        verdict = "Reasonable, but incomplete"
        bottom_line = (
            f"Macabets supports part of your {selected_team} thesis but sees a meaningful counter-matchup. "
            "Your confidence should depend on which conflict controls the game script."
        )
    elif confirmed:
        verdict = "Mostly confirmed" if assumptions or any(point.status == "Mixed" for point in points) else "Confirmed"
        bottom_line = (
            f"Your reasoning for {selected_team} is consistent with the current matchup ratings. "
            "It strengthens the case, but does not override price, injuries, or late information."
        )
    else:
        verdict = "Reasonable, but incomplete"
        bottom_line = (
            "The identified matchup is close enough that it should not be the sole reason for the bet."
        )

    return {
        "selected_team": selected_team,
        "opponent": opponent,
        "verdict": verdict,
        "model_alignment": model_alignment,
        "points": [asdict(point) for point in points],
        "assumptions": assumptions,
        "missing_factors": missing_factors,
        "confidence_adjustment": confidence_adjustment,
        "bottom_line": bottom_line,
        "notice": (
            "This challenges the logic of your thesis. It does not change Macabets' win probability or fair line."
        ),
    }
