"""Explanation-only tennis matchup intelligence for Macabets.

The purpose of this engine is NOT to predict the winner.
Its purpose is to explain how two tennis players interact.

Prediction happens elsewhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import re
import unicodedata

from .player_traits import PlayerTraitsDatabase
from .tournament_profiles import resolve_tournament_profile


SEVERITY_ORDER = {
    "Neutral": 0,
    "Minor": 1,
    "Moderate": 2,
    "Major": 3,
    "Match-Defining": 4,
}

RALLY_ORDER = {
    "Short": 1,
    "Short-Medium": 2,
    "Medium": 3,
    "Medium-Long": 4,
    "Long": 5,
}


@dataclass(frozen=True)
class MatchupEdge:
    category: str
    winner: str | None
    strength: str
    explanation: str
    score_a: float
    score_b: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _skill(profile: dict[str, Any], name: str) -> int:
    return int(profile.get("skills", {}).get(name, 3))


def _traits(profile: dict[str, Any]) -> set[str]:
    return set(profile.get("signature_traits", []))


def _severity(diff: float, *, defining_at: float = 2.5) -> str:
    gap = abs(diff)
    if gap < 0.75:
        return "Neutral"
    if gap < 1.25:
        return "Minor"
    if gap < 2.0:
        return "Moderate"
    if gap < defining_at:
        return "Major"
    return "Match-Defining"


def _winner(name_a: str, name_b: str, diff: float) -> str | None:
    if abs(diff) < 0.75:
        return None
    return name_a if diff > 0 else name_b


def _confidence(diff: float, profile_confidence: str = "medium") -> float:
    base = min(0.98, 0.55 + abs(diff) * 0.10)
    if profile_confidence == "high":
        base += 0.05
    elif profile_confidence in {"fallback", "low"}:
        base -= 0.10
    return round(max(0.35, min(base, 0.98)), 2)


def _name_tokens(name: str) -> list[str]:
    """Return accent-free alphanumeric tokens for reliable player-name matching."""
    normalized = unicodedata.normalize("NFKD", str(name or ""))
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", normalized.casefold())


def _abbreviated_name_candidates(
    player_name: str,
    canonical_names: Iterable[str],
) -> list[str]:
    """Return canonical matches for common ATP dataset abbreviations.

    Supported examples:
    - ``Sinner J.`` -> ``Jannik Sinner``
    - ``Zverev A.`` -> ``Alexander Zverev``
    - ``de Minaur A.`` -> ``Alex de Minaur``
    - ``J. Sinner`` -> ``Jannik Sinner``

    A result is accepted only when it is unique. This avoids silently assigning
    the wrong curated profile when two players share a surname and initial.
    """
    input_tokens = _name_tokens(player_name)
    if len(input_tokens) < 2:
        return []

    matches: list[str] = []
    for canonical_name in canonical_names:
        canonical_tokens = _name_tokens(canonical_name)
        if len(canonical_tokens) < 2:
            continue

        first_initial = canonical_tokens[0][0]
        surname_tokens = canonical_tokens[1:]

        surname_first_match = (
            input_tokens[-1] == first_initial
            and input_tokens[:-1] == surname_tokens
        )
        initial_first_match = (
            input_tokens[0] == first_initial
            and input_tokens[1:] == surname_tokens
        )

        if surname_first_match or initial_first_match:
            matches.append(canonical_name)

    return matches


def resolve_player_profile(
    database: PlayerTraitsDatabase,
    player_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve an app/API player name to a curated database profile.

    Resolution order:
    1. Existing canonical-name or alias lookup.
    2. Unique surname-plus-initial abbreviation lookup.
    3. Safe unresolved result; no profile is invented.
    """
    direct_profile = database.get(player_name)
    if direct_profile is not None:
        return direct_profile, {
            "input": player_name,
            "resolved": direct_profile["name"],
            "method": "exact_or_alias",
        }

    candidates = _abbreviated_name_candidates(
        player_name,
        database.all_players(),
    )
    if len(candidates) == 1:
        resolved_profile = database.get(candidates[0])
        return resolved_profile, {
            "input": player_name,
            "resolved": candidates[0],
            "method": "surname_initial",
        }

    return None, {
        "input": player_name,
        "resolved": None,
        "method": "ambiguous" if len(candidates) > 1 else "unresolved",
        "candidates": candidates,
    }


def evaluate_serve(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    court: dict[str, Any],
) -> MatchupEdge:
    name_a, name_b = profile_a["name"], profile_b["name"]
    traits_a, traits_b = _traits(profile_a), _traits(profile_b)

    speed = int(court.get("speed", 3))
    serve_multiplier = 1.15 if speed >= 4 else 0.90 if speed <= 2 else 1.0

    score_a = _skill(profile_a, "serve") * serve_multiplier
    score_b = _skill(profile_b, "serve") * serve_multiplier

    if "Big Server" in traits_a:
        score_a += 0.35
    if "Big Server" in traits_b:
        score_b += 0.35
    if "Serve Plus One" in traits_a:
        score_a += 0.20
    if "Serve Plus One" in traits_b:
        score_b += 0.20
    if "Varied Serving Patterns" in traits_a:
        score_a += 0.15
    if "Varied Serving Patterns" in traits_b:
        score_b += 0.15

    diff = score_a - score_b
    winner = _winner(name_a, name_b, diff)
    strength = _severity(diff)

    if winner is None:
        explanation = "Neither player owns a clear independent serve advantage in these conditions."
    else:
        loser = name_b if winner == name_a else name_a
        condition = "faster conditions" if speed >= 4 else "slower conditions" if speed <= 2 else "neutral-speed conditions"
        explanation = (
            f"{winner} projects to create more reliable first-strike pressure on serve than "
            f"{loser}, with {condition} shaping the size of that edge."
        )

    return MatchupEdge(
        category="Serve",
        winner=winner,
        strength=strength,
        explanation=explanation,
        score_a=round(score_a, 2),
        score_b=round(score_b, 2),
        confidence=_confidence(diff, str(court.get("confidence", "medium"))),
    )


def evaluate_return(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
) -> MatchupEdge:
    name_a, name_b = profile_a["name"], profile_b["name"]
    traits_a, traits_b = _traits(profile_a), _traits(profile_b)

    score_a = _skill(profile_a, "return") + (0.35 if "Elite Returner" in traits_a else 0)
    score_b = _skill(profile_b, "return") + (0.35 if "Elite Returner" in traits_b else 0)

    # Return quality is most valuable when it can attack an opponent whose serve
    # is not elite. This is still one return path, not a second counted edge.
    score_a += max(0, 4 - _skill(profile_b, "serve")) * 0.20
    score_b += max(0, 4 - _skill(profile_a, "serve")) * 0.20

    diff = score_a - score_b
    winner = _winner(name_a, name_b, diff)
    strength = _severity(diff)

    if winner is None:
        explanation = "The return matchup is balanced enough that neither player clearly owns this phase."
    else:
        loser = name_b if winner == name_a else name_a
        explanation = (
            f"{winner} is better equipped to neutralize first serves and apply second-serve pressure "
            f"against {loser}."
        )

    return MatchupEdge(
        category="Return",
        winner=winner,
        strength=strength,
        explanation=explanation,
        score_a=round(score_a, 2),
        score_b=round(score_b, 2),
        confidence=_confidence(diff),
    )


def evaluate_baseline(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    court: dict[str, Any],
) -> MatchupEdge:
    name_a, name_b = profile_a["name"], profile_b["name"]
    traits_a, traits_b = _traits(profile_a), _traits(profile_b)
    speed = int(court.get("speed", 3))
    bounce = str(court.get("bounce", "medium")).casefold()

    score_a = (
        _skill(profile_a, "forehand") * 0.42
        + _skill(profile_a, "backhand") * 0.38
        + _skill(profile_a, "movement") * 0.20
    )
    score_b = (
        _skill(profile_b, "forehand") * 0.42
        + _skill(profile_b, "backhand") * 0.38
        + _skill(profile_b, "movement") * 0.20
    )

    if "Baseline Pressure" in traits_a:
        score_a += 0.20
    if "Baseline Pressure" in traits_b:
        score_b += 0.20
    if "Excellent Depth" in traits_a:
        score_a += 0.15
    if "Excellent Depth" in traits_b:
        score_b += 0.15
    if "Early Ball Striker" in traits_a and speed >= 4:
        score_a += 0.20
    if "Early Ball Striker" in traits_b and speed >= 4:
        score_b += 0.20
    if "Heavy Topspin" in traits_a and "high" in bounce:
        score_a += 0.25
    if "Heavy Topspin" in traits_b and "high" in bounce:
        score_b += 0.25
    if "Flat Ball Striking" in traits_a and speed >= 4:
        score_a += 0.15
    if "Flat Ball Striking" in traits_b and speed >= 4:
        score_b += 0.15

    rally_a = RALLY_ORDER.get(str(profile_a.get("preferred_rally", "Medium")), 3)
    rally_b = RALLY_ORDER.get(str(profile_b.get("preferred_rally", "Medium")), 3)
    expected_rally = 4 if speed <= 2 else 2 if speed >= 4 else 3
    score_a += max(0, 2 - abs(rally_a - expected_rally)) * 0.10
    score_b += max(0, 2 - abs(rally_b - expected_rally)) * 0.10

    diff = score_a - score_b
    winner = _winner(name_a, name_b, diff)
    strength = _severity(diff)

    if winner is None:
        explanation = "The baseline exchange is close, with no clear evidence that one player can consistently impose a superior pattern."
    else:
        loser = name_b if winner == name_a else name_a
        explanation = (
            f"{winner} has the stronger combination of groundstroke quality, depth and rally fit "
            f"for the expected conditions against {loser}."
        )

    return MatchupEdge(
        category="Baseline",
        winner=winner,
        strength=strength,
        explanation=explanation,
        score_a=round(score_a, 2),
        score_b=round(score_b, 2),
        confidence=_confidence(diff, str(court.get("confidence", "medium"))),
    )


def evaluate_movement(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    court: dict[str, Any],
) -> MatchupEdge:
    name_a, name_b = profile_a["name"], profile_b["name"]
    traits_a, traits_b = _traits(profile_a), _traits(profile_b)
    speed = int(court.get("speed", 3))

    multiplier = 1.15 if speed <= 2 else 0.95 if speed >= 4 else 1.0
    score_a = _skill(profile_a, "movement") * multiplier
    score_b = _skill(profile_b, "movement") * multiplier

    if "Relentless Defender" in traits_a or "Defensive Retrieval" in traits_a:
        score_a += 0.20
    if "Relentless Defender" in traits_b or "Defensive Retrieval" in traits_b:
        score_b += 0.20

    diff = score_a - score_b
    winner = _winner(name_a, name_b, diff)
    strength = _severity(diff)

    if winner is None:
        explanation = "Movement is unlikely to create a decisive independent edge."
    else:
        loser = name_b if winner == name_a else name_a
        explanation = (
            f"{winner} is more likely to extend points, recover court position and force {loser} "
            f"to hit additional quality shots."
        )

    return MatchupEdge(
        category="Movement",
        winner=winner,
        strength=strength,
        explanation=explanation,
        score_a=round(score_a, 2),
        score_b=round(score_b, 2),
        confidence=_confidence(diff, str(court.get("confidence", "medium"))),
    )


def evaluate_variety(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
) -> MatchupEdge:
    name_a, name_b = profile_a["name"], profile_b["name"]
    traits_a, traits_b = _traits(profile_a), _traits(profile_b)

    score_a = _skill(profile_a, "variety") * 0.65 + _skill(profile_a, "net_play") * 0.35
    score_b = _skill(profile_b, "variety") * 0.65 + _skill(profile_b, "net_play") * 0.35

    for trait in ("Drop Shot Threat", "Slice Variety", "Transition Game", "Net Rushing"):
        if trait in traits_a:
            score_a += 0.10
        if trait in traits_b:
            score_b += 0.10

    diff = score_a - score_b
    winner = _winner(name_a, name_b, diff)
    strength = _severity(diff)

    if winner is None:
        explanation = "Neither player has a clear independent advantage in changing patterns or finishing away from the baseline."
    else:
        loser = name_b if winner == name_a else name_a
        explanation = (
            f"{winner} has more credible ways to change pace, court position and finishing patterns "
            f"than {loser}."
        )

    return MatchupEdge(
        category="Variety",
        winner=winner,
        strength=strength,
        explanation=explanation,
        score_a=round(score_a, 2),
        score_b=round(score_b, 2),
        confidence=_confidence(diff),
    )


def evaluate_surface(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    court: dict[str, Any],
) -> MatchupEdge:
    name_a, name_b = profile_a["name"], profile_b["name"]
    traits_a, traits_b = _traits(profile_a), _traits(profile_b)
    speed = int(court.get("speed", 3))
    bounce = str(court.get("bounce", "medium")).casefold()
    surface = str(court.get("surface", "Hard")).casefold()

    score_a = 0.0
    score_b = 0.0

    def apply(profile: dict[str, Any], traits: set[str]) -> float:
        score = 0.0
        if speed >= 4:
            score += (_skill(profile, "serve") - 3) * 0.35
            if "Early Ball Striker" in traits or "Flat Ball Striking" in traits:
                score += 0.30
        elif speed <= 2:
            score += (_skill(profile, "movement") - 3) * 0.30
            score += (_skill(profile, "return") - 3) * 0.20
            if "Relentless Defender" in traits or "Point Construction" in traits:
                score += 0.25

        if "high" in bounce and "Heavy Topspin" in traits:
            score += 0.35
        if "low" in bounce and ("Slice Variety" in traits or "Early Ball Striker" in traits):
            score += 0.25
        if surface == "grass" and "Net Rushing" in traits:
            score += 0.20
        return score

    score_a = apply(profile_a, traits_a)
    score_b = apply(profile_b, traits_b)

    diff = score_a - score_b
    winner = _winner(name_a, name_b, diff)
    strength = _severity(diff, defining_at=2.0)

    label = court.get("matched_name") or court.get("requested_name") or "the selected event"
    if winner is None:
        explanation = f"{label} does not create a clear independent surface advantage for either player."
    else:
        loser = name_b if winner == name_a else name_a
        explanation = (
            f"{label}'s {court.get('speed_label', 'medium')} pace and {court.get('bounce', 'medium')} "
            f"bounce amplify more of {winner}'s established strengths than {loser}'s."
        )

    return MatchupEdge(
        category="Surface",
        winner=winner,
        strength=strength,
        explanation=explanation,
        score_a=round(score_a, 2),
        score_b=round(score_b, 2),
        confidence=_confidence(diff, str(court.get("confidence", "medium"))),
    )


def _independent_paths(edges: Iterable[MatchupEdge], player_name: str) -> list[str]:
    paths: list[str] = []
    for edge in edges:
        if edge.winner != player_name:
            continue
        if SEVERITY_ORDER.get(edge.strength, 0) < SEVERITY_ORDER["Moderate"]:
            continue
        paths.append(edge.category)
    return paths


def analyze_matchup(
    player_a: str,
    player_b: str,
    tournament: str,
    surface: str,
    environment: str = "Outdoor",
    *,
    database: PlayerTraitsDatabase | None = None,
) -> dict[str, Any]:
    """Return a deterministic, explanation-only scouting report.

    Unknown players are reported explicitly rather than assigned invented traits.
    """
    db = database or PlayerTraitsDatabase()
    profile_a, resolution_a = resolve_player_profile(db, player_a)
    profile_b, resolution_b = resolve_player_profile(db, player_b)
    missing = [
        name
        for name, profile in ((player_a, profile_a), (player_b, profile_b))
        if profile is None
    ]
    if missing:
        return {
            "status": "insufficient_profile_data",
            "missing_players": missing,
            "player_a": player_a,
            "player_b": player_b,
            "tournament": tournament,
            "name_resolution": {
                "player_a": resolution_a,
                "player_b": resolution_b,
            },
            "explanation_only": True,
        }

    court = resolve_tournament_profile(tournament, surface, environment)

    edges = [
        evaluate_serve(profile_a, profile_b, court),
        evaluate_return(profile_a, profile_b),
        evaluate_baseline(profile_a, profile_b, court),
        evaluate_movement(profile_a, profile_b, court),
        evaluate_variety(profile_a, profile_b),
        evaluate_surface(profile_a, profile_b, court),
    ]

    name_a = profile_a["name"]
    name_b = profile_b["name"]
    paths_a = _independent_paths(edges, name_a)
    paths_b = _independent_paths(edges, name_b)

    defining = [
        edge for edge in edges if edge.strength in {"Major", "Match-Defining"}
    ]
    defining.sort(
        key=lambda edge: (
            SEVERITY_ORDER[edge.strength],
            abs(edge.score_a - edge.score_b),
            edge.category,
        ),
        reverse=True,
    )

    path_gap = len(paths_a) - len(paths_b)
    if abs(path_gap) <= 1 and not any(
        edge.strength == "Match-Defining" for edge in defining
    ):
        tactical_read = (
            "Too close to call tactically. The matchup is more likely to be decided "
            "by execution, current form and match-day conditions than by a clear style edge."
        )
        tactical_edge = None
    else:
        tactical_edge = name_a if path_gap > 0 else name_b
        tactical_read = (
            f"{tactical_edge} has more independent paths to victory in this matchup "
            f"({len(paths_a)} to {len(paths_b)})."
        )

    return {
        "status": "ok",
        "explanation_only": True,
        "player_a": name_a,
        "player_b": name_b,
        "name_resolution": {
            "player_a": resolution_a,
            "player_b": resolution_b,
        },
        "court_profile": court,
        "edges": [edge.to_dict() for edge in edges],
        "paths_to_victory": {
            name_a: paths_a,
            name_b: paths_b,
        },
        "path_counts": {
            name_a: len(paths_a),
            name_b: len(paths_b),
        },
        "tactical_edge": tactical_edge,
        "tactical_read": tactical_read,
        "decisive_factors": [edge.to_dict() for edge in defining[:3]],
    }
