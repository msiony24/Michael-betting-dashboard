"""Opponent-specific NFL matchup reasoning for Macabets.

The brain layer does not replace the prediction model. It converts the team
components already available to Macabets into direct offense-versus-defense
conflicts and simple football chain reactions. It deliberately avoids claiming
scheme, injury, or player-level facts that are not present in the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class MatchupConflict:
    name: str
    offense_team: str
    defense_team: str
    edge_team: str | None
    edge: float
    strength: str
    question: str
    explanation: str
    consequence: str


@dataclass(frozen=True)
class ChainReaction:
    team: str
    trigger: str
    steps: list[str]
    summary: str


def _score(components: Mapping[str, float], key: str, fallback: float = 67.5) -> float:
    try:
        return float(components.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _strength(edge: float) -> str:
    magnitude = abs(edge)
    if magnitude >= 8.0:
        return "Major"
    if magnitude >= 5.0:
        return "Clear"
    if magnitude >= 2.5:
        return "Lean"
    return "Even"


def _winner(offense_team: str, defense_team: str, edge: float) -> str | None:
    if abs(edge) < 2.5:
        return None
    return offense_team if edge > 0 else defense_team


def _conflict(
    *,
    name: str,
    offense_team: str,
    defense_team: str,
    offense_value: float,
    defense_value: float,
    question: str,
    offense_win_text: str,
    defense_win_text: str,
    even_text: str,
    offense_consequence: str,
    defense_consequence: str,
) -> MatchupConflict:
    edge = round(offense_value - defense_value, 2)
    leader = _winner(offense_team, defense_team, edge)
    strength = _strength(edge)

    if leader == offense_team:
        explanation = offense_win_text
        consequence = offense_consequence
    elif leader == defense_team:
        explanation = defense_win_text
        consequence = defense_consequence
    else:
        explanation = even_text
        consequence = "Neither side has enough separation here to drive the prediction by itself."

    return MatchupConflict(
        name=name,
        offense_team=offense_team,
        defense_team=defense_team,
        edge_team=leader,
        edge=round(abs(edge), 2),
        strength=strength,
        question=question,
        explanation=explanation,
        consequence=consequence,
    )


def _team_conflicts(
    offense_team: str,
    defense_team: str,
    offense: Mapping[str, float],
    defense: Mapping[str, float],
) -> list[MatchupConflict]:
    quarterback = _score(offense, "quarterback")
    offense_line = _score(offense, "offensive_line")
    offense_grade = _score(offense, "offense")
    skill = _score(offense, "skill_positions")

    defense_grade = _score(defense, "defense")
    defensive_line = _score(defense, "defensive_line")
    secondary = _score(defense, "secondary")

    passing_structure = quarterback * 0.58 + skill * 0.22 + offense_grade * 0.20
    coverage_structure = secondary * 0.70 + defense_grade * 0.30

    rushing_structure = offense_line * 0.46 + skill * 0.24 + offense_grade * 0.30
    run_front = defensive_line * 0.58 + defense_grade * 0.42

    return [
        _conflict(
            name="Quarterback vs coverage",
            offense_team=offense_team,
            defense_team=defense_team,
            offense_value=passing_structure,
            defense_value=coverage_structure,
            question=f"Can {offense_team}'s passing structure consistently solve {defense_team}'s coverage profile?",
            offense_win_text=(
                f"{offense_team}'s quarterback-led passing profile is strong enough to stress "
                f"{defense_team}'s secondary rather than simply relying on an overall team edge."
            ),
            defense_win_text=(
                f"{defense_team}'s coverage profile is better positioned to make {offense_team} "
                "win through longer drives instead of easy passing efficiency."
            ),
            even_text=(
                f"{offense_team}'s passing profile and {defense_team}'s coverage profile are closely matched."
            ),
            offense_consequence=(
                "If the protection holds, the offense should have enough answers to sustain the passing game."
            ),
            defense_consequence=(
                "The offense may be pushed away from its preferred passing rhythm and into lower-efficiency possessions."
            ),
        ),
        _conflict(
            name="Pass protection vs defensive front",
            offense_team=offense_team,
            defense_team=defense_team,
            offense_value=offense_line,
            defense_value=defensive_line,
            question=f"Can {offense_team} protect long enough for its passing advantage to matter?",
            offense_win_text=(
                f"{offense_team}'s offensive-line profile projects to hold up against {defense_team}'s front."
            ),
            defense_win_text=(
                f"{defense_team}'s defensive front is positioned to disrupt {offense_team} before its passing game fully develops."
            ),
            even_text=(
                f"The protection-versus-front battle between {offense_team} and {defense_team} is approximately even."
            ),
            offense_consequence=(
                "Cleaner pockets increase the value of the quarterback and receiving matchup at the same time."
            ),
            defense_consequence=(
                "Pressure can erase advantages elsewhere by shortening the quarterback's decision window."
            ),
        ),
        _conflict(
            name="Receiving weapons vs secondary",
            offense_team=offense_team,
            defense_team=defense_team,
            offense_value=skill * 0.72 + offense_grade * 0.28,
            defense_value=secondary * 0.78 + defense_grade * 0.22,
            question=f"Do {offense_team}'s weapons create more stress than {defense_team}'s secondary can absorb?",
            offense_win_text=(
                f"{offense_team}'s receiving and skill-position profile creates a direct problem for {defense_team}'s secondary."
            ),
            defense_win_text=(
                f"{defense_team}'s secondary is strong enough to reduce the value of {offense_team}'s weapons."
            ),
            even_text=(
                f"{offense_team}'s weapons and {defense_team}'s secondary grade out as a balanced conflict."
            ),
            offense_consequence=(
                "Winning individual receiving matchups can create explosive-play and third-down opportunities."
            ),
            defense_consequence=(
                "The quarterback may need to hold the ball longer or rely more heavily on the run game."
            ),
        ),
        _conflict(
            name="Run structure vs defensive front",
            offense_team=offense_team,
            defense_team=defense_team,
            offense_value=rushing_structure,
            defense_value=run_front,
            question=f"Can {offense_team} stay efficient enough on the ground to control down and distance?",
            offense_win_text=(
                f"{offense_team}'s line, supporting skill talent and overall offense combine for a favorable rushing structure "
                f"against {defense_team}'s front."
            ),
            defense_win_text=(
                f"{defense_team}'s front is equipped to prevent {offense_team} from using the run game as a stabilizer."
            ),
            even_text=(
                f"The projected run-game conflict between {offense_team} and {defense_team} is close to neutral."
            ),
            offense_consequence=(
                "Efficient early-down rushing keeps play action, manageable third downs and the full offense available."
            ),
            defense_consequence=(
                "If early-down runs fail, the offense becomes more predictable and the pass rush gains leverage."
            ),
        ),
    ]


def _chain_reactions(conflicts: list[MatchupConflict]) -> list[ChainReaction]:
    by_team: dict[str, list[MatchupConflict]] = {}
    for conflict in conflicts:
        if conflict.edge_team and conflict.strength in {"Clear", "Major"}:
            by_team.setdefault(conflict.edge_team, []).append(conflict)

    chains: list[ChainReaction] = []
    for team, won in by_team.items():
        names = {item.name for item in won}

        if {
            "Pass protection vs defensive front",
            "Quarterback vs coverage",
        }.issubset(names):
            chains.append(
                ChainReaction(
                    team=team,
                    trigger="Protection and passing conflicts both favor the same offense",
                    steps=[
                        "Protection holds up",
                        "Quarterback keeps a fuller decision window",
                        "Coverage has to defend the entire route progression",
                        "Passing efficiency and sustained-drive probability rise together",
                    ],
                    summary=(
                        f"{team}'s protection edge reinforces its quarterback-versus-coverage edge, "
                        "so these are not independent advantages."
                    ),
                )
            )

        if {
            "Run structure vs defensive front",
            "Pass protection vs defensive front",
        }.issubset(names):
            chains.append(
                ChainReaction(
                    team=team,
                    trigger="The offensive line projects to control both run and pass situations",
                    steps=[
                        "Early downs remain manageable",
                        "The defense cannot attack obvious passing situations as often",
                        "Play action and the full playbook remain credible",
                        "The offense gains control over game script",
                    ],
                    summary=(
                        f"{team}'s trench advantage can influence both efficiency and play-calling freedom."
                    ),
                )
            )

        if {
            "Receiving weapons vs secondary",
            "Quarterback vs coverage",
        }.issubset(names):
            chains.append(
                ChainReaction(
                    team=team,
                    trigger="Quarterback and receiver conflicts both favor the passing offense",
                    steps=[
                        "Receivers create viable targets",
                        "The quarterback does not need perfect throws on every down",
                        "Third-down and explosive-play chances improve",
                        "The defense is forced to allocate extra help",
                    ],
                    summary=(
                        f"{team}'s passing edge is supported by both the quarterback and the receiving matchup."
                    ),
                )
            )

    return chains[:4]


def build_matchup_brain(
    *,
    away_team: str,
    home_team: str,
    away_components: Mapping[str, float],
    home_components: Mapping[str, float],
) -> dict:
    conflicts = [
        *_team_conflicts(away_team, home_team, away_components, home_components),
        *_team_conflicts(home_team, away_team, home_components, away_components),
    ]

    coaching_gap = round(_score(home_components, "coaching") - _score(away_components, "coaching"), 2)
    coaching_leader = _winner(home_team, away_team, coaching_gap)
    if coaching_leader:
        conflicts.append(
            MatchupConflict(
                name="Coaching and adjustment leverage",
                offense_team=home_team,
                defense_team=away_team,
                edge_team=coaching_leader,
                edge=abs(coaching_gap),
                strength=_strength(coaching_gap),
                question="Which staff is better positioned to adjust when the first plan stops working?",
                explanation=(
                    f"{coaching_leader} owns the clearer coaching and adjustment profile in the current ratings."
                ),
                consequence=(
                    "This matters most in close games where early matchup advantages are countered and recreated."
                ),
            )
        )

    chains = _chain_reactions(conflicts)

    team_scores = {away_team: 0.0, home_team: 0.0}
    for conflict in conflicts:
        if conflict.edge_team in team_scores:
            multiplier = {"Lean": 1.0, "Clear": 1.35, "Major": 1.65}.get(conflict.strength, 0.0)
            team_scores[conflict.edge_team] += conflict.edge * multiplier

    score_gap = round(team_scores[home_team] - team_scores[away_team], 2)
    leader = home_team if score_gap > 0 else away_team if score_gap < 0 else None

    ranked = sorted(
        conflicts,
        key=lambda item: (
            {"Major": 3, "Clear": 2, "Lean": 1, "Even": 0}[item.strength],
            item.edge,
        ),
        reverse=True,
    )

    if leader:
        top = [item for item in ranked if item.edge_team == leader][:2]
        if top:
            summary = (
                f"{leader} owns the stronger opponent-specific conflict profile, led by "
                + " and ".join(item.name.lower() for item in top)
                + "."
            )
        else:
            summary = f"{leader} holds a narrow matchup-conflict edge."
    else:
        summary = "The matchup conflicts are balanced enough that no single football interaction clearly controls the game."

    return {
        "version": "NFL Brain v0.1",
        "matchup_score_home": score_gap,
        "matchup_leader": leader or "Even",
        "summary": summary,
        "conflicts": [asdict(item) for item in ranked],
        "chain_reactions": [asdict(item) for item in chains],
        "limitations": [
            "This layer uses current Macabets team components, not verified play-calling or coverage-frequency data.",
            "Injuries, confirmed starters and current scheme tendencies must still be checked separately.",
        ],
    }
