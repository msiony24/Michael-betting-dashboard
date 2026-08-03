"""Opponent-specific NFL matchup reasoning for Macabets.

The brain layer does not replace the prediction model. It converts the team
components already available to Macabets into direct offense-versus-defense
conflicts and simple football chain reactions. It deliberately avoids claiming
scheme, injury, or player-level facts that are not present in the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from engine.nfl_team_schema import NFLTeamProfile, profile_from_legacy_components


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
class ExploitOpportunity:
    team: str
    target_team: str
    name: str
    score: float
    level: str
    weapon: str
    weakness: str
    explanation: str
    consequence: str


@dataclass(frozen=True)
class WinCondition:
    team: str
    title: str
    requirements: list[str]
    chain: list[str]
    realism_score: float
    explanation: str


@dataclass(frozen=True)
class FailureCondition:
    team: str
    threat_team: str
    title: str
    trigger: str
    consequence: str
    threat_score: float


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


def _exploit_level(score: float) -> str:
    if score >= 9.0:
        return "Severe"
    if score >= 6.0:
        return "High"
    if score >= 3.0:
        return "Moderate"
    return "Limited"


def _make_exploit(
    *,
    team: str,
    target_team: str,
    name: str,
    weapon_score: float,
    resistance_score: float,
    weapon: str,
    weakness: str,
    positive_text: str,
    consequence: str,
) -> ExploitOpportunity:
    score = round(max(0.0, weapon_score - resistance_score), 2)
    level = _exploit_level(score)
    if score < 3.0:
        explanation = (
            f"{team} does not currently own enough separation in {weapon.lower()} "
            f"to label {target_team}'s {weakness.lower()} a dependable exploit."
        )
    else:
        explanation = positive_text
    return ExploitOpportunity(
        team=team,
        target_team=target_team,
        name=name,
        score=score,
        level=level,
        weapon=weapon,
        weakness=weakness,
        explanation=explanation,
        consequence=consequence,
    )


def _team_exploits(profile: NFLTeamProfile, opponent: NFLTeamProfile) -> list[ExploitOpportunity]:
    offense = profile.offense
    defense = opponent.defense

    passing_weapon = (
        offense.quarterback * 0.42
        + offense.pass_protection * 0.25
        + offense.receiving_weapons * 0.33
    )
    passing_resistance = (
        defense.pass_rush * 0.30
        + defense.cornerbacks * 0.42
        + defense.safeties * 0.28
    )

    pressure_survival = offense.quarterback * 0.48 + offense.pass_protection * 0.52
    pressure_threat = defense.pass_rush * 0.72 + defense.overall * 0.28

    rushing_weapon = offense.run_blocking * 0.58 + offense.running_backs * 0.42
    rushing_resistance = defense.run_defense * 0.78 + defense.overall * 0.22

    middle_weapon = offense.quarterback * 0.34 + offense.receiving_weapons * 0.48 + offense.overall * 0.18
    middle_resistance = defense.linebacker_coverage * 0.56 + defense.safeties * 0.44

    return [
        _make_exploit(
            team=profile.team,
            target_team=opponent.team,
            name="Passing structure against coverage",
            weapon_score=passing_weapon,
            resistance_score=passing_resistance,
            weapon="Quarterback, protection and receiving structure",
            weakness="Pass rush and coverage resistance",
            positive_text=(
                f"{profile.team}'s quarterback, protection and receiving profile combine to create "
                f"a direct passing-game problem for {opponent.team}."
            ),
            consequence="If this advantage holds, the offense can sustain drives without needing unusual turnover luck.",
        ),
        _make_exploit(
            team=profile.team,
            target_team=opponent.team,
            name="Quarterback survival against pressure",
            weapon_score=pressure_survival,
            resistance_score=pressure_threat,
            weapon="Quarterback capability and pass protection",
            weakness="Defensive pressure profile",
            positive_text=(
                f"{profile.team} is positioned to keep its quarterback functional against "
                f"{opponent.team}'s defensive-front pressure."
            ),
            consequence="A viable quarterback under pressure keeps the team's full winning path available.",
        ),
        _make_exploit(
            team=profile.team,
            target_team=opponent.team,
            name="Run-game leverage",
            weapon_score=rushing_weapon,
            resistance_score=rushing_resistance,
            weapon="Run blocking and backfield quality",
            weakness="Run-defense resistance",
            positive_text=(
                f"{profile.team}'s run structure can attack {opponent.team}'s front strongly enough "
                "to create manageable down-and-distance situations."
            ),
            consequence="Successful early-down runs protect the quarterback and preserve play-action credibility.",
        ),
        _make_exploit(
            team=profile.team,
            target_team=opponent.team,
            name="Middle-field access",
            weapon_score=middle_weapon,
            resistance_score=middle_resistance,
            weapon="Quarterback and receiving weapons",
            weakness="Linebacker and safety coverage",
            positive_text=(
                f"{profile.team}'s passing personnel projects to stress the middle layers of "
                f"{opponent.team}'s coverage profile."
            ),
            consequence="Middle-field access can improve third-down conversion and reduce reliance on low-percentage deep throws.",
        ),
    ]


def _qb_gate(profile: NFLTeamProfile, opponent: NFLTeamProfile) -> dict:
    capability = (
        profile.offense.quarterback * 0.55
        + profile.offense.pass_protection * 0.20
        + profile.offense.receiving_weapons * 0.15
        + profile.offense.overall * 0.10
    )
    challenge = (
        opponent.defense.pass_rush * 0.35
        + opponent.defense.cornerbacks * 0.30
        + opponent.defense.safeties * 0.20
        + opponent.defense.overall * 0.15
    )
    margin = round(capability - challenge, 2)
    if margin >= 4.0:
        verdict = "Pass"
        explanation = f"{profile.team}'s quarterback environment is capable of winning this specific matchup."
    elif margin <= -5.0:
        verdict = "Fail"
        explanation = f"{profile.team}'s quarterback path is currently overmatched by {opponent.team}'s pressure-and-coverage profile."
    else:
        verdict = "Conditional"
        explanation = f"{profile.team}'s quarterback can win, but the supporting matchup must hold together."
    return {
        "team": profile.team,
        "opponent": opponent.team,
        "verdict": verdict,
        "margin": margin,
        "explanation": explanation,
    }


def _win_condition(
    team: NFLTeamProfile,
    opponent: NFLTeamProfile,
    exploits: list[ExploitOpportunity],
    qb_gate: dict,
) -> WinCondition:
    ranked = sorted(exploits, key=lambda item: item.score, reverse=True)
    primary = ranked[0]
    secondary = ranked[1]
    realism = max(0.0, min(100.0, 50.0 + primary.score * 2.4 + secondary.score * 1.2 + qb_gate["margin"] * 0.8))
    requirements = [
        f"The quarterback gate must remain {qb_gate['verdict'].lower()} rather than collapsing under pressure.",
        f"Convert the {primary.name.lower()} advantage into sustained offensive efficiency.",
        f"Prevent {opponent.team}'s strongest counter-exploit from controlling game script.",
    ]
    chain = [
        primary.weapon,
        f"attacks {primary.weakness.lower()}",
        primary.consequence,
        "the offense keeps its preferred game script available",
    ]
    return WinCondition(
        team=team.team,
        title=f"Win through {primary.name.lower()}",
        requirements=requirements,
        chain=chain,
        realism_score=round(realism, 1),
        explanation=(
            f"{team.team}'s cleanest path is to make {primary.name.lower()} the central conflict, "
            f"with {secondary.name.lower()} as the supporting advantage."
        ),
    )


def _failure_condition(
    team: NFLTeamProfile,
    opponent: NFLTeamProfile,
    opponent_exploits: list[ExploitOpportunity],
    qb_gate: dict,
) -> FailureCondition:
    threat = max(opponent_exploits, key=lambda item: item.score)
    if qb_gate["verdict"] == "Fail":
        title = "Quarterback environment breaks down"
        trigger = qb_gate["explanation"]
        consequence = "The rest of the roster would need to overcome an unstable passing environment, which sharply narrows the betting case."
        score = max(threat.score, abs(float(qb_gate["margin"])))
    else:
        title = f"Opponent activates {threat.name.lower()}"
        trigger = threat.explanation
        consequence = threat.consequence
        score = threat.score
    return FailureCondition(
        team=team.team,
        threat_team=opponent.team,
        title=title,
        trigger=trigger,
        consequence=consequence,
        threat_score=round(score, 2),
    )


def build_matchup_brain(
    *,
    away_team: str,
    home_team: str,
    away_components: Mapping[str, float],
    home_components: Mapping[str, float],
) -> dict:
    away_profile = profile_from_legacy_components(away_team, away_components)
    home_profile = profile_from_legacy_components(home_team, home_components)

    away_exploits = _team_exploits(away_profile, home_profile)
    home_exploits = _team_exploits(home_profile, away_profile)
    away_qb_gate = _qb_gate(away_profile, home_profile)
    home_qb_gate = _qb_gate(home_profile, away_profile)

    away_win_condition = _win_condition(away_profile, home_profile, away_exploits, away_qb_gate)
    home_win_condition = _win_condition(home_profile, away_profile, home_exploits, home_qb_gate)
    away_failure = _failure_condition(away_profile, home_profile, home_exploits, away_qb_gate)
    home_failure = _failure_condition(home_profile, away_profile, away_exploits, home_qb_gate)

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

    all_exploits = sorted(
        [*away_exploits, *home_exploits],
        key=lambda item: item.score,
        reverse=True,
    )

    return {
        "version": "NFL Brain v0.2-schema",
        "matchup_score_home": score_gap,
        "matchup_leader": leader or "Even",
        "summary": summary,
        "team_profiles": {
            away_team: away_profile.to_dict(),
            home_team: home_profile.to_dict(),
        },
        "qb_gates": {
            away_team: away_qb_gate,
            home_team: home_qb_gate,
        },
        "exploits": [asdict(item) for item in all_exploits],
        "win_conditions": {
            away_team: asdict(away_win_condition),
            home_team: asdict(home_win_condition),
        },
        "failure_conditions": {
            away_team: asdict(away_failure),
            home_team: asdict(home_failure),
        },
        "conflicts": [asdict(item) for item in ranked],
        "chain_reactions": [asdict(item) for item in chains],
        "data_contract": {
            "status": "Ready for upgraded ratings",
            "current_source": "Legacy Macabets ratings translated through a stable schema",
            "future_sources_can_replace": [
                "Madden 27 roster ratings",
                "Advanced team metrics",
                "Depth charts and injuries",
                "Player-level and scheme-level data",
            ],
        },
        "limitations": [
            "The brain now uses a stable detailed schema, but several detailed fields are provisional estimates derived from broad legacy ratings.",
            "Injuries, confirmed starters and current scheme tendencies must still be checked separately.",
        ],
    }
