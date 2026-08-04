"""Macabets NFL decision framework and evidence-readiness contract.

The framework separates two concerns:

1. What football question Macabets wants to answer.
2. Whether Macabets has enough verified evidence to answer it responsibly.

Missing optional evidence lowers readiness. Missing required evidence blocks the
question. Legacy, estimated, stale, or unknown evidence never silently becomes
a neutral input and can never influence the prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from engine.nfl_team_schema import NFLTeamProfile


TRUSTED_QUALITIES = {"current", "verified", "current_verified", "verified_baseline"}
CURRENT_QUALITIES = {"current", "current_verified"}
BLOCKED_QUALITIES = {"legacy", "complete_legacy", "partial_legacy", "estimated", "stale", "missing", "unknown"}


@dataclass(frozen=True)
class EvidenceRequirement:
    key: str
    label: str
    scope: str
    baseline_allowed: bool = True


@dataclass(frozen=True)
class QuestionSpecification:
    number: int
    key: str
    question: str
    purpose: str
    required: tuple[EvidenceRequirement, ...]
    optional: tuple[EvidenceRequirement, ...]
    refusal_rule: str
    week_one_policy: str


@dataclass(frozen=True)
class DecisionQuestion:
    number: int
    key: str
    question: str
    purpose: str
    status: str
    answer: str
    reason: str
    readiness_score: int
    readiness_label: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    source_quality: str
    confidence_reason: str
    refusal_rule: str
    week_one_policy: str
    can_influence_prediction: bool


def _req(key: str, label: str, scope: str, baseline_allowed: bool = True) -> EvidenceRequirement:
    return EvidenceRequirement(key, label, scope, baseline_allowed)


QUESTION_SPECIFICATIONS = (
    QuestionSpecification(
        1,
        "qb_trust",
        "Do I trust this quarterback in this matchup?",
        "Determine whether the verified starting quarterback is capable of executing against this specific defense.",
        (
            _req("starting_quarterback", "Verified starting quarterback", "offense"),
            _req("quarterback_health", "Current quarterback health", "offense", False),
            _req("quarterback_baseline", "Verified quarterback baseline profile", "offense"),
            _req("pass_defense_baseline", "Opponent pass-defense baseline", "defense"),
        ),
        (
            _req("quarterback_recent_form", "Current-season quarterback form", "offense", False),
            _req("pressure_performance", "Performance under pressure", "offense", False),
            _req("blitz_performance", "Performance against blitz", "offense", False),
        ),
        "Refuse when the starter, health, baseline quarterback profile, or opponent pass-defense baseline is not verified.",
        "Week 1 is answerable from verified starter, health, quarterback baseline, and opponent baseline. No current-season form lowers readiness but does not block the answer.",
    ),
    QuestionSpecification(
        2,
        "offensive_line",
        "Can the offensive line protect and create running lanes?",
        "Evaluate whether the line can give the quarterback time and create enough space for the running game.",
        (
            _req("projected_offensive_line", "Verified projected offensive line", "offense"),
            _req("offensive_line_health", "Current offensive-line health", "offense", False),
            _req("pass_protection_baseline", "Pass-protection baseline", "offense"),
            _req("run_blocking_baseline", "Run-blocking baseline", "offense"),
            _req("pass_rush_baseline", "Opponent pass-rush baseline", "defense"),
            _req("run_defense_baseline", "Opponent run-defense baseline", "defense"),
        ),
        (
            _req("current_pass_blocking", "Current-season pass blocking", "offense", False),
            _req("pressure_rate_allowed", "Current pressure rate allowed", "offense", False),
            _req("current_run_blocking", "Current-season run blocking", "offense", False),
        ),
        "Refuse when the projected line, health, or either side's baseline trench profile is not verified.",
        "Week 1 is answerable from verified lineups, health, and baseline trench ratings. In-season efficiency improves readiness later.",
    ),
    QuestionSpecification(
        3,
        "offensive_capability",
        "Can this offense consistently move the football and score?",
        "Judge whether the offense has a trustworthy quarterback, protection, runners, and receiving weapons capable of sustaining drives.",
        (
            _req("starting_quarterback", "Verified starting quarterback", "offense"),
            _req("offensive_personnel", "Verified offensive personnel", "offense"),
            _req("offense_baseline", "Verified offense baseline", "offense"),
            _req("offensive_health", "Current offensive health", "offense", False),
        ),
        (
            _req("offense_recent_form", "Current-season offensive form", "offense", False),
            _req("drive_efficiency", "Drive efficiency", "offense", False),
            _req("red_zone_execution", "Red-zone execution", "offense", False),
        ),
        "Refuse when the quarterback, offensive personnel, baseline offense, or current health is not verified.",
        "Week 1 is answerable from verified personnel, health, and baseline offense. Current-season drive data is optional.",
    ),
    QuestionSpecification(
        4,
        "run_defense",
        "Can this defense stop the run?",
        "Compare the defense's verified front and tackling baseline with the opponent's line and rushing personnel.",
        (
            _req("defensive_front", "Verified defensive front", "defense"),
            _req("defensive_health", "Current defensive-front health", "defense", False),
            _req("run_defense_baseline", "Run-defense baseline", "defense"),
            _req("projected_offensive_line", "Opponent projected offensive line", "opponent_offense"),
            _req("rushing_personnel", "Opponent rushing personnel", "opponent_offense"),
        ),
        (
            _req("current_run_defense", "Current-season run defense", "defense", False),
            _req("tackling_form", "Current tackling form", "defense", False),
            _req("opponent_rushing_form", "Opponent current rushing form", "opponent_offense", False),
        ),
        "Refuse when the defensive front, health, run-defense baseline, or opposing rushing personnel is not verified.",
        "Week 1 is answerable from verified fronts, personnel, health, and baseline ratings. Current-season rushing samples are optional.",
    ),
    QuestionSpecification(
        5,
        "pass_defense",
        "Can this defense stop the pass?",
        "Evaluate whether pass rush and coverage personnel can contain the opponent's verified quarterback and receiving group.",
        (
            _req("pass_rush_personnel", "Verified pass-rush personnel", "defense"),
            _req("secondary_personnel", "Verified secondary personnel", "defense"),
            _req("defensive_health", "Current defensive health", "defense", False),
            _req("pass_defense_baseline", "Pass-defense baseline", "defense"),
            _req("starting_quarterback", "Opponent verified starting quarterback", "opponent_offense"),
            _req("receiving_personnel", "Opponent verified receiving personnel", "opponent_offense"),
        ),
        (
            _req("current_pass_defense", "Current-season pass defense", "defense", False),
            _req("current_pass_rush", "Current-season pass rush", "defense", False),
            _req("opponent_passing_form", "Opponent current passing form", "opponent_offense", False),
        ),
        "Refuse when pass-rush personnel, secondary, health, baseline pass defense, or opposing passing personnel is not verified.",
        "Week 1 is answerable from verified personnel, health, and baseline pass-defense profiles. Current-season pass data is optional.",
    ),
    QuestionSpecification(
        6,
        "better_team",
        "Which is the better overall football team?",
        "Establish the baseline team-strength edge before matchup-specific adjustments.",
        (
            _req("current_roster", "Verified current roster", "team"),
            _req("team_health", "Current team health", "team", False),
            _req("offense_baseline", "Verified offense baseline", "team"),
            _req("defense_baseline", "Verified defense baseline", "team"),
            _req("depth_baseline", "Verified depth baseline", "team"),
        ),
        (
            _req("team_recent_form", "Current-season team form", "team", False),
            _req("continuity", "Roster and coaching continuity", "team"),
            _req("special_teams", "Special-teams baseline", "team"),
        ),
        "Refuse when either current roster, health, offense, defense, or depth baseline is unverified.",
        "Week 1 is answerable from verified rosters, health, and preseason baseline ratings. Recent form is not required.",
    ),
    QuestionSpecification(
        7,
        "exploits",
        "What can each team exploit?",
        "Identify where a verified team strength directly intersects with a verified opponent weakness.",
        (
            _req("current_roster", "Verified current rosters", "both_teams"),
            _req("unit_strengths", "Verified unit strengths", "both_teams"),
            _req("unit_weaknesses", "Verified unit weaknesses", "both_teams"),
            _req("personnel_health", "Current relevant personnel health", "both_teams", False),
        ),
        (
            _req("scheme_tendencies", "Current scheme tendencies", "both_teams", False),
            _req("matchup_history", "Relevant matchup history", "both_teams", False),
            _req("recent_unit_form", "Current unit form", "both_teams", False),
        ),
        "Refuse when the rosters, unit strengths, unit weaknesses, or relevant health are not verified for both teams.",
        "Week 1 is answerable from verified rosters and baseline unit profiles. Scheme and recent-form evidence improve specificity later.",
    ),
    QuestionSpecification(
        8,
        "critical_plays",
        "Which team is more likely to execute the critical plays?",
        "Estimate which team is better equipped for the small number of third-down, red-zone, late-game, and pressure moments that decide the game.",
        (
            _req("quarterback_baseline", "Verified quarterback baseline", "both_teams"),
            _req("situational_baseline", "Verified situational execution baseline", "both_teams"),
            _req("current_roster", "Verified current rosters", "both_teams"),
            _req("team_health", "Current team health", "both_teams", False),
        ),
        (
            _req("current_third_down", "Current-season third-down execution", "both_teams", False),
            _req("current_red_zone", "Current-season red-zone execution", "both_teams", False),
            _req("late_game_form", "Current late-game execution", "both_teams", False),
        ),
        "Refuse when quarterback baselines, situational baselines, current rosters, or health are not verified for both teams.",
        "Week 1 is answerable from verified career/prior-season situational baselines and current rosters. Current-season situational samples are optional.",
    ),
)

QUESTION_DEFINITIONS = tuple(
    (spec.number, spec.key, spec.question, tuple(req.key for req in spec.required))
    for spec in QUESTION_SPECIFICATIONS
)


def _normalized_quality(value: object) -> str:
    return str(value or "missing").strip().lower()


def _evidence_quality(
    evidence: Mapping[str, Mapping[str, str]] | None,
    team: str,
    key: str,
) -> str:
    if not evidence:
        return "missing"
    team_evidence = evidence.get(team, {})
    return _normalized_quality(team_evidence.get(key, "missing"))


def _requirement_teams(requirement: EvidenceRequirement, away: str, home: str) -> tuple[str, ...]:
    if requirement.scope in {"both_teams", "team"}:
        return (away, home)
    if requirement.scope in {"offense", "defense"}:
        return (away, home)
    if requirement.scope in {"opponent_offense", "opponent_defense"}:
        return (away, home)
    return (away, home)


def _quality_satisfies(requirement: EvidenceRequirement, quality: str) -> bool:
    if quality in CURRENT_QUALITIES:
        return True
    if requirement.baseline_allowed and quality == "verified_baseline":
        return True
    return False


def _profile_is_legacy(profile: NFLTeamProfile) -> bool:
    return _normalized_quality(profile.data_quality) in BLOCKED_QUALITIES


def _evaluate_spec(
    spec: QuestionSpecification,
    away: NFLTeamProfile,
    home: NFLTeamProfile,
    evidence: Mapping[str, Mapping[str, str]] | None,
) -> DecisionQuestion:
    missing_required: list[str] = []
    missing_optional: list[str] = []
    required_checks = 0
    required_satisfied = 0
    optional_checks = 0
    optional_satisfied = 0

    for requirement in spec.required:
        for team in _requirement_teams(requirement, away.team, home.team):
            required_checks += 1
            quality = _evidence_quality(evidence, team, requirement.key)
            if _quality_satisfies(requirement, quality):
                required_satisfied += 1
            else:
                missing_required.append(f"{team}: {requirement.label}")

    for requirement in spec.optional:
        for team in _requirement_teams(requirement, away.team, home.team):
            optional_checks += 1
            quality = _evidence_quality(evidence, team, requirement.key)
            if _quality_satisfies(requirement, quality):
                optional_satisfied += 1
            else:
                missing_optional.append(f"{team}: {requirement.label}")

    required_ratio = required_satisfied / required_checks if required_checks else 0.0
    optional_ratio = optional_satisfied / optional_checks if optional_checks else 0.0
    readiness_score = round((required_ratio * 80.0) + (optional_ratio * 20.0))

    if missing_required:
        status = "insufficient_current_data"
        answer = "Insufficient current data"
        readiness_label = "Blocked"
        reason = (
            "Macabets is missing required verified evidence and will not answer this question "
            "from legacy, stale, estimated, or assumed inputs."
        )
        confidence_reason = (
            f"{required_satisfied}/{required_checks} required evidence checks passed. "
            "All required checks must pass before the football conclusion can be scored."
        )
    else:
        status = "ready_for_scoring"
        answer = "Ready for validated scoring"
        if optional_ratio >= 0.8:
            readiness_label = "High readiness"
        elif optional_ratio >= 0.35:
            readiness_label = "Moderate readiness"
        else:
            readiness_label = "Baseline readiness"
        reason = (
            "All required evidence is verified. The question may be scored once its validated "
            "football scoring rule is connected."
        )
        if missing_optional:
            confidence_reason = (
                "Required evidence is complete, but optional current-season evidence is incomplete. "
                "This lowers confidence without blocking the answer."
            )
        else:
            confidence_reason = "Required and optional evidence checks are complete."

    source_quality = f"{away.team}: {away.data_quality}; {home.team}: {home.data_quality}"

    return DecisionQuestion(
        number=spec.number,
        key=spec.key,
        question=spec.question,
        purpose=spec.purpose,
        status=status,
        answer=answer,
        reason=reason,
        readiness_score=readiness_score,
        readiness_label=readiness_label,
        required_inputs=tuple(req.label for req in spec.required),
        optional_inputs=tuple(req.label for req in spec.optional),
        missing_required=tuple(missing_required),
        missing_optional=tuple(missing_optional),
        source_quality=source_quality,
        confidence_reason=confidence_reason,
        refusal_rule=spec.refusal_rule,
        week_one_policy=spec.week_one_policy,
        can_influence_prediction=False,
    )


def build_decision_framework(
    away: NFLTeamProfile,
    home: NFLTeamProfile,
    evidence: Mapping[str, Mapping[str, str]] | None = None,
    season_week: int | None = None,
) -> dict:
    """Build readiness results for Michael's eight football questions.

    ``evidence`` is intentionally source-agnostic. A future automated data engine
    can populate each team's evidence keys from Madden, roster/depth-chart feeds,
    injuries, advanced metrics, or a validated blended model.
    """

    questions = [
        _evaluate_spec(spec, away, home, evidence)
        for spec in QUESTION_SPECIFICATIONS
    ]

    ready_count = sum(question.status == "ready_for_scoring" for question in questions)
    legacy_profiles = _profile_is_legacy(away) or _profile_is_legacy(home)

    if ready_count == len(questions):
        status = "data_ready"
        message = "All eight questions have the required verified evidence. Scoring remains disabled until validated rules are activated."
    elif ready_count > 0:
        status = "partially_ready"
        message = f"{ready_count} of 8 questions have enough verified evidence. Missing required evidence blocks the remaining questions."
    else:
        status = "blocked_by_data_quality"
        message = "Decision Framework blocked: required verified NFL evidence is not connected."

    if legacy_profiles and not evidence:
        status = "blocked_by_data_quality"

    return {
        "version": "Decision Framework v1.1-readiness",
        "status": status,
        "prediction_influence": "disabled",
        "season_week": season_week,
        "ready_questions": ready_count,
        "message": message,
        "questions": [asdict(item) for item in questions],
        "guardrails": [
            "Missing optional evidence lowers readiness but does not automatically block an answer.",
            "Missing required evidence blocks the question.",
            "Estimated, stale, legacy, missing, or unknown evidence cannot create a football conclusion.",
            "An unanswered question cannot be converted into a neutral score.",
            "No framework answer can change win probability, confidence, fair line, or BET/PASS until its scoring rule and data feeds are validated.",
            "Every future conclusion must expose its evidence quality and freshness.",
        ],
    }
