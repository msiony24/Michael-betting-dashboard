"""Macabets NFL decision framework with strict data-quality gates.

The framework asks Michael's eight football questions, but it will not answer
one from legacy, estimated, stale, or missing inputs. This prevents polished
language from being mistaken for verified football analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from engine.nfl_team_schema import NFLTeamProfile


TRUSTED_QUALITIES = {"current", "verified", "current_verified"}


@dataclass(frozen=True)
class DecisionQuestion:
    number: int
    key: str
    question: str
    status: str
    answer: str
    reason: str
    required_inputs: tuple[str, ...]
    source_quality: str
    can_influence_prediction: bool


QUESTION_DEFINITIONS = (
    (1, "qb_trust", "Do I trust this quarterback in this matchup?", ("quarterback", "pass_rush", "cornerbacks", "safeties")),
    (2, "offensive_line", "Can the offensive line protect and create running lanes?", ("pass_protection", "run_blocking", "pass_rush", "run_defense")),
    (3, "offensive_capability", "Can this offense consistently move the football and score?", ("quarterback", "receiving_weapons", "running_backs", "offense_overall")),
    (4, "run_defense", "Can this defense stop the run?", ("run_defense", "opponent_run_blocking", "opponent_running_backs")),
    (5, "pass_defense", "Can this defense stop the pass?", ("pass_rush", "cornerbacks", "safeties", "opponent_quarterback", "opponent_receiving_weapons")),
    (6, "better_team", "Which is the better overall football team?", ("offense_overall", "defense_overall", "depth", "coaching")),
    (7, "exploits", "What can each team exploit?", ("current_unit_ratings", "current_personnel")),
    (8, "critical_plays", "Which team is more likely to execute the critical plays?", ("situational_offense", "situational_defense", "late_game_execution")),
)


def _trusted(profile: NFLTeamProfile) -> bool:
    return profile.data_quality in TRUSTED_QUALITIES


def build_decision_framework(
    away: NFLTeamProfile,
    home: NFLTeamProfile,
) -> dict:
    trusted = _trusted(away) and _trusted(home)
    questions: list[DecisionQuestion] = []

    for number, key, question, inputs in QUESTION_DEFINITIONS:
        if trusted:
            # The live-data scoring rules will populate these answers later.
            status = "ready_for_scoring"
            answer = "Not scored yet"
            reason = "Current verified inputs are available, but this question's scoring rule has not been activated."
        else:
            status = "insufficient_current_data"
            answer = "Insufficient current data"
            reason = (
                "Macabets will not answer this question from legacy or estimated ratings. "
                "Verified, current roster and performance inputs are required."
            )

        questions.append(
            DecisionQuestion(
                number=number,
                key=key,
                question=question,
                status=status,
                answer=answer,
                reason=reason,
                required_inputs=tuple(inputs),
                source_quality=f"{away.team}: {away.data_quality}; {home.team}: {home.data_quality}",
                can_influence_prediction=False,
            )
        )

    return {
        "version": "Decision Framework v1.0-strict",
        "status": "data_ready" if trusted else "blocked_by_data_quality",
        "prediction_influence": "disabled",
        "message": (
            "Verified current inputs are available, but decision scoring remains disabled until validated."
            if trusted
            else "Decision Framework blocked: current verified NFL inputs are not connected."
        ),
        "questions": [asdict(item) for item in questions],
        "guardrails": [
            "Estimated, stale, legacy, or missing values cannot create a football conclusion.",
            "An unanswered question cannot be converted into a neutral score.",
            "The framework cannot change win probability, confidence, fair line, or BET/PASS status until validated current data is present.",
            "Macabets must expose the source quality behind every future answer.",
        ],
    }
