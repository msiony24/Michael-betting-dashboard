from engine.nfl_decision_framework import (
    QUESTION_SPECIFICATIONS,
    build_decision_framework,
)
from engine.nfl_team_schema import profile_from_legacy_components


COMPONENTS = {
    "quarterback": 85,
    "offense": 82,
    "defense": 80,
    "coaching": 84,
    "offensive_line": 81,
    "defensive_line": 83,
    "skill_positions": 82,
    "secondary": 79,
    "special_teams": 75,
    "continuity": 80,
}


def _profiles():
    return (
        profile_from_legacy_components("Away", COMPONENTS),
        profile_from_legacy_components("Home", COMPONENTS),
    )


def _question(result, key):
    return next(item for item in result["questions"] if item["key"] == key)


def test_week_one_qb_question_is_ready_without_recent_form():
    away, home = _profiles()
    evidence = {
        team: {
            "starting_quarterback": "current_verified",
            "quarterback_health": "current_verified",
            "quarterback_baseline": "verified_baseline",
            "pass_defense_baseline": "verified_baseline",
        }
        for team in ("Away", "Home")
    }

    result = build_decision_framework(away, home, evidence=evidence, season_week=1)
    qb = _question(result, "qb_trust")

    assert qb["status"] == "ready_for_scoring"
    assert qb["answer"] == "Ready for validated scoring"
    assert qb["readiness_label"] == "Baseline readiness"
    assert qb["missing_required"] == ()
    assert qb["missing_optional"]
    assert qb["can_influence_prediction"] is False


def test_stale_required_evidence_blocks_question():
    away, home = _profiles()
    evidence = {
        team: {
            "starting_quarterback": "current_verified",
            "quarterback_health": "stale",
            "quarterback_baseline": "verified_baseline",
            "pass_defense_baseline": "verified_baseline",
        }
        for team in ("Away", "Home")
    }

    result = build_decision_framework(away, home, evidence=evidence)
    qb = _question(result, "qb_trust")

    assert qb["status"] == "insufficient_current_data"
    assert qb["readiness_label"] == "Blocked"
    assert any("health" in item.lower() for item in qb["missing_required"])


def test_all_required_evidence_can_make_all_questions_ready_without_optional_data():
    away, home = _profiles()
    required_keys = {
        requirement.key
        for specification in QUESTION_SPECIFICATIONS
        for requirement in specification.required
    }
    evidence = {
        team: {
            key: (
                "current_verified"
                if any(
                    requirement.key == key and not requirement.baseline_allowed
                    for specification in QUESTION_SPECIFICATIONS
                    for requirement in specification.required
                )
                else "verified_baseline"
            )
            for key in required_keys
        }
        for team in ("Away", "Home")
    }

    result = build_decision_framework(away, home, evidence=evidence, season_week=1)

    assert result["status"] == "data_ready"
    assert result["ready_questions"] == 8
    assert all(item["status"] == "ready_for_scoring" for item in result["questions"])
    assert all(item["can_influence_prediction"] is False for item in result["questions"])
