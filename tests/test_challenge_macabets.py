from engine.challenge_macabets import _normalize_response


def test_challenge_revision_is_capped_per_turn():
    payload = {
        "reply": "I would revise this.",
        "agree_points": ["The reliability concern is material."],
        "pushback_points": ["The baseline edge remains."],
        "question_to_user": "",
        "stance": "revise",
        "adjustment_category": "reliability",
        "proposed_probability_a": 0.20,
        "proposed_confidence": 40,
        "proposed_verdict": "Pass",
        "revision_summary": "Reliability concern",
        "should_offer_apply": True,
        "uses_unverified_user_claim": False,
    }
    result = _normalize_response(payload, 0.70, 82, "Strong Bet")
    assert result["proposed_probability_a"] == 0.62
    assert result["proposed_confidence"] == 72
    assert result["proposed_verdict"] == "Pass"
    assert result["should_offer_apply"] is True


def test_no_change_does_not_offer_apply():
    payload = {
        "reply": "I am keeping the original opinion.",
        "agree_points": [],
        "pushback_points": ["The supplied data still supports the original edge."],
        "question_to_user": "What specific matchup factor are you weighting differently?",
        "stance": "defend",
        "adjustment_category": "none",
        "proposed_probability_a": 0.65,
        "proposed_confidence": 78,
        "proposed_verdict": "Worth Betting",
        "revision_summary": "No change",
        "should_offer_apply": True,
        "uses_unverified_user_claim": False,
    }
    result = _normalize_response(payload, 0.65, 78, "Worth Betting")
    assert result["should_offer_apply"] is False


def test_structured_debate_fields_survive_normalization():
    payload = {
        "reply": "I would trim confidence but keep the winner.",
        "agree_points": ["Volatility is higher than the headline verdict suggests."],
        "pushback_points": ["The baseline matchup still favors Player A."],
        "question_to_user": "Are you challenging the winner or only the bet grade?",
        "stance": "partially_agree",
        "adjustment_category": "reliability",
        "proposed_probability_a": 0.61,
        "proposed_confidence": 72,
        "proposed_verdict": "Lean",
        "revision_summary": "Lower bet quality, same winner.",
        "should_offer_apply": True,
        "uses_unverified_user_claim": False,
    }
    result = _normalize_response(payload, 0.63, 76, "Strong Bet")
    assert result["agree_points"][0].startswith("Volatility")
    assert result["pushback_points"][0].startswith("The baseline")
    assert result["question_to_user"].startswith("Are you")
    assert result["proposed_verdict"] == "Lean"
