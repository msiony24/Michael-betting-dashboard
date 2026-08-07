from engine.challenge_macabets import _normalize_response


def test_challenge_revision_is_capped_per_turn():
    payload = {
        "reply": "I would revise this.",
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
