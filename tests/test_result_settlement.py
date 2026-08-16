from __future__ import annotations

import unittest

from engine.result_settlement import (
    clv_metrics,
    consensus_moneyline_close,
    grade_moneyline_prediction,
    grade_spread,
    grade_total,
    name_signature,
    names_compatible,
    no_vig_two_way_probability,
)
from engine.settlement_providers import APITennisSettlementClient
from settle_results import settle_tennis


class IdentityTests(unittest.TestCase):
    def test_cerundolo_initials_do_not_collapse(self):
        self.assertEqual(name_signature("Juan Manuel Cerundolo").surname, "cerundolo")
        self.assertEqual(name_signature("Juan Manuel Cerundolo").first_initial, "j")
        self.assertEqual(name_signature("Cerundolo J.M.").first_initial, "j")
        self.assertTrue(names_compatible("Juan Manuel Cerundolo", "Cerundolo J.M."))
        self.assertFalse(names_compatible("Juan Manuel Cerundolo", "Cerundolo F."))

    def test_full_and_initial_name_match(self):
        self.assertTrue(names_compatible("Jannik Sinner", "Sinner J."))
        self.assertFalse(names_compatible("Jannik Sinner", "Sinner A."))


class MoneylineSettlementTests(unittest.TestCase):
    def test_actionable_win_grades_prediction_and_value_call(self):
        grade = grade_moneyline_prediction(
            prediction="Jannik Sinner",
            actual_winner="Sinner J.",
            recommendation="Worth Betting",
            provider_status="Finished",
        )
        self.assertEqual(grade.status, "Won")
        self.assertTrue(grade.prediction_correct)
        self.assertTrue(grade.value_call_correct)
        self.assertEqual(grade.value_call_result, "Bet won")

    def test_actionable_loss(self):
        grade = grade_moneyline_prediction(
            prediction="Player A",
            actual_winner="Player B",
            recommendation="Strong Bet",
            provider_status="Finished",
        )
        self.assertEqual(grade.status, "Lost")
        self.assertFalse(grade.prediction_correct)
        self.assertFalse(grade.value_call_correct)

    def test_pass_does_not_claim_value_correctness_from_one_result(self):
        grade = grade_moneyline_prediction(
            prediction="Jannik Sinner",
            actual_winner="Sinner J.",
            recommendation="Pass",
            provider_status="Finished",
        )
        self.assertEqual(grade.status, "Won")
        self.assertTrue(grade.prediction_correct)
        self.assertIsNone(grade.value_call_correct)
        self.assertIn("No wager", grade.value_call_result)

    def test_retirement_requires_manual_review(self):
        grade = grade_moneyline_prediction(
            prediction="Player A",
            actual_winner="Player A",
            recommendation="Worth Betting",
            provider_status="Retired",
        )
        self.assertEqual(grade.status, "Pending")
        self.assertIsNone(grade.prediction_correct)
        self.assertIsNone(grade.value_call_correct)

    def test_retirement_text_in_tennis_score_forces_manual_review(self):
        row = {"prediction": "Player A", "recommendation": "Worth Betting"}
        link = {
            "fixture": {
                "event_status": "Finished",
                "event_first_player": "Player A",
                "event_second_player": "Player B",
                "event_winner": "First Player",
                "event_final_result": "6-3 2-1 Retired",
            }
        }
        changes, _ = settle_tennis(row, link)
        self.assertIsNotNone(changes)
        self.assertEqual(changes["status"], "Pending")
        self.assertEqual(changes["provider_link_status"], "needs_review")


class MarketGradingTests(unittest.TestCase):
    def test_spread_win_loss_push(self):
        self.assertEqual(grade_spread(selected_score=24, opponent_score=20, spread=-3.5), "Won")
        self.assertEqual(grade_spread(selected_score=20, opponent_score=24, spread=3.0), "Lost")
        self.assertEqual(grade_spread(selected_score=20, opponent_score=24, spread=4.0), "Push")

    def test_total_win_loss_push(self):
        self.assertEqual(grade_total(combined_score=48, total_line=45.5, side="Over"), "Won")
        self.assertEqual(grade_total(combined_score=42, total_line=45.5, side="Over"), "Lost")
        self.assertEqual(grade_total(combined_score=45, total_line=45, side="Under"), "Push")


class ClosingLineTests(unittest.TestCase):
    def test_consensus_close_uses_two_way_no_vig_books(self):
        rows = [
            {"bookmaker": "Book 1", "participant": "A", "american_odds": -180},
            {"bookmaker": "Book 1", "participant": "B", "american_odds": 155},
            {"bookmaker": "Book 2", "participant": "A", "american_odds": -175},
            {"bookmaker": "Book 2", "participant": "B", "american_odds": 150},
            # Incomplete book must be ignored.
            {"bookmaker": "Book 3", "participant": "A", "american_odds": -190},
        ]
        close = consensus_moneyline_close(
            rows,
            prediction="A",
            participant_a="A",
            participant_b="B",
        )
        self.assertIsNotNone(close)
        self.assertEqual(close["book_count"], 2)
        self.assertEqual(close["closing_book"], "consensus_median")
        self.assertTrue(0.60 < close["closing_no_vig_probability"] < 0.70)

    def test_positive_clv_when_market_moves_toward_prediction(self):
        row = {
            "prediction": "A",
            "participant_a": "A",
            "participant_b": "B",
            "market_odds_a": -150,
            "market_odds_b": 130,
            "predicted_probability": 0.67,
        }
        entry = no_vig_two_way_probability(-150, 130)
        metrics = clv_metrics(row=row, closing_no_vig_probability=entry + 0.03)
        self.assertAlmostEqual(metrics["clv_probability"], 0.03, places=8)
        self.assertAlmostEqual(metrics["model_edge_at_close"], 0.67 - (entry + 0.03), places=8)


class APITennisPayloadTests(unittest.TestCase):
    def test_first_player_winner_maps_to_name(self):
        fixture = {
            "event_first_player": "Cerundolo J.M.",
            "event_second_player": "Rinderknech A.",
            "event_winner": "First Player",
        }
        self.assertEqual(
            APITennisSettlementClient.actual_winner(fixture),
            "Cerundolo J.M.",
        )

    def test_second_player_winner_maps_to_name(self):
        fixture = {
            "event_first_player": "Cerundolo J.M.",
            "event_second_player": "Rinderknech A.",
            "event_winner": "Second Player",
        }
        self.assertEqual(
            APITennisSettlementClient.actual_winner(fixture),
            "Rinderknech A.",
        )


if __name__ == "__main__":
    unittest.main()
