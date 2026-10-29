import unittest
from engine import choose_action, normalize_outcome, historical_oof_predictions
from llm_address_review import _extract_json
import pandas as pd

class TestDecisionRules(unittest.TestCase):
    def test_low_risk_ships(self):
        self.assertEqual(choose_action(0.10, 0)[0], "Ship as-is")
    def test_medium_risk_confirmation(self):
        self.assertEqual(choose_action(0.25, 0)[0], "Ship after confirmation")
    def test_elevated_risk_prepaid(self):
        self.assertEqual(choose_action(0.40, 0)[0], "Nudge to prepaid")
    def test_high_risk_repeat_rto_hold(self):
        self.assertEqual(choose_action(0.60, 1)[0], "Hold / cancel")
    def test_high_risk_new_customer_not_auto_cancelled(self):
        self.assertNotEqual(choose_action(0.60, 0)[0], "Hold / cancel")
    def test_oof_predictions_are_complete_and_bounded(self):
        result = historical_oof_predictions("data/orders_sample.csv")
        self.assertTrue(result["risk_probability"].notna().all())
        self.assertTrue(result["risk_probability"].between(0, 1).all())
        self.assertEqual(set(result["target_rto"].unique()), {0, 1})
        self.assertTrue(result["evaluation_method"].str.contains("OOF").all())

    def test_local_llm_json_parser_handles_fenced_json(self):
        result = _extract_json('```json\\n{"address_ambiguous": true, "issues": ["landmark unclear"]}\\n```')
        self.assertTrue(result["address_ambiguous"])
        self.assertEqual(result["issues"], ["landmark unclear"])

    def test_status_normalization(self):
        result = normalize_outcome(pd.Series(["Rto", " delivered ", "IN_TRANSIT"]))
        self.assertEqual(result.iloc[0], 1)
        self.assertEqual(result.iloc[1], 0)
        self.assertTrue(pd.isna(result.iloc[2]))

if __name__ == "__main__":
    unittest.main()
