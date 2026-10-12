"""Offline results use recorded budgets, not the analyst machine's speed."""
import math
import unittest
from autodrive.tools.replay_outer_loop import recorded_processing_budget


class RecordedBudgetTests(unittest.TestCase):
    def test_exact_recorded_gate_wins_over_rounded_milliseconds(self):
        self.assertIs(recorded_processing_budget({'stationary_perception_budget_valid':'False',
            'inference_ms':'100', 'boundary_ms':'150'}, .25), False)
        self.assertIs(recorded_processing_budget({'stationary_perception_budget_valid':'True',
            'inference_ms':'100.001', 'boundary_ms':'150'}, .25), True)

    def test_legacy_times_and_explicit_veto(self):
        self.assertIs(recorded_processing_budget({'inference_ms':'200','boundary_ms':'60'},.25),False)
        self.assertIs(recorded_processing_budget({'inference_ms':'10','boundary_ms':'20'},.25),True)
        self.assertIs(recorded_processing_budget({'reason':'inference exceeded safety limit'},.25),False)
        self.assertIsNone(recorded_processing_budget({},.25))

    def test_malformed_timing_is_rejected(self):
        for row in ({'stationary_perception_budget_valid':'yes'},
                    {'inference_ms':'nan','boundary_ms':'1'},
                    {'inference_ms':'-1','boundary_ms':'1'}):
            with self.assertRaises(ValueError):recorded_processing_budget(row,.25)
