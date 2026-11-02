import json
from pathlib import Path
import tempfile
import unittest

from fri_bci_sim.replay_evaluation import (
    AnnotatedTrial,
    LabelInterval,
    evaluate_decision_trace,
)


def row(time_s, skill, *, source="s", episode="trial", estimate_end=None):
    estimate_end = time_s if estimate_end is None else estimate_end
    return {
        "estimate": {"source": source, "episode": episode,
                     "window_end_s": estimate_end, "available_at_s": time_s,
                     "p_rest": 0.5, "p_hand": 0.5},
        "decisions": {"raw": {"skill": skill, "reason": "test", "issued_at_s": time_s}},
    }


class ReplayEvaluationMetricsTest(unittest.TestCase):
    def test_rest_false_activation_delay_switches_and_censored_horizon(self):
        trial = AnnotatedTrial("s", "trial", 10.0, (
            LabelInterval(0.0, 4.0, "rest"),
            LabelInterval(4.0, 8.0, "imagery"),
            LabelInterval(8.0, 10.0, "rest"),
        ))
        rows = [row(1, "grasp_lift"), row(3, "hold"), row(5, "grasp_lift"), row(7, "hold")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            result = evaluate_decision_trace(path, [trial])["conditions"]["raw"]
        self.assertAlmostEqual(result["false_activation_duration_s"], 2.0)
        self.assertEqual(result["detected_bouts"], 1)
        self.assertEqual(result["missed_bouts"], 0)
        self.assertAlmostEqual(result["mean_detection_delay_s"], 1.0)
        self.assertEqual(result["state_switches"], 4)
        self.assertEqual(result["truncated_trials"], 1)
        self.assertAlmostEqual(result["truncated_duration_s"], 3.0)
        self.assertEqual(result["censored_bouts"], 0)

    def test_validation_rejects_future_labels_as_early_decisions_and_out_of_order_events(self):
        trial = AnnotatedTrial("s", "trial", 3.0, (LabelInterval(0, 3, "rest"),))
        bad_future = row(1, "hold")
        bad_future["decisions"]["raw"]["issued_at_s"] = 0.5
        out_of_order = [row(2, "hold"), row(1, "hold")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text(json.dumps(bad_future), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "precedes estimate availability"):
                evaluate_decision_trace(path, [trial])
            path.write_text("\n".join(json.dumps(r) for r in out_of_order), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "out of event order"):
                evaluate_decision_trace(path, [trial])

    def test_partial_imagery_is_censored_not_called_a_miss(self):
        trial = AnnotatedTrial("s", "trial", 8.0, (
            LabelInterval(0, 4, "rest"), LabelInterval(4, 8, "imagery"),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text(json.dumps(row(5, "hold")), encoding="utf-8")
            result = evaluate_decision_trace(path, [trial])["conditions"]["raw"]
        self.assertEqual(result["missed_bouts"], 0)
        self.assertEqual(result["censored_bouts"], 1)
        self.assertIsNone(result["detection_coverage_fraction"])


if __name__ == "__main__":
    unittest.main()
