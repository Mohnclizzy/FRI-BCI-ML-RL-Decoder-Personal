import json
from pathlib import Path
import tempfile
import unittest

from fri_bci_sim.commands import CommandGate, Estimate, Skill
from fri_bci_sim.launch import build_command
from fri_bci_sim.metrics import Outcome, summarize
from fri_bci_sim.replay import load_trial_estimates


def estimate(t=2.0, episode="a", hand=0.9):
    return Estimate("subject1", episode, t, t, 1-hand, hand)


class ContractsTest(unittest.TestCase):
    def test_temporal_gate_needs_new_evidence(self):
        gate = CommandGate()
        self.assertEqual(gate.update(estimate(), 2).reason, "collecting_evidence")
        self.assertEqual(gate.update(estimate(), 2).reason, "duplicate_or_out_of_order")
        self.assertEqual(gate.update(estimate(2.3), 2.3).skill, Skill.HOLD)
        self.assertEqual(gate.update(estimate(2.6), 2.6).skill, Skill.GRASP_LIFT)

    def test_rest_stale_future_and_episode_boundary(self):
        gate = CommandGate()
        gate.update(estimate(), 2)
        self.assertEqual(gate.update(estimate(2.3, episode="b"), 2.3).skill, Skill.HOLD)
        self.assertEqual(gate.update(estimate(2.6, episode="b", hand=0.1), 2.6).reason, "rest")
        self.assertEqual(gate.update(estimate(3), 5).reason, "stale")
        self.assertEqual(gate.update(estimate(3), 2).reason, "not_yet_available")

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            Estimate("s", "a", 1, 1, 0.7, 0.7)
        with self.assertRaises(ValueError):
            Estimate("s", "a", 1, 1, float("nan"), 0.5)

    def test_adapter_discards_true_label(self):
        row = {"trial_id": "t", "subject": 1, "run": 12, "window_end_seconds": 3,
               "available_at_recording_seconds": 3, "probabilities": {"REST": 0.2, "HAND_IMAGERY": 0.8},
               "label_id": 0, "label": "REST"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(json.dumps([row]), encoding="utf-8")
            first = load_trial_estimates(path)
            row["label_id"], row["label"] = 1, "HAND_IMAGERY"
            path.write_text(json.dumps([row]), encoding="utf-8")
            self.assertEqual(first, load_trial_estimates(path))
            path.write_text(json.dumps([row, row]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trial_estimates(path)

    def test_metrics_cannot_hide_missing_class(self):
        hold = Outcome("a", "hold", True, False, False, 2)
        grasp = Outcome("b", "grasp_lift", False, False, False, 4)
        self.assertIsNone(summarize([hold])["balanced_success"])
        self.assertEqual(summarize([hold, grasp])["balanced_success"], 0.5)
        with self.assertRaises(ValueError):
            summarize([hold, hold])

    def test_launch_is_argument_list_and_requires_explicit_play_checkpoint(self):
        command = build_command(Path("path with spaces"), "train", 16, 0, 2)
        self.assertIn("FRI-Franka-Lift", command)
        self.assertEqual(command[-1], "physics=newton_mjwarp")
        with self.assertRaises(ValueError):
            build_command(Path("root"), "play", 1, 0, 2)


if __name__ == "__main__":
    unittest.main()
