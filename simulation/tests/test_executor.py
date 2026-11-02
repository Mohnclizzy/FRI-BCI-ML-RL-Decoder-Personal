import json
from pathlib import Path
import tempfile
import unittest

from fri_bci_sim.commands import Skill
from fri_bci_sim.executor import DecisionReplay, ExecutionMonitor, HoldTargetLatch, SkillExecutor, ExecutorState


def row(end=1.0, available=1.0, issued=1.0, skill="grasp_lift", p_hand=0.9):
    return {
        "estimate": {"source": "S006", "episode": "R12", "window_end_s": end,
                     "available_at_s": available, "p_rest": 1-p_hand, "p_hand": p_hand},
        "decisions": {"temporal": {"issued_at_s": issued, "skill": skill, "reason": "fixture"}},
    }


class DecisionReplayTest(unittest.TestCase):
    def replay(self, rows, age=0.5):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        return DecisionReplay(path, "temporal", age)

    def test_waits_for_future_issue_time_then_holds_when_stale(self):
        replay = self.replay([row(end=1, available=1, issued=1.2)], age=0.5)
        self.assertEqual(replay.at(1.19).reason, "awaiting_estimate")
        self.assertEqual(replay.at(1.2).skill, Skill.GRASP_LIFT)
        self.assertEqual(replay.at(1.501).reason, "stale")

    def test_unrounded_simulator_clock_preserves_boundary_decision(self):
        issue_time = 1.00004
        replay = self.replay([row(end=1.0, available=1.0, issued=issue_time)], age=1.0)
        self.assertEqual(replay.at(issue_time).skill, Skill.GRASP_LIFT)
        self.assertEqual(replay.at(round(issue_time, 4)).reason, "awaiting_estimate")

    def test_invalid_probability_and_causality_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid decision trace"):
            self.replay([row(p_hand=1.2)])
        with self.assertRaisesRegex(ValueError, "causality"):
            self.replay([row(end=2, available=2, issued=1.9)])

    def test_duplicate_and_out_of_order_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate or out-of-order"):
            self.replay([row(end=1), row(end=1)])
        with self.assertRaisesRegex(ValueError, "Duplicate or out-of-order"):
            self.replay([row(end=2, available=2, issued=2), row(end=1, available=1, issued=3)])

    def test_missing_trace_and_trace_for_explicit_control_are_rejected(self):
        with self.assertRaises(ValueError):
            DecisionReplay(None, "temporal")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                DecisionReplay(path, "always-hold")


class HoldAndCompletionTest(unittest.TestCase):
    def test_hold_preserves_arm_and_finger_targets_despite_measured_drift(self):
        latch = HoldTargetLatch([0.1, -0.2], 0.03)
        arm, fingers = latch.resolve(False, [0.2, -0.3], 0.0)
        self.assertEqual((arm, fingers), ([0.2, -0.3], 0.0))
        # Proposed values can reflect measured pose after drift; HOLD ignores them.
        arm, fingers = latch.resolve(True, [9.0, 9.0], 0.04)
        self.assertEqual((arm, fingers), ([0.2, -0.3], 0.0))
        arm[0] = 99
        self.assertEqual(latch.hold(), ([0.2, -0.3], 0.0))

    def test_complete_is_absorbing_even_when_active_request_returns(self):
        executor = SkillExecutor()
        executor.update(type("Request", (), {"skill": Skill.GRASP_LIFT})())
        done = executor.update(type("Request", (), {"skill": Skill.GRASP_LIFT})(), skill_complete=True)
        self.assertEqual(done.state, ExecutorState.COMPLETE)
        resumed = executor.update(type("Request", (), {"skill": Skill.GRASP_LIFT})())
        self.assertEqual(resumed.state, ExecutorState.COMPLETE)
        self.assertTrue(resumed.hold_position)

    def test_hold_monitor_compares_held_targets_and_records_physical_drift(self):
        monitor = ExecutionMonitor([0.0], [0.0, 0.0, 0.0], 0.0)
        monitor.observe([0.2], [0.03, 0.0, 0.0], 0.0, [0, 0], holding=True,
                        arm_target=[0.0], finger_target=0.04, dt=0.01)
        summary = monitor.summary()
        self.assertAlmostEqual(summary["max_hand_displacement_m"], 0.03)
        self.assertAlmostEqual(summary["max_hold_hand_displacement_m"], 0.03)
        self.assertAlmostEqual(summary["max_hold_joint_displacement_rad"], 0.2)


if __name__ == "__main__":
    unittest.main()
