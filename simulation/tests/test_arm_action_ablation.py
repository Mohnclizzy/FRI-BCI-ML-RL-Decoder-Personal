import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation.scripts import run_near_table_arm_action_ablation as ablation


class SavedActionTests(unittest.TestCase):
    def test_only_selected_arm_components_are_zeroed(self):
        rows = [
            {"record_type": "episode_start"},
            {"record_type": "policy_step_start", "policy_action": list(range(1, 10))},
            {"record_type": "policy_step_start", "policy_action": list(range(11, 20))},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(ablation, "SOURCE_TELEMETRY_SHA256", digest):
                full = ablation.saved_actions(path, "full")
                changed = ablation.saved_actions(path, "joint1_joint3_off")
                self.assertEqual(full, [list(range(1, 10)), list(range(11, 20))])
                self.assertEqual(changed[0], [0.0, 2, 0.0, 4, 5, 6, 7, 8, 9])
                self.assertEqual(changed[1], [0.0, 12, 0.0, 14, 15, 16, 17, 18, 19])

    def test_rejects_modified_source_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                ablation.saved_actions(path, "full")


if __name__ == "__main__":
    unittest.main()
