import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from fri_bci_sim.curriculum import build_stage_plan, main


class CurriculumPlanTest(unittest.TestCase):
    def test_stage_selection_and_checkpoint_transfer(self):
        root = Path("IsaacLab")
        pregrasp = build_stage_plan(root, "pregrasp", 16, 7, 2, None)
        self.assertEqual(pregrasp["task"], "FRI-Franka-Lift-Pregrasp")
        self.assertEqual(pregrasp["reset_protocol"], "in_hand_pregrasp")
        self.assertNotIn("--checkpoint", pregrasp["argv"])
        self.assertEqual(pregrasp["argv"][-2:], ["--viz", "none"])

        checkpoint = Path("pilot") / "model_10.pt"
        table = build_stage_plan(root, "table", 16, 8, 2, checkpoint)
        self.assertEqual(table["task"], "FRI-Franka-Lift")
        self.assertEqual(table["reset_protocol"], "table_only")
        self.assertEqual(table["argv"][table["argv"].index("--checkpoint") + 1], str(checkpoint.resolve()))
        self.assertFalse(table["measured_table_pickup_evaluation"])

        approach = build_stage_plan(root, "table-approach", 16, 8, 2, checkpoint)
        self.assertEqual(approach["task"], "FRI-Franka-Lift-Approach")
        self.assertEqual(approach["reward_variant"], "approach_contact_shaped_v1")
        self.assertIn("--checkpoint", approach["argv"])

        near = build_stage_plan(root, "near-table", 16, 9, 2, None)
        self.assertEqual(near["task"], "FRI-Franka-Lift-NearTable")
        self.assertEqual(near["reset_protocol"], "table_near_grasp_open")
        self.assertEqual(near["reward_variant"], "approach_contact_shaped_v1")
        self.assertNotIn("--checkpoint", near["argv"])

        progress = build_stage_plan(root, "near-table-progress", 256, 8, 250, None)
        self.assertEqual(progress["task"], "FRI-Franka-Lift-NearTable-Progress")
        self.assertEqual(progress["reset_protocol"], "table_near_grasp_open")
        self.assertEqual(progress["reward_variant"], "near_table_progress_v2")
        self.assertNotIn("--checkpoint", progress["argv"])

    def test_invalid_stage_transitions(self):
        with self.assertRaises(ValueError):
            build_stage_plan(Path("root"), "table", 16, 0, 2, None)
        with self.assertRaises(ValueError):
            build_stage_plan(Path("root"), "table-approach", 16, 0, 2, None)
        with self.assertRaises(ValueError):
            build_stage_plan(Path("root"), "pregrasp", 16, 0, 2, Path("model.pt"))
        with self.assertRaises(ValueError):
            build_stage_plan(Path("root"), "pregrasp", 0, 0, 2, None)

    def test_dry_run_does_not_create_a_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "planned"
            output = StringIO()
            with redirect_stdout(output):
                main(["--isaaclab-root", directory, "--stage", "pregrasp", "--run-dir", str(run_dir)])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["stage"], "pregrasp")
            self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
