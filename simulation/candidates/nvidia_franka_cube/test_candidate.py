import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import io
import json
from types import SimpleNamespace

import prepare
from run import Recorder
from scoring import score, reset_continuity


def row(t, rise=0.1, forces=(1.0, 1.0)):
    return {"time_s": t, "cube_height_m": 0.025 + rise, "finger_contact_n": forces}


class ScoringTests(unittest.TestCase):
    def test_incomplete_lift_is_observed_but_not_a_valid_pass(self):
        result = score([row(t / 10) for t in range(1, 8)], .025,
                       max_gap_s=.11, complete=False)
        self.assertTrue(result["sampled_lift_observed"])
        self.assertIsNone(result["physical_lift_pass"])

    def test_high_forces_flag_review_without_changing_lift_threshold(self):
        result = score([row(t / 10, forces=(150, 2)) for t in range(1, 8)], .025,
                       max_gap_s=.11, complete=True)
        self.assertTrue(result["physical_lift_pass"])
        self.assertIn("finger_force_at_or_above_100N", result["review_flags"])
        self.assertEqual(result["max_finger_contact_n"], [150, 2])

    def test_nonfinite_contacts_cannot_produce_success(self):
        result = score([row(t / 10, forces=(float("nan"), 2)) for t in range(1, 8)],
                       .025, max_gap_s=.11, complete=True)
        self.assertIsNone(result["physical_lift_pass"])
        self.assertEqual(result["contact_coverage"], 0)

    def test_elapsed_span_not_sample_count(self):
        short = [row(t / 10) for t in range(1, 6)]
        self.assertFalse(score(short, 0.025, max_gap_s=.11, complete=True)["physical_lift_pass"])
        self.assertTrue(score(short + [row(.6)], .025, max_gap_s=.11, complete=True)["physical_lift_pass"])

    def test_unilateral_contact_cannot_pass(self):
        result = score([row(t / 10, forces=(0, 5)) for t in range(1, 10)], .025,
                       max_gap_s=.11, complete=True)
        self.assertFalse(result["physical_lift_pass"])

    def test_contact_gap_breaks_sustain(self):
        result = score([row(t) for t in (.1, .2, .3, .8, .9, 1.0)], .025,
                       max_gap_s=.11, complete=True)
        self.assertFalse(result["physical_lift_pass"])

    def test_missing_contacts_are_unavailable(self):
        result = score([row(.1, forces=(None, None))], .025, max_gap_s=.02, complete=False)
        self.assertIsNone(result["physical_lift_pass"])
        self.assertEqual(result["status"], "incomplete")

    def test_drop_requires_acquisition_and_quarter_second(self):
        samples = [row(t / 10) for t in range(1, 7)]
        samples += [row(t / 20, rise=0) for t in range(13, 19)]
        result = score(samples, .025, max_gap_s=.11, complete=True)
        self.assertEqual(len(result["drop_detection_times_s"]), 1)

    def test_nonmonotonic_times_rejected(self):
        with self.assertRaises(ValueError):
            score([row(.1), row(.1)], .025, max_gap_s=.1, complete=True)


class PreservationTests(unittest.TestCase):
    def test_existing_work_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                prepare.prepare(root / "absent.zip", root)

    def test_bad_archive_hash_does_not_create_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            archive.write_bytes(b"wrong artifact")
            with self.assertRaises(ValueError):
                prepare.prepare(archive, root / "work")
            self.assertFalse((root / "work").exists())

    def test_path_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape", "bad")
            with patch.object(prepare, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()):
                with self.assertRaises(ValueError):
                    prepare.prepare(archive, root / "work")
            self.assertFalse((root / "escape").exists())


class ResetContinuityTests(unittest.TestCase):
    def test_recorder_stops_at_bad_first_update_and_preserves_error_evidence(self):
        stream = io.StringIO()
        env = SimpleNamespace(scene=SimpleNamespace(update=lambda **kwargs: None),
                              physics_dt=1 / 120, step_dt=1 / 60)
        recorder = Recorder(env, stream)
        initial = {"cube_pose_w": [.4, 0, .025], "hand_pose_w": [.4, 0, .5]}
        first = {"episode": 1, "cube_pose_w": [.82, 0, .022],
                 "hand_pose_w": [.48, 0, .1], "cube_height_m": .022,
                 "finger_contact_n": [0, 0]}
        recorder.snapshot = lambda: first
        recorder.active = {"initial": initial, "samples": [], "baseline": .025,
                           "interventions_at_start": dict(recorder.interventions)}
        recorder.stepping = True
        with self.assertRaisesRegex(RuntimeError, "Reset state lost"):
            env.scene.update(dt=1 / 120)
        self.assertEqual(len(recorder.active["samples"]), 1)
        self.assertEqual(json.loads(stream.getvalue())["record_type"], "sample")
        recorder.finish(False, {"reason": "reset_failure"})
        self.assertIsNone(recorder.results[0]["physical_lift_pass"])
        self.assertFalse(recorder.results[0]["measurement_valid"])

    def test_missing_contact_stops_on_first_observed_update(self):
        stream = io.StringIO()
        env = SimpleNamespace(scene=SimpleNamespace(update=lambda **kwargs: None),
                              physics_dt=1 / 120)
        recorder = Recorder(env, stream)
        initial = {"cube_pose_w": [.4, 0, .025], "hand_pose_w": [.4, 0, .5]}
        recorder.snapshot = lambda: dict(initial, finger_contact_n=[None, None])
        recorder.active = {"initial": initial, "samples": []}
        recorder.stepping = True
        with self.assertRaisesRegex(RuntimeError, "contact measurement"):
            env.scene.update(dt=1 / 120)
        self.assertEqual(len(recorder.active["samples"]), 1)

    def test_stale_physics_state_is_rejected(self):
        initial = {"cube_pose_w": [.4, 0, .025], "hand_pose_w": [.4, 0, .5]}
        first = {"cube_pose_w": [.82, 0, .022], "hand_pose_w": [.48, 0, .1]}
        self.assertFalse(reset_continuity(initial, first)["reset_continuity_pass"])

    def test_normal_first_physics_tick_is_accepted(self):
        initial = {"cube_pose_w": [.4, 0, .025], "hand_pose_w": [.4, 0, .5]}
        first = {"cube_pose_w": [.4, 0, .0246], "hand_pose_w": [.4, -.001, .498]}
        self.assertTrue(reset_continuity(initial, first)["reset_continuity_pass"])


if __name__ == "__main__":
    unittest.main()
