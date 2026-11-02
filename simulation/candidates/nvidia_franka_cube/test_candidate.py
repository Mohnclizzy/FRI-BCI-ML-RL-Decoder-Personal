import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import prepare
from scoring import score


def row(t, rise=0.1, forces=(1.0, 1.0)):
    return {"time_s": t, "cube_height_m": 0.025 + rise, "finger_contact_n": forces}


class ScoringTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
