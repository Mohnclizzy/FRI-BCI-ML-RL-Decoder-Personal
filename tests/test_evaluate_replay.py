import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eeg_decoder.evaluate_replay import (
    annotations_to_trial,
    evaluate_files,
    parse_eegbci_recording_name,
)
from fri_bci_sim.replay_evaluation import AnnotatedTrial, LabelInterval, evaluate_decision_trace


class EvaluateReplayTest(unittest.TestCase):
    def test_only_eegbci_run_4_8_12_filenames_are_accepted(self):
        for run in (4, 8, 12):
            self.assertEqual(parse_eegbci_recording_name(Path(f"S007R{run:02d}.edf")),
                             (f"S007R{run:02d}", 7, run))
        for name in ("S007R03.edf", "S007R04_extra.edf", "subjectS007R04.edf", "S007R4.edf"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_eegbci_recording_name(Path(name))

    def test_mapping_merges_t1_t2_and_clips_at_horizon(self):
        annotations = [
            {"onset": 0, "duration": 0, "description": "T0"},
            {"onset": 2, "duration": 3, "description": "T1"},
            {"onset": 4, "duration": 0, "description": "T2"},
            {"onset": 6, "duration": 10, "description": "T0"},
            {"onset": 9, "duration": 0, "description": "T1"},
        ]
        trial, details = annotations_to_trial("S001R04", annotations, 10, horizon_s=8)
        self.assertEqual(trial.episode, "S001R04")
        self.assertEqual(trial.horizon_s, 8)
        self.assertEqual(trial.intervals, (
            LabelInterval(0, 2, "rest"),
            LabelInterval(2, 6, "imagery"),
            LabelInterval(6, 8, "rest"),
        ))
        self.assertEqual(details["clipped_away_events"], 1)
        self.assertEqual(details["requested_horizon_s"], 8)
        self.assertEqual(details["recording_duration_s"], 10)

    def test_rejects_unknown_events_bad_durations_and_unordered_annotations(self):
        invalid_sets = [
            [{"onset": 0, "duration": 1, "description": "BAD boundary"}],
            [{"onset": 0, "duration": -1, "description": "T0"}],
            [{"onset": 0, "duration": float("nan"), "description": "T0"}],
            [{"onset": 2, "duration": 0, "description": "T0"},
             {"onset": 2, "duration": 0, "description": "T1"}],
        ]
        for annotations in invalid_sets:
            with self.subTest(annotations=annotations), self.assertRaises(ValueError):
                annotations_to_trial("S001R04", annotations, 10)

    def test_trace_source_and_episode_must_match_annotation_source(self):
        trial = AnnotatedTrial("S001R04", "S001R04", 3, (LabelInterval(0, 3, "rest"),))
        row = {"estimate": {"source": "S002R04", "episode": "S002R04",
                            "window_end_s": 1, "available_at_s": 1},
               "decisions": {"raw": {"skill": "hold", "issued_at_s": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no matching annotated trial"):
                evaluate_decision_trace(path, [trial])

    def test_report_output_is_created_exclusively(self):
        trial = AnnotatedTrial("S001R04", "S001R04", 3, (LabelInterval(0, 3, "rest"),))
        details = {"source": "S001R04", "effective_horizon_s": 3}
        row = {"estimate": {"source": "S001R04", "episode": "S001R04",
                            "window_end_s": 1, "available_at_s": 1},
               "decisions": {"raw": {"skill": "hold", "issued_at_s": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decisions, edf, output = root / "decisions.jsonl", root / "S001R04.edf", root / "report.json"
            decisions.write_text(json.dumps(row), encoding="utf-8")
            with patch("eeg_decoder.evaluate_replay.load_edf_trial", return_value=(trial, details)):
                report = evaluate_files(decisions, edf, output)
                self.assertEqual(report["annotation_source"], details)
                self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["evaluation"]["conditions"]["raw"])
                with self.assertRaises(FileExistsError):
                    evaluate_files(decisions, edf, output)


if __name__ == "__main__":
    unittest.main()
