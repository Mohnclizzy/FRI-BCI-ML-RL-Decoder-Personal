from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import butter, sosfilt

from fri_bci_sim.commands import CommandGate, Estimate, Skill
from fri_bci_sim.comparison import compare_trace
from fri_bci_sim.streaming import CausalDecoder


CONFIG = {"channels": ["C3", "Cz", "C4"], "sample_rate_hz": 160,
          "window_seconds": 2, "bandpass_hz": [8, 30], "filter_order": 4}


class RecordingModel:
    classes_ = [0, 1]

    def __init__(self):
        self.windows = []

    def predict_proba(self, window):
        self.windows.append(window[0].copy())
        return np.array([[0.1, 0.9]])


class StreamingTest(unittest.TestCase):
    def test_filter_matches_full_run_and_windows_ignore_chunk_boundaries(self):
        samples = np.random.default_rng(4).normal(size=(3, 801)) * 1e-6
        referenced = samples - samples.mean(axis=0, keepdims=True)
        expected = sosfilt(butter(4, [8, 30], fs=160, btype="bandpass", output="sos"), referenced, axis=-1)
        for chunk_size in (17, 80, 801):
            model = RecordingModel()
            decoder = CausalDecoder(model, CONFIG, "recording")
            events = []
            for start in range(0, samples.shape[1], chunk_size):
                events.extend(decoder.push(samples[:, start:start+chunk_size]))
            ends = list(range(320, 802, 40))
            self.assertEqual(len(events), len(ends))
            for event, window, end in zip(events, model.windows, ends):
                np.testing.assert_allclose(window, expected[:, end-320:end], atol=1e-18)
                self.assertGreaterEqual(event["estimate"]["available_at_s"], end/160)
            self.assertLessEqual(decoder.buffer.shape[1], 320)

    def test_warmup_and_chunk_latency(self):
        decoder = CausalDecoder(RecordingModel(), CONFIG, "recording")
        self.assertEqual(decoder.push(np.ones((3, 319))), [])
        events = decoder.push(np.ones((3, 11)))
        self.assertEqual(events[0]["estimate"]["window_end_s"], 2)
        self.assertEqual(events[0]["estimate"]["available_at_s"], 330/160)

    def test_same_batch_timestamps_are_not_duplicate_evidence(self):
        gate = CommandGate(mode="raw")
        for end in (2, 2.25):
            request = gate.update(Estimate("r", "r", end, 2.5, 0.1, 0.9), 2.5)
            self.assertEqual(request.skill, Skill.GRASP_LIFT)

    def test_watchdog_expires_previously_accepted_command(self):
        gate = CommandGate(mode="raw")
        gate.update(Estimate("r", "r", 2, 2, 0.1, 0.9), 2)
        self.assertEqual(gate.tick(2.1).skill, Skill.GRASP_LIFT)
        self.assertEqual(gate.tick(3.1).skill, Skill.HOLD)
        self.assertEqual(gate.tick(1.9).skill, Skill.HOLD)

    def test_ablations_receive_identical_trace(self):
        rows = [Estimate("r", "r", 2+i*.25, 2+i*.25, 1-hand, hand)
                for i, hand in enumerate((0.6, 0.9, 0.9, 0.9, 0.1))]
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory)/"trace.jsonl", Path(directory)/"decisions.jsonl"
            path.write_text("\n".join(json.dumps({"estimate": asdict(row)}) for row in rows), encoding="utf-8")
            result = compare_trace(path, output)
            counts = result["conditions"]
            self.assertEqual([counts[k]["estimates"] for k in counts], [5, 5, 5])
            self.assertEqual([counts[k]["execute_estimates"] for k in counts], [4, 3, 1])
            for row in output.read_text().splitlines():
                self.assertEqual(len(json.loads(row)["decisions"]), 3)


if __name__ == "__main__":
    unittest.main()
