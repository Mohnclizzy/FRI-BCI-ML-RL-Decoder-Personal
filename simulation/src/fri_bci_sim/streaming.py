"""Causal full-recording replay for the existing nine-channel CSP/LDA artifact.

Annotations are never read. SOS state persists across chunks, and estimates
become available only after the whole incoming chunk containing their window.
This is logical-time recorded replay, not live acquisition or user feedback.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import butter, sosfilt

from .commands import Estimate


class CausalDecoder:
    def __init__(self, model, config, source, hop_s=0.25, decode_delay_s=0.0):
        self.model, self.config, self.source = model, config, source
        self.fs = float(config["sample_rate_hz"])
        self.window = int(round(float(config["window_seconds"]) * self.fs))
        self.hop = int(round(hop_s * self.fs))
        if not np.isfinite(hop_s) or not np.isfinite(decode_delay_s) or hop_s <= 0 or decode_delay_s < 0:
            raise ValueError("Invalid replay timing")
        if self.hop < 1 or self.window < 1 or self.fs <= 0:
            raise ValueError("Invalid sample/window configuration")
        if not np.isclose(self.hop / self.fs, hop_s):
            raise ValueError("Hop must be an integer number of samples")
        if list(model.classes_) != [0, 1]:
            raise ValueError("Require the existing REST=0 / HAND_IMAGERY=1 artifact")
        self.decode_delay = decode_delay_s
        self.channels = len(config["channels"])
        self.sos = butter(config["filter_order"], config["bandpass_hz"],
                          btype="bandpass", fs=self.fs, output="sos")
        self.zi = np.zeros((len(self.sos), self.channels, 2))
        self.buffer = np.empty((self.channels, 0))
        self.total, self.start, self.next_end = 0, 0, self.window

    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != self.channels or samples.shape[1] == 0:
            raise ValueError("Require a nonempty selected-channel by time chunk")
        if not np.isfinite(samples).all():
            raise ValueError("EEG contains non-finite samples; abort rather than silently skip")
        referenced = samples - samples.mean(axis=0, keepdims=True)
        filtered, self.zi = sosfilt(self.sos, referenced, axis=-1, zi=self.zi)
        self.buffer = np.concatenate((self.buffer, filtered), axis=1)
        self.total += samples.shape[1]
        output = []
        while self.next_end <= self.total:
            window = self.buffer[:, self.next_end-self.window-self.start:self.next_end-self.start]
            began = time.perf_counter()
            probabilities = self.model.predict_proba(window[None])[0]
            compute_s = time.perf_counter() - began
            estimate = Estimate(self.source, self.source, self.next_end / self.fs,
                                self.total / self.fs + self.decode_delay,
                                float(probabilities[0]), float(probabilities[1]))
            output.append({"estimate": asdict(estimate), "compute_seconds": compute_s})
            self.next_end += self.hop
        keep_from = max(0, self.total - self.window)
        self.buffer = self.buffer[:, keep_from-self.start:]
        self.start = keep_from
        return output


def decode_recording(artifact_path: Path, edf_path: Path, output: Path,
                     chunk_samples=16, hop_s=0.25, decode_delay_s=0.0):
    """Decode a local recording with a trusted local model; never fit on this file."""
    import joblib
    import mne
    from mne.datasets import eegbci

    if chunk_samples < 1:
        raise ValueError("Chunk length must be positive")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Choose new output paths to preserve earlier runs")
    artifact = joblib.load(artifact_path)
    if artifact["class_names"] != ["REST", "HAND_IMAGERY"]:
        raise ValueError("Unsupported artifact class meanings")
    config = artifact["config"]
    source = edf_path.stem
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    try:
        eegbci.standardize(raw)
        raw.pick(config["channels"])
        raw.reorder_channels(config["channels"])
        if raw.info["sfreq"] != config["sample_rate_hz"]:
            raise ValueError("Recording sample rate must match training")
        decoder = CausalDecoder(artifact["model"], config, source, hop_s, decode_delay_s)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("x", encoding="utf-8") as stream:
            for start in range(0, raw.n_times, chunk_samples):
                block = raw.get_data(start=start, stop=min(start+chunk_samples, raw.n_times))
                for event in decoder.push(block):
                    stream.write(json.dumps(event) + "\n")
                    count += 1
    finally:
        raw.close()
    manifest = {"recording": str(edf_path.resolve()), "recording_sha256": file_hash(edf_path),
                "artifact_sha256": file_hash(artifact_path), "source": source, "config": config,
                "events": count, "chunk_samples": chunk_samples, "hop_s": hop_s,
                "injected_decode_delay_s": decode_delay_s, "real_time": False,
                "annotations_used_by_decoder": False,
                "availability": "chunk arrival plus injected delay; measured compute reported separately"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return count


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
