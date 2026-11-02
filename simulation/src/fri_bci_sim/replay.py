"""Adapter for the existing offline EEG export. Each trial is a separate episode."""

import json
from pathlib import Path
from .commands import Estimate


def load_trial_estimates(path: Path) -> list[Estimate]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of decoder predictions")
    estimates, seen = [], set()
    for row in rows:
        trial_id = str(row["trial_id"])
        if trial_id in seen:
            raise ValueError(f"Duplicate trial: {trial_id}")
        seen.add(trial_id)
        probabilities = row["probabilities"]
        estimates.append(Estimate(
            source=f"physionet:S{int(row['subject']):03d}R{int(row['run']):02d}",
            episode=trial_id,
            window_end_s=float(row["window_end_seconds"]),
            available_at_s=float(row["available_at_recording_seconds"]),
            p_rest=float(probabilities["REST"]), p_hand=float(probabilities["HAND_IMAGERY"]),
        ))
    # Ground-truth labels and annotation classes are deliberately absent.
    return estimates
