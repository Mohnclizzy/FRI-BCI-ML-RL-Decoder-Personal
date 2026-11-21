"""Check the proposed one-request-per-slot budget against cached R08 annotations.

Run from the repository root with --cache pointing to the directory containing
S001R08.json through S015R08.json and the matching recording NPZ files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from numpy.lib import format as npy


def request_capacity(intervals, duration_s, slot_s=20):
    """Maximum slot/episode matches ignoring score, re-arm and actuation delays."""
    if not math.isfinite(duration_s) or duration_s <= 0 or slot_s <= 0:
        raise ValueError("Require positive recording and slot durations")
    slot_count = math.floor(duration_s / slot_s)
    prefix_end = slot_count * slot_s
    imagery = []
    previous_end = 0
    for start, end, label in intervals:
        if not (0 <= start < end <= duration_s) or start < previous_end - 1e-9:
            raise ValueError("Expected chronological disjoint intervals inside the recording")
        if label not in (0, 1):
            raise ValueError("Expected binary instruction labels")
        if label == 1:
            imagery.append((start, end))
        previous_end = end

    # Both slots and annotated episodes have disjoint, ordered interiors.
    # Assign the earliest eligible episode to each slot. Choosing a later
    # episode cannot improve any remaining slot's access to the earlier one.
    episode = 0
    matched = 0
    for slot in range(slot_count):
        start, end = slot * slot_s, (slot + 1) * slot_s
        while episode < len(imagery) and imagery[episode][1] <= start:
            episode += 1
        if episode < len(imagery) and imagery[episode][0] < end:
            matched += 1
            episode += 1

    overlap_count = sum(start < prefix_end for start, _ in imagery)
    return {
        "duration_s": duration_s,
        "complete_slots": slot_count,
        "robot_horizon_end_s": prefix_end,
        "incomplete_tail_s": duration_s - prefix_end,
        "imagery_full_recording": len(imagery),
        "imagery_overlapping_robot_horizon": overlap_count,
        "max_distinct_matches": matched,
        "full_recording_ceiling": matched / len(imagery) if imagery else None,
        "robot_horizon_ceiling": matched / overlap_count if overlap_count else None,
    }


def read_signal_length(npz_path):
    # The last annotation can end before the recording ends. Use the signal
    # shape from the NPY header; the preflight never loads EEG samples.
    with ZipFile(npz_path) as arrays, arrays.open("signal.npy") as signal:
        version = npy.read_magic(signal)
        if version == (1, 0):
            shape, _, _ = npy.read_array_header_1_0(signal)
        elif version == (2, 0):
            shape, _, _ = npy.read_array_header_2_0(signal)
        else:
            raise ValueError(f"Unsupported signal NPY header {version} in {npz_path}")
    if len(shape) != 2:
        raise ValueError(f"Expected channel by sample signal in {npz_path}")
    return shape[1]


def audit_r08(cache):
    participants = []
    for subject in range(1, 16):
        metadata = cache / f"S{subject:03d}R08.json"
        raw = metadata.read_bytes()
        record = json.loads(raw)
        if record["subject"] != subject or record["run"] != 8:
            raise ValueError(f"Incorrect subject/run in {metadata}")
        length = read_signal_length(metadata.with_suffix(".npz"))
        duration_s = length / record["preprocessing"]["sample_rate_hz"]
        counts = request_capacity(record["intervals"], duration_s)
        if not counts["imagery_overlapping_robot_horizon"]:
            raise ValueError(f"No imagery overlaps complete slots in {metadata}")
        participants.append({"subject": subject, "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                             "signal_samples": length, **counts})
    totals = {key: sum(p[key] for p in participants) for key in (
        "complete_slots", "imagery_full_recording", "imagery_overlapping_robot_horizon",
        "max_distinct_matches")}
    return {
        "run": 8,
        "slot_s": 20,
        "proposed_min_absolute_episode_detection": 0.70,
        "assumptions": "One accepted request per complete slot starting at time zero, "
                       "one distinct imagery episode per request, half-open intervals. "
                       "This annotation-only upper bound allows requests anywhere in eligible "
                       "overlaps and ignores all timing, re-arm and motion constraints.",
        "denominators": "Full recording counts every imagery episode. Robot horizon counts "
                        "every imagery episode with positive overlap before the end of the "
                        "last complete slot, including episodes crossing that boundary.",
        "totals": totals,
        "mean_participant_full_recording_ceiling": mean(p["full_recording_ceiling"] for p in participants),
        "mean_participant_robot_horizon_ceiling": mean(p["robot_horizon_ceiling"] for p in participants),
        "participants": participants,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_r08(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
