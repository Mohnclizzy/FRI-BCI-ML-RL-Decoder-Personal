"""Audit the R04 elapsed-time control on saved R08 replay endpoints.

From the repository root, with Python and NumPy:
    python analysis/audit_cue_clock.py

Pass --metadata-dir outputs/local_scratch_2026-11-11/cache/7d3de6fbf853d44a
to additionally verify the compact R04 evidence against the original cache.
The calculation reads timestamps and cue annotations. It needs no model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "results" / "project_review_2026-11-18"
SCORES = ROOT / "results" / "bounded_eeg_robot_2026-11-14" / "scores"


def labels_at(intervals, times):
    labels = np.full(len(times), -1, dtype=int)
    for start, stop, label in intervals:
        labels[(start <= times) & (times < stop)] = label
    return labels


def score_subject(subject, entry, fit_intervals, metadata_dir):
    dev_intervals = json.loads((SCORES / f"{subject}_annotations.json").read_bytes())
    if metadata_dir is not None:
        fit_bytes = (metadata_dir / f"{subject}R04.json").read_bytes()
        fit = json.loads(fit_bytes)
        dev = json.loads((metadata_dir / f"{subject}R08.json").read_bytes())
        if not (hashlib.sha256(fit_bytes).hexdigest() == entry["fit_metadata_sha256"]
                and (fit["subject"], fit["run"]) == (int(subject[1:]), 4)
                and (dev["subject"], dev["run"]) == (int(subject[1:]), 8)
                and fit["source_edf_sha256"] == entry["fit_edf_sha256"]
                and dev["source_edf_sha256"] == entry["dev_edf_sha256"]
                and fit["intervals"] == fit_intervals and dev["intervals"] == dev_intervals):
            raise ValueError(f"{subject} annotation provenance mismatch")
    if entry["fit_edf_sha256"] == entry["dev_edf_sha256"]:
        raise ValueError(f"{subject} fit and development share an EDF digest")

    with np.load(SCORES / f"eegnet_seed42_{subject}.npz", allow_pickle=False) as scores:
        times = scores["times"]
    truth = labels_at(dev_intervals, times)
    replay = labels_at(fit_intervals, times)
    # Half-open 4.1-second periods beginning with REST at time zero.
    period = (np.rint(times * 1000).astype(int) // 4100) % 2
    annotated = truth >= 0

    def balanced_accuracy(pred):
        return mean(float(np.mean(pred[truth == label] == label)) for label in (0, 1))

    return {
        "subject": subject,
        "interval_lists_identical": fit_intervals == dev_intervals,
        "endpoints": len(times),
        "rest": int(sum(truth == 0)),
        "imagery": int(sum(truth == 1)),
        "unannotated_r08": int(sum(~annotated)),
        "annotated_both": int(sum(annotated & (replay >= 0))),
        "elapsed_time_correct": int(sum(annotated & (replay == truth))),
        "elapsed_time_balanced_accuracy": balanced_accuracy(replay),
        "periodic_timer_balanced_accuracy": balanced_accuracy(period),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON destination")
    args = parser.parse_args()
    fit = json.loads((REVIEW / "r04_cue_schedules.json").read_bytes())
    people = []
    for i in range(1, 16):
        subject = f"S{i:03d}"
        entry = fit["subjects"][subject]
        people.append(score_subject(subject, entry, fit["schedules"][entry["schedule"]], args.metadata_dir))
    report = {
        "task": "Predict the annotated R08 REST/imagery label at each 4 Hz prediction time using elapsed time alone.",
        "scoring": "Intervals are half-open. Unannotated R08 endpoints are excluded. A missing R04 label counts as an incorrect prediction at an annotated R08 endpoint. Balanced accuracy averages class recall within person, then weights participants equally.",
        "elapsed_time_control": "Use the participant's R04 cue schedule with the R08 recording clock. This diagnostic control has access to the prior run's cue timings.",
        "periodic_timer": "Descriptive control chosen during review. Alternate REST and imagery every 4.1 seconds from recording onset.",
        "same_schedule_participants": sum(p["interval_lists_identical"] for p in people),
        "totals": {k: sum(p[k] for p in people) for k in
                   ("endpoints", "rest", "imagery", "unannotated_r08", "annotated_both", "elapsed_time_correct")},
        "mean_elapsed_time_balanced_accuracy": mean(p["elapsed_time_balanced_accuracy"] for p in people),
        "mean_periodic_timer_balanced_accuracy": mean(p["periodic_timer_balanced_accuracy"] for p in people),
        "participants": people,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_bytes(payload.encode("utf-8"))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
