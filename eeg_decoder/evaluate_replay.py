"""Evaluate saved decisions against EEGBCI run 4/8/12 annotation intervals.

Labels are loaded only in this post-processing module, after decision traces
have been generated. Nothing here is imported by the decoder or command gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping

try:
    from fri_bci_sim.replay_evaluation import AnnotatedTrial, LabelInterval, evaluate_decision_trace
except ModuleNotFoundError as exc:
    if exc.name != "fri_bci_sim":
        raise
    # Allow the documented project-root CLI without requiring an editable
    # install of the separate simulation package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "src"))
    from fri_bci_sim.replay_evaluation import AnnotatedTrial, LabelInterval, evaluate_decision_trace


EEGBCI_RUNS = {4, 8, 12}
EEGBCI_LABELS = {"T0": "rest", "T1": "imagery", "T2": "imagery"}
_RECORDING_RE = re.compile(r"S(?P<subject>\d{3})R(?P<run>\d{2})\Z")


def parse_eegbci_recording_name(edf_path: Path) -> tuple[str, int, int]:
    """Return (source stem, subject number, run) or reject unsupported EDFs."""
    source = Path(edf_path).stem
    match = _RECORDING_RE.fullmatch(source)
    if match is None:
        raise ValueError("EDF filename must use EEGBCI form S###R##.edf")
    run = int(match.group("run"))
    if run not in EEGBCI_RUNS:
        raise ValueError("EEGBCI annotation mapping is supported only for runs 4, 8, and 12")
    return source, int(match.group("subject")), run


def annotations_to_trial(
    source: str,
    annotations: Iterable[Mapping],
    recording_duration_s: float,
    *,
    horizon_s: float | None = None,
) -> tuple[AnnotatedTrial, dict]:
    """Map T0/T1/T2 annotations to bounded rest/imagery intervals.

    Positive annotation durations are clipped at the next annotation onset.
    A zero duration means the annotation state continues to the next onset,
    or recording end for the final annotation. All interval ends are clipped
    to recording end and the optional explicit evaluation horizon. Annotation
    onsets at or beyond that effective horizon contribute no interval.
    Adjacent intervals with the same mapped label are merged, so T1-to-T2 is
    one continuous imagery bout. Unknown labels, negative/non-finite durations,
    and unordered or duplicate annotation onsets are rejected.
    """
    recording_duration = _finite_nonnegative(recording_duration_s, "recording duration")
    if recording_duration <= 0:
        raise ValueError("Recording duration must be positive")
    if horizon_s is None:
        requested_horizon = recording_duration
    else:
        requested_horizon = _finite_nonnegative(horizon_s, "evaluation horizon")
        if requested_horizon <= 0:
            raise ValueError("Evaluation horizon must be positive")
    effective_horizon = min(recording_duration, requested_horizon)

    records = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"Annotation {index} must be a mapping")
        code = annotation.get("description")
        if code not in EEGBCI_LABELS:
            raise ValueError(f"Unsupported EEGBCI event {code!r}; expected T0, T1, or T2")
        onset = _finite_nonnegative(annotation.get("onset"), f"annotation {index} onset")
        duration = _finite_nonnegative(annotation.get("duration"), f"annotation {index} duration")
        records.append((onset, duration, EEGBCI_LABELS[code]))

    if not records:
        raise ValueError("EDF contains no EEGBCI state annotations")
    if any(a[0] >= b[0] for a, b in zip(records, records[1:])):
        raise ValueError("Annotation onsets must be strictly increasing")

    intervals: list[LabelInterval] = []
    clipped_away = 0
    for index, (onset, duration, label) in enumerate(records):
        next_onset = records[index + 1][0] if index + 1 < len(records) else recording_duration
        natural_end = min(onset + duration, next_onset) if duration > 0 else next_onset
        end = min(natural_end, recording_duration, effective_horizon)
        start = min(onset, effective_horizon)
        if onset >= effective_horizon or end <= start:
            clipped_away += 1
            continue
        if intervals and intervals[-1].label == label and math.isclose(intervals[-1].end_s, start, abs_tol=1e-9):
            previous = intervals[-1]
            intervals[-1] = LabelInterval(previous.start_s, end, label)
        else:
            intervals.append(LabelInterval(start, end, label))

    if not intervals:
        raise ValueError("No labeled intervals fall within the effective evaluation horizon")
    # Unannotated gaps remain gaps; trial horizon is still the declared bounded clock.
    trial = AnnotatedTrial(source, source, effective_horizon, tuple(intervals))
    return trial, {
        "recording_duration_s": recording_duration,
        "requested_horizon_s": requested_horizon,
        "effective_horizon_s": effective_horizon,
        "clipped_away_events": clipped_away,
        "duration_semantics": (
            "positive duration ends at the earlier of onset+duration and next onset; "
            "zero duration continues to next onset (or recording end for final event); "
            "ends are clipped to recording end/effective horizon; onsets at or after "
            "effective horizon are omitted"
        ),
        "label_mapping": dict(EEGBCI_LABELS),
    }


def load_edf_trial(edf_path: Path, horizon_s: float | None = None) -> tuple[AnnotatedTrial, dict]:
    """Read only EDF metadata/annotations; do not touch EEG samples or fit models."""
    source, subject, run = parse_eegbci_recording_name(edf_path)
    import mne

    raw = mne.io.read_raw_edf(Path(edf_path), preload=False, verbose="ERROR")
    try:
        duration_s = raw.n_times / float(raw.info["sfreq"])
        annotations = [
            {"onset": onset, "duration": duration, "description": description}
            for onset, duration, description in zip(
                raw.annotations.onset, raw.annotations.duration, raw.annotations.description
            )
        ]
    finally:
        raw.close()
    trial, details = annotations_to_trial(source, annotations, duration_s, horizon_s=horizon_s)
    details.update({"edf": str(Path(edf_path).resolve()), "source": source,
                    "subject": subject, "run": run})
    return trial, details


def evaluate_files(decisions_path: Path, edf_path: Path, output_path: Path,
                   horizon_s: float | None = None) -> dict:
    """Evaluate a pre-generated trace and create a new JSON output exclusively."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial, edf_details = load_edf_trial(edf_path, horizon_s=horizon_s)
    result = evaluate_decision_trace(Path(decisions_path), [trial])
    report = {
        "protocol": "EEGBCI annotated replay evaluation; post-processing only",
        "scope": (
            "scores pre-generated decisions against labels; no decoder fitting or tuning, "
            "no labels exposed during decoding, and no physical/live result"
        ),
        "decisions_file": str(Path(decisions_path).resolve()),
        "annotation_source": edf_details,
        "evaluation": result,
    }
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate saved EEGBCI run 4/8/12 decisions against EDF annotations"
    )
    parser.add_argument("--decisions", required=True, type=Path, help="pre-generated comparison JSONL")
    parser.add_argument("--edf", required=True, type=Path, help="annotated EEGBCI S###R04/08/12 EDF")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report path (must not exist)")
    parser.add_argument("--horizon-s", type=float, help="optional explicit upper bound in recording seconds")
    args = parser.parse_args(argv)
    report = evaluate_files(args.decisions, args.edf, args.output, args.horizon_s)
    print(json.dumps({"output": str(args.output.resolve()), "trials": report["evaluation"]["trials"]}))
    return 0


def _finite_nonnegative(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
