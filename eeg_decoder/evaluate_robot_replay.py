"""Score saved robot telemetry, then optionally join separate EEG annotations.

Usage from project root: python -m eeg_decoder.evaluate_robot_replay
    --telemetry PATH --edf S006R12.edf --decisions PATH --condition raw
    --horizon-s 60 --output NEW_REPORT.json
Omit EDF/decisions/condition for a purely physical control report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluate_replay import load_edf_trial, evaluate_decision_trace
from fri_bci_sim.executor import DecisionReplay
from fri_bci_sim.physical_evaluation import (
    PhysicalThresholds, evaluate_physical_rows, evaluate_request_target_delays,
)


def _executed_request_events(rows, decisions: Path, condition: str) -> list[dict]:
    """Validate per-step executor requests against a replay trace and collapse transitions."""
    replay = DecisionReplay(decisions, condition)
    executed_rows = [r for r in rows if r.get("row_kind") == "post_step"]
    if not executed_rows:
        raise ValueError("Decision-matched scoring requires per-step post_step telemetry")
    by_time = {}
    for row in executed_rows:
        if "interval_start_s" not in row:
            raise ValueError("Decision-matched per-step telemetry requires interval_start_s")
        time_s = float(row["interval_start_s"])
        expected = replay.at(time_s).skill.value
        if row.get("requested_skill") != expected:
            raise ValueError(
                "Telemetry requested_skill does not match the replayed decision trace "
                f"at {time_s} s: expected {expected}, got {row.get('requested_skill')}"
            )
        by_time[time_s] = row["requested_skill"]
    events = []
    prior_skill = None
    for time_s, skill in sorted(by_time.items()):
        if skill != prior_skill:
            events.append({"time_s": time_s, "skill": skill})
            prior_skill = skill
    return events


def evaluate_files(telemetry: Path, output: Path, *, edf: Path | None = None,
                   decisions: Path | None = None, condition: str | None = None,
                   horizon_s: float | None = None,
                   thresholds: PhysicalThresholds = PhysicalThresholds()) -> dict:
    if (decisions is None) != (condition is None):
        raise ValueError("Provide both decisions and condition, or neither")
    if decisions is not None and edf is None:
        raise ValueError("An EDF is required to validate the decision source and score imagery coverage")
    rows = [json.loads(line) for line in Path(telemetry).read_text(encoding="utf-8").splitlines() if line.strip()]
    trial, annotation_details = load_edf_trial(edf, horizon_s=horizon_s) if edf else (None, None)
    if horizon_s is not None and trial is None:
        if horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        rows = [r for r in rows if r["time_s"] <= horizon_s + 1e-9]
    physical = evaluate_physical_rows(rows, trial=trial, thresholds=thresholds)
    inputs = {"telemetry": telemetry}
    command = delay = None
    if edf:
        inputs["edf_annotations"] = edf
    if decisions:
        inputs["decisions"] = decisions
        command = evaluate_decision_trace(decisions, [trial], modes=[condition])
        # Dispatch events come from the validated commands actually presented to
        # the executor, so stale-input watchdog HOLD transitions are retained.
        events = _executed_request_events(rows, decisions, condition)
        delay_rows = [r for r in rows if r["time_s"] <= trial.horizon_s + 1e-9]
        delay = evaluate_request_target_delays(delay_rows, events, thresholds=thresholds)
    report = {
        "protocol": "offline physical robot replay evaluation v1",
        "time_alignment": "Simulator time zero equals recording time zero; no timestamp rescaling or label-dependent resets",
        "condition": condition,
        "inputs": {name: {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                   for name, path in inputs.items()},
        "annotation_source": annotation_details,
        "physical": physical,
        "command_evaluation": command,
        "request_to_action": delay,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edf", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--condition", choices=("raw", "confidence", "temporal", "crc_hysteresis"))
    parser.add_argument("--horizon-s", type=float)
    parser.add_argument("--max-sample-gap-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    result = evaluate_files(args.telemetry, args.output, edf=args.edf, decisions=args.decisions,
                            condition=args.condition, horizon_s=args.horizon_s,
                            thresholds=PhysicalThresholds(max_sample_gap_s=args.max_sample_gap_s))
    print(json.dumps({"output": str(args.output), "physical": result["physical"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
