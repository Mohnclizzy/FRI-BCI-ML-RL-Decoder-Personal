"""Collect one corrected 120 s EEG matched batch and its three controls.

This script never runs a simulator. It requires all 15 S006-S010/R12/gate
cases and 3 controls from a successful corrected batch, evaluates each telemetry
file independently, and writes a fresh report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ISAAC_OUT = ROOT / "outputs" / "isaac"
EDF_ROOT_DEFAULT = ROOT / "data" / "physionet" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
sys.path.insert(0, str(ROOT / "simulation" / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_decoder.evaluate_replay import load_edf_trial
from eeg_decoder.evaluate_robot_replay import evaluate_files
from fri_bci_sim.physical_evaluation import PhysicalThresholds

PEOPLE = [f"S{n:03d}" for n in range(6, 11)]
MODES = ("raw", "confidence", "temporal")
EEG_CASES = [f"{person}R12-{mode}" for person in PEOPLE for mode in MODES]
CONTROL_CASES = ("always-hold", "scripted-positive", "scripted-pause")
EXPECTED_CASES = set(EEG_CASES) | set(CONTROL_CASES)
TRIAL_CACHE = {}
MOTION_THRESHOLDS = PhysicalThresholds()


def read_summary(log_path: Path) -> dict:
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "physical_lift_pass" in row and "physics_steps" in row:
            return row
    raise ValueError(f"Missing simulator summary in {log_path}")


def validate_batch(run_dir: Path) -> dict[str, int]:
    manifest = run_dir / "manifest.txt"
    if not manifest.is_file():
        raise ValueError(f"Missing batch manifest: {manifest}")
    manifest_values = dict(line.split("=", 1) for line in manifest.read_text(encoding="utf-8").splitlines()
                           if "=" in line)
    if manifest_values.get("budget_s") != "120" or manifest_values.get("exploratory_post_hoc") != "true":
        raise ValueError("Collector accepts only the frozen exploratory 120 s corrected batch")
    results_path = run_dir / "results.tsv"
    if not results_path.is_file():
        raise ValueError(f"Missing batch results table: {results_path}")
    codes = {}
    with results_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name = row["condition"]
            if name in codes:
                raise ValueError(f"Duplicate batch result row: {name}")
            codes[name] = int(row["exit_code"])
    if set(codes) != EXPECTED_CASES:
        raise ValueError(f"Expected 18 corrected cases; missing={sorted(EXPECTED_CASES-set(codes))}, extra={sorted(set(codes)-EXPECTED_CASES)}")
    failed = [name for name, code in codes.items() if code != 0]
    if failed:
        raise ValueError(f"Batch contains unsuccessful cases: {failed}")
    return codes


def verify_decision_hash(run_dir: Path, decision_path: Path) -> str:
    expected = None
    for line in (run_dir / "manifest.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and Path(parts[1]).name == decision_path.name:
            expected = parts[0]
            break
    if expected is None:
        raise ValueError(f"Batch manifest does not record {decision_path.name}")
    actual = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Decision trace hash mismatch for {decision_path.name}: batch={expected}, local={actual}")
    return actual


def validate_initialization(case_dir: Path, summary: dict) -> None:
    path = case_dir / "initialization.jsonl"
    if not path.is_file():
        raise ValueError(f"Missing common pre-roll telemetry: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = int(summary.get("initialization_steps", 0))
    dt = float(summary["physics_dt_s"])
    duration = float(summary.get("initialization_duration_s", -1))
    if steps <= 0 or len(rows) != steps or abs(duration - 2.0) > dt + 1e-6:
        raise ValueError(f"{case_dir.name} lacks the required 2 s physical pre-roll")
    if any(row.get("row_kind") != "pre_roll" for row in rows):
        raise ValueError(f"Invalid pre-roll row schema in {path}")
    if abs(float(rows[-1]["initialization_time_s"]) - duration) > 1e-9:
        raise ValueError(f"Pre-roll timestamps do not cover its declared duration: {path}")
    target = summary.get("initial_hold_arm_target")
    if not isinstance(target, list) or not target:
        raise ValueError(f"Missing commanded pre-roll target in {case_dir.name} summary")
    for row in rows:
        row_target = row["joint_target"]
        if (len(row_target) != len(target)
                or max(abs(float(a) - float(b)) for a, b in zip(row_target, target)) > 1e-9):
            raise ValueError(f"Arm command changed during pre-roll: {path}")
        if abs(float(row["finger_target_width_m"]) - 0.04) > 1e-9:
            raise ValueError(f"Finger command was not open during pre-roll: {path}")


def executed_request_metrics(rows: list[dict], summary: dict, trial=None) -> dict:
    """Integrate actual requested skills across the full simulator horizon."""
    post = [row for row in rows if row.get("row_kind") == "post_step"]
    if len(post) != int(summary["physics_steps"]):
        raise ValueError("Corrected replay telemetry must include every physics step")
    horizon = float(summary["simulated_duration_s"])
    active_duration = switches = 0
    active_intervals = []
    prior = "hold"
    for row in post:
        start, end = float(row["interval_start_s"]), min(horizon, float(row["time_s"]))
        skill = row["requested_skill"]
        if skill != prior:
            switches += 1
            prior = skill
        if skill == "grasp_lift" and end > start:
            active_duration += end - start
            active_intervals.append((start, end))

    rest_duration = rest_active = 0.0
    if trial is not None:
        for interval in trial.intervals:
            if interval.label != "rest":
                continue
            start, end = max(0.0, interval.start_s), min(horizon, interval.end_s)
            if end <= start:
                continue
            rest_duration += end - start
            for active_start, active_end in active_intervals:
                rest_active += max(0.0, min(end, active_end) - max(start, active_start))
    return {
        "evaluation_horizon_s": horizon,
        "request_active_duration_s": active_duration,
        "request_active_fraction_of_horizon": active_duration / horizon if horizon else None,
        "request_switch_count": switches,
        "request_switches_per_minute": switches * 60.0 / horizon if horizon else None,
        "rest_annotation_duration_s": rest_duration if trial is not None else None,
        "rest_command_active_duration_s": rest_active if trial is not None else None,
        "rest_command_activation_fraction": rest_active / rest_duration if rest_duration else None,
    }


def physical_action_latencies(rows: list[dict], summary: dict) -> list[dict]:
    """Sampled motion response per active request; completion is unavailable, never delayed."""
    post = [row for row in rows if row.get("row_kind") == "post_step"]
    spans = []
    active_start, active_rows = None, []
    for row in post:
        start = float(row["interval_start_s"])
        if row["requested_skill"] == "grasp_lift":
            if active_start is None:
                active_start, active_rows = start, []
            active_rows.append(row)
        elif active_start is not None:
            spans.append((active_start, start, active_rows, False))
            active_start, active_rows = None, []
    horizon = float(summary["simulated_duration_s"])
    if active_start is not None:
        spans.append((active_start, horizon, active_rows, True))

    row_index = {id(row): index for index, row in enumerate(rows)}
    results = []
    for start, end, span_rows, censored in spans:
        if span_rows and span_rows[0].get("executor_state") == "complete":
            results.append({"request_time_s": start, "request_end_s": end,
                            "response_latency_s": None,
                            "status": "unavailable_after_absorbing_completion",
                            "right_censored": False})
            continue
        response = None
        completed = False
        for row in span_rows:
            if row.get("executor_state") == "complete":
                completed = True
                break
            idx = row_index[id(row)]
            if idx == 0:
                continue
            previous = rows[idx - 1]
            dt = float(row["time_s"]) - float(previous["time_s"])
            if dt <= 0 or dt > MOTION_THRESHOLDS.max_sample_gap_s:
                continue
            joint_speed = max(abs(float(a) - float(b)) for a, b in zip(row["joint_pos"], previous["joint_pos"])) / dt
            hand_speed = math.dist(row["hand_pose_w"][:3], previous["hand_pose_w"][:3]) / dt
            if (joint_speed > MOTION_THRESHOLDS.joint_speed_rad_s
                    or hand_speed > MOTION_THRESHOLDS.hand_speed_m_s):
                response = float(row["time_s"]) - start
                break
        if response is not None:
            status = "observed_motion_upper_bound"
        elif completed:
            status = "unavailable_after_absorbing_completion"
        elif censored:
            status = "not_responding_within_horizon"
        else:
            status = "no_motion_before_hold"
        results.append({"request_time_s": start, "request_end_s": end,
                        "response_latency_s": response, "status": status,
                        "right_censored": censored})
    return results


def check_duration(summary: dict, requested_s: float, case_name: str) -> None:
    dt = float(summary["physics_dt_s"])
    actual = float(summary["simulated_duration_s"])
    if (float(summary.get("requested_duration_s", -1)) != requested_s
            or int(summary["physics_steps"]) <= 0 or abs(actual - requested_s) > dt + 1e-6):
        raise ValueError(f"{case_name} duration {actual}s does not match {requested_s}s within one physics step")


def collect_case(run_dir: Path, output_dir: Path, edf_root: Path,
                 case_name: str) -> dict:
    case_dir = run_dir / case_name
    telemetry = case_dir / "telemetry.jsonl"
    log = case_dir / "run.log"
    if not telemetry.is_file() or not log.is_file():
        raise ValueError(f"Missing telemetry or run log for {case_name}")
    summary = read_summary(log)
    is_control = case_name in CONTROL_CASES
    expected_condition = case_name if is_control else case_name.split("R12-", 1)[1]
    if summary.get("condition") != expected_condition:
        raise ValueError(f"{case_name} log reports condition {summary.get('condition')!r}")
    requested = {"always-hold": 24.0, "scripted-positive": 24.0,
                 "scripted-pause": 34.0}.get(case_name, 120.0)
    check_duration(summary, requested, case_name)
    validate_initialization(case_dir, summary)
    report_path = output_dir / "metrics" / f"{case_name}.json"
    raw_rows = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines() if line.strip()]
    decision_hash = None
    if is_control:
        report = evaluate_files(telemetry, report_path, horizon_s=requested)
        person, mode = "control", case_name
        imagery = None
        annotations = None
        trial = None
    else:
        person, mode = case_name.split("R12-", 1)
        edf = edf_root / person / f"{person}R12.edf"
        decisions = ISAAC_OUT / f"{person}R12-decisions.jsonl"
        if not edf.is_file():
            raise FileNotFoundError(f"Missing expected EEGBCI EDF: {edf}")
        if not decisions.is_file():
            raise FileNotFoundError(f"Missing saved decision trace: {decisions}")
        decision_hash = verify_decision_hash(run_dir, decisions)
        if person not in TRIAL_CACHE:
            TRIAL_CACHE[person] = load_edf_trial(edf, horizon_s=120.0)
        trial, annotations = TRIAL_CACHE[person]
        report = evaluate_files(telemetry, report_path, edf=edf, decisions=decisions,
                                condition=mode, horizon_s=120.0)
        imagery = report["command_evaluation"]["conditions"][mode]["per_trial"][0]

    physical = report["physical"]
    physical_lift = (physical["sampled_physical_lift_pass"] if "sampled_physical_lift_pass" in physical
                     else summary["physical_lift_pass"])
    if case_name == "always-hold":
        checks = [not summary["physical_lift_pass"],
                  summary["max_hand_displacement_m"] <= 0.02,
                  summary["max_joint_displacement_rad"] <= 0.1,
                  summary["max_hold_target_change_rad"] <= 1e-6,
                  summary["max_hold_finger_target_change_m"] <= 1e-6]
        if not all(checks):
            raise ValueError("Corrected always-HOLD control failed its frozen checks")
    elif case_name in ("scripted-positive", "scripted-pause"):
        checks = [summary["physical_lift_pass"], summary["skill_complete"],
                  not summary["dropped_after_acquisition"],
                  summary["max_hold_target_change_rad"] <= 1e-6,
                  summary["max_hold_finger_target_change_m"] <= 1e-6]
        if not all(checks):
            raise ValueError(f"Corrected positive control failed its frozen checks: {case_name}")
    rest_motion = physical["rest_motion"]
    dispatch = report["request_to_action"]
    request_metrics = executed_request_metrics(raw_rows, summary, trial)
    action_latency_events = physical_action_latencies(raw_rows, summary)
    measured_latencies = [event["response_latency_s"] for event in action_latency_events
                          if event["response_latency_s"] is not None]
    trajectory_complete_time = next((float(row["time_s"]) for row in raw_rows
                                     if row.get("row_kind") == "post_step" and row.get("skill_complete")), None)
    absorbing_complete_time = next((float(row["time_s"]) for row in raw_rows
                                    if row.get("row_kind") == "post_step"
                                    and row.get("executor_state") == "complete"), None)
    report["collector_metrics"] = {
        **request_metrics,
        "decision_trace_sha256": decision_hash,
        "first_sampled_physical_success_time_s": physical["first_sampled_success_time_s"],
        "trajectory_complete_time_s": trajectory_complete_time,
        "absorbing_executor_complete_time_s": absorbing_complete_time,
        "physical_action_latency_definition": (
            "first sampled interval exceeding the configured joint/hand speed threshold after an active request; "
            "sample time is an upper bound, and requests after absorbing completion are unavailable"
        ),
        "physical_action_latency_events": action_latency_events,
        "mean_observed_physical_action_latency_s": (
            sum(measured_latencies) / len(measured_latencies) if measured_latencies else None
        ),
        "unavailable_after_completion_count": sum(
            event["status"] == "unavailable_after_absorbing_completion" for event in action_latency_events
        ),
        "not_responding_count": sum(
            event["status"] in ("not_responding_within_horizon", "no_motion_before_hold")
            for event in action_latency_events
        ),
        "target_dispatch_metrics": dispatch,
    }
    # Keep the evaluator's annotation and physical sections intact while adding
    # full-horizon executed-request endpoints in a separate section.
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "case": case_name,
        "person": person,
        "condition": mode,
        "case_kind": "control" if is_control else "eeg",
        "requested_duration_s": requested,
        "simulated_duration_s": summary["simulated_duration_s"],
        "physics_steps": summary["physics_steps"],
        "physical_lift_pass": bool(physical_lift),
        "first_sampled_physical_success_time_s": physical["first_sampled_success_time_s"],
        "skill_complete": bool(summary["skill_complete"]),
        "skill_elapsed_s": float(summary["skill_elapsed_s"]),
        "trajectory_complete_time_s": trajectory_complete_time,
        "absorbing_executor_complete_time_s": absorbing_complete_time,
        "dropped_after_acquisition": bool(summary["dropped_after_acquisition"]),
        "max_cube_lift_m": physical["max_cube_lift_m"],
        "request_active_duration_s": request_metrics["request_active_duration_s"],
        "request_active_fraction_of_horizon": request_metrics["request_active_fraction_of_horizon"],
        "request_switch_count": request_metrics["request_switch_count"],
        "request_switches_per_minute": request_metrics["request_switches_per_minute"],
        "rest_command_active_duration_s": request_metrics["rest_command_active_duration_s"],
        "rest_command_activation_fraction": request_metrics["rest_command_activation_fraction"],
        "rest_moving_fraction": None if rest_motion is None else rest_motion["moving_fraction"],
        "rest_moving_duration_s": None if rest_motion is None else rest_motion["moving_duration_s"],
        "imagery_bouts_detected": None if imagery is None else imagery["detected_bouts"],
        "imagery_bouts_missed": None if imagery is None else imagery["missed_bouts"],
        "imagery_bouts_censored": None if imagery is None else imagery["censored_bouts"],
        "mean_imagery_detection_delay_s": None if imagery is None else imagery["mean_detection_delay_s"],
        "mean_target_dispatch_delay_s": None if dispatch is None else dispatch["mean_target_change_delay_s"],
        "mean_physical_action_latency_s": report["collector_metrics"]["mean_observed_physical_action_latency_s"],
        "physical_action_latency_unavailable_after_completion": report["collector_metrics"]["unavailable_after_completion_count"],
        "physical_action_latency_not_responding": report["collector_metrics"]["not_responding_count"],
        "observed_interval_duration_s": physical["observed_interval_duration_s"],
        "unobserved_duration_s": physical["unobserved_duration_s"],
        "annotation_horizon_s": None if annotations is None else annotations["effective_horizon_s"],
        "metric_json": str(report_path.resolve()),
    }


def write_summary(rows: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ["case", "physical_lift_pass", "first_sampled_physical_success_time_s",
               "skill_complete", "trajectory_complete_time_s", "request_active_duration_s",
               "request_switch_count", "request_switches_per_minute",
               "rest_command_activation_fraction", "rest_moving_fraction", "imagery_bouts_missed",
               "mean_imagery_detection_delay_s", "mean_target_dispatch_delay_s",
               "mean_physical_action_latency_s", "physical_action_latency_unavailable_after_completion",
               "physical_action_latency_not_responding"]
    md = ["# Corrected matched batch summary", "",
          "Exploratory 120 s S006–S010/R12 comparison; the three controls use 24/24/34 s. ",
          "Command annotations and physical telemetry metrics are kept in separate columns and per-case JSON reports.", "",
          "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append("NA" if value is None else str(value))
        md.append("| " + " | ".join(vals) + " |")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def make_plot(rows: list[dict], path: Path) -> None:
    eeg_rows = [row for row in rows if row["case_kind"] == "eeg"]
    controls = [row for row in rows if row["case_kind"] == "control"]
    colors = {"raw": "#4863a0", "confidence": "#df8b36", "temporal": "#429b76"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    axes = axes.ravel()
    modes = list(MODES)

    # Completion/lift counts include the separately identified positive controls.
    names = modes + [row["condition"] for row in controls]
    grouped = [[r for r in eeg_rows if r["condition"] == mode] for mode in modes]
    grouped += [[r] for r in controls]
    for metric, label, offset, color in [
        ("physical_lift_pass", "physical lift", -.12, "#3b8c6e"),
        ("skill_complete", "trajectory complete", .12, "#5867a8"),
    ]:
        means = [100 * sum(bool(r[metric]) for r in group) / len(group) for group in grouped]
        axes[0].bar([i + offset for i in range(len(names))], means, width=.22,
                    label=label, color=color, alpha=.9)
    axes[0].set_ylabel("Cases meeting criterion (%)")
    axes[0].set_title("Physical lift and completion")
    axes[0].set_xticks(range(len(names)), ["Raw", "Confidence", "Temporal", "HOLD", "Positive", "Pause"],
                       rotation=18, ha="right")
    axes[0].legend(frameon=False, fontsize=8)

    for i, mode in enumerate(modes):
        mode_rows = [r for r in eeg_rows if r["condition"] == mode]
        for metric, marker, shift, label in [
            ("rest_moving_fraction", "o", -.025, "Measured rest motion"),
            ("rest_command_activation_fraction", "s", .025, "Active rest commands"),
        ]:
            values = [r[metric] * 100 for r in mode_rows if r[metric] is not None]
            jitter = [i + shift + (j - (len(values) - 1) / 2) * .07 for j in range(len(values))]
            axes[1].scatter(jitter, values, marker=marker, color=colors[mode], edgecolor="white",
                            label=label if i == 0 else None, zorder=3)
            if values:
                axes[1].plot([i - .2 + shift, i + .2 + shift],
                             [sum(values) / len(values)] * 2, color=colors[mode], linewidth=2)
    axes[1].set_xticks(range(3), ["Raw", "Confidence", "Temporal"])
    axes[1].set_ylabel("Fraction of annotated rest (%)")
    axes[1].set_title("Rest commands and measured rest motion")
    axes[1].legend(frameon=False, fontsize=7)

    for i, mode in enumerate(modes):
        values = [r["imagery_bouts_missed"] for r in eeg_rows
                  if r["condition"] == mode and r["imagery_bouts_missed"] is not None]
        jitter = [i + (j - (len(values) - 1) / 2) * .07 for j in range(len(values))]
        axes[2].scatter(jitter, values, color=colors[mode], edgecolor="white", zorder=3)
        if values:
            axes[2].plot([i - .2, i + .2], [sum(values) / len(values)] * 2, color=colors[mode], linewidth=2)
    axes[2].set_xticks(range(3), ["Raw", "Confidence", "Temporal"])
    axes[2].set_ylabel("Missed imagery bouts (per recording)")
    axes[2].set_title("Annotation-based misses")

    for i, mode in enumerate(modes):
        mode_rows = [r for r in eeg_rows if r["condition"] == mode]
        for metric, marker, shift, label in [
            ("mean_target_dispatch_delay_s", "o", -.025, "Target dispatch"),
            ("mean_physical_action_latency_s", "s", .025, "Measured motion"),
        ]:
            values = [r[metric] for r in mode_rows if r[metric] is not None]
            jitter = [i + shift + (j - (len(values) - 1) / 2) * .07 for j in range(len(values))]
            axes[3].scatter(jitter, values, marker=marker, color=colors[mode], edgecolor="white",
                            label=label if i == 0 else None, zorder=3)
            if values:
                axes[3].plot([i - .2 + shift, i + .2 + shift],
                             [sum(values) / len(values)] * 2, color=colors[mode], linewidth=2)
    axes[3].set_xticks(range(3), ["Raw", "Confidence", "Temporal"])
    axes[3].set_ylabel("Mean latency from active request (s)")
    axes[3].set_title("Target dispatch vs physical motion onset")
    axes[3].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("Corrected exploratory matched replay · S006–S010 / R12 · 120 s", fontsize=13)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="completed corrected 120 s batch directory (not the historical 24 s run)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for 18 individual reports, CSV/Markdown and plot")
    parser.add_argument("--edf-root", type=Path, default=EDF_ROOT_DEFAULT)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"Output directory already exists; choose a fresh path: {output_dir}")
    validate_batch(run_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "metrics").mkdir()
    rows = [collect_case(run_dir, output_dir, args.edf_root.resolve(), case)
            for case in (*CONTROL_CASES, *EEG_CASES)]
    write_summary(rows, output_dir)
    make_plot(rows, output_dir / "matched_results.png")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "collector_manifest.json").write_text(json.dumps({
        "protocol": "corrected exploratory matched batch collector v1",
        "run_dir": str(run_dir),
        "run_manifest_sha256": digest(run_dir / "manifest.txt"),
        "results_tsv_sha256": digest(run_dir / "results.tsv"),
        "case_count": len(rows),
        "edf_root": str(args.edf_root.resolve()),
        "horizon_s": 120,
        "scope": "No stale 24 s physical outcomes accepted; annotation metrics remain separate from simulator-state metrics.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "cases": len(rows),
                      "metrics": str((output_dir / "metrics").resolve()),
                      "summary_csv": str((output_dir / "summary.csv").resolve()),
                      "plot": str((output_dir / "matched_results.png").resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
