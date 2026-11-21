"""Audit fixed-open physics traces and compare aligned baseline windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def vector_distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a[:3], b[:3])))


def analyze(path: Path, *, expected_targets: bool, max_steps: int | None = None):
    rows = read_rows(path)
    start = next(row for row in rows if row["record_type"] == "episode_start")
    samples = [row for row in rows if row["record_type"] == "physics_step"]
    full_sample_count = len(samples)
    end_types = ("terminal_pre_reset", "episode_incomplete_end")
    endings = [row for row in rows if row["record_type"] in end_types]
    episode_end = endings[-1] if endings else (samples[-1] if samples else start)
    if max_steps is not None:
        samples = samples[:max_steps * 4]
    end = (samples[-1] if max_steps is not None and len(samples) < full_sample_count
           else episode_end)
    right = [row["finger_contact_n"][1] if row["finger_contact_n"] else None for row in samples]
    first_right = next((row["episode_time_s"] for row, force in zip(samples, right)
                        if force is not None and force >= 1.0), None)
    peak_right = max((force for force in right if force is not None), default=None)
    initial_cube = start["cube_pose_w"]
    displacements = [vector_distance(row["cube_pose_w"], initial_cube) for row in samples]
    max_disp_i = max(range(len(samples)), key=lambda i: displacements[i]) if samples else None
    target_bad = []
    for row in samples:
        targets = row.get("joint_pos_target", {})
        vals = [targets.get("panda_finger_joint1"), targets.get("panda_finger_joint2")]
        if any(value is None or abs(value - 0.04) > 1e-7 for value in vals):
            target_bad.append({"physics_step": row.get("physics_step"), "values": vals})
    contact_missing = sum(
        not isinstance(row.get("finger_contact_n"), list)
        or len(row["finger_contact_n"]) != 2
        or any(force is None for force in row["finger_contact_n"])
        for row in samples
    )
    gap_steps = [i + 1 for i, row in enumerate(samples)
                 if row.get("physics_step") != i + 1]
    episode_dts_ok = all(abs(row["episode_time_s"] - (i + 1) / 120) < 2e-6
                         for i, row in enumerate(samples))
    return {
        "trace": str(path), "trace_sha256": sha256(path),
        "episode_status": ("completed" if any(r["record_type"] == "terminal_pre_reset" for r in endings)
                           else "incomplete" if any(r["record_type"] == "episode_incomplete_end" for r in endings)
                           else "end_record_missing"),
        "episode_end_record_type": end.get("record_type"),
        "episode_end_reason": episode_end.get("end_reason"),
        "termination": episode_end.get("termination"),
        "comparison_window_policy_steps": max_steps,
        "baseline_cube_pose_w": initial_cube,
        "baseline_hand_pose_w": start["hand_pose_w"],
        "end_cube_pose_w": end["cube_pose_w"],
        "end_hand_pose_w": end["hand_pose_w"],
        "sampled_physics_steps": len(samples),
        "first_right_contact_ge_1n_time_s": first_right,
        "peak_left_contact_n": max((row["finger_contact_n"][0] for row in samples
                                     if row["finger_contact_n"][0] is not None), default=None),
        "peak_right_contact_n": peak_right,
        "max_cube_displacement_m": displacements[max_disp_i] if max_disp_i is not None else None,
        "max_cube_displacement_time_s": samples[max_disp_i]["episode_time_s"] if max_disp_i is not None else None,
        "end_cube_displacement_m": vector_distance(end["cube_pose_w"], initial_cube),
        "max_cube_rise_m": max((row["cube_height_m"] - initial_cube[2] for row in samples), default=None),
        "contact_samples": len(samples) - contact_missing,
        "contact_missing_samples": contact_missing,
        "contact_coverage": (len(samples) - contact_missing) / len(samples) if samples else 0,
        "physics_step_gaps": gap_steps,
        "physics_time_grid_ok": episode_dts_ok,
        "fixed_open_targets_all_samples_pass": not target_bad if expected_targets else None,
        "fixed_open_target_violation_count": len(target_bad),
        "fixed_open_target_violation_examples": target_bad[:3],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-open", required=True, type=Path, nargs=2)
    parser.add_argument("--baseline", required=True, type=Path, nargs=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"scope": "selected contact mechanism diagnostic; not a pickup-success estimate",
              "fixed_open": {}, "published_progress_trace_aligned_window": {}}
    for seed, fixed, baseline in zip((8, 11), args.fixed_open, args.baseline):
        fixed_result = analyze(fixed / "ppo-telemetry.jsonl", expected_targets=True)
        baseline_result = analyze(baseline / "ppo-telemetry.jsonl", expected_targets=False,
                                  max_steps=fixed_result["sampled_physics_steps"] // 4)
        launch = fixed / "launch.json"
        exit_path = fixed / "exit.json"
        console = fixed / "console.log"
        fixed_result["launch_sha256"] = sha256(launch)
        fixed_result["console_sha256"] = sha256(console)
        fixed_result["exit"] = json.loads(exit_path.read_text())
        fixed_result["checkpoint"] = json.loads(launch.read_text())["checkpoint"]
        fixed_result["checkpoint_sha256"] = json.loads(launch.read_text())["checkpoint_sha256"]
        for result, root in ((fixed_result, fixed), (baseline_result, baseline)):
            summary = root / "ppo-telemetry.summary.json"
            if summary.exists():
                result["summary_sha256"] = sha256(summary)
                result["summary"] = json.loads(summary.read_text())
        report["fixed_open"][str(seed)] = fixed_result
        report["published_progress_trace_aligned_window"][str(seed)] = baseline_result
    report["checks"] = {
        "all_fixed_open_runs_have_explicit_end_state": all(
            value["episode_status"] == "incomplete" and value["episode_end_record_type"] == "episode_incomplete_end"
            for value in report["fixed_open"].values()),
        "all_fixed_open_targets_exact_each_sample": all(
            value["fixed_open_targets_all_samples_pass"] for value in report["fixed_open"].values()),
        "all_fixed_open_contact_coverage_complete": all(
            value["contact_coverage"] == 1 for value in report["fixed_open"].values()),
        "all_fixed_open_have_no_physics_gaps": all(
            not value["physics_step_gaps"] and value["physics_time_grid_ok"]
            for value in report["fixed_open"].values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["checks"], indent=2))
    if not all(report["checks"].values()):
        raise SystemExit("Trace audit failed")


if __name__ == "__main__":
    main()
