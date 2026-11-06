"""Audit saved Approach playback traces without launching Isaac Lab.

This script reports sampled geometry and contact; it does not rescore a policy
or turn a contact sample into a pickup. The recorder summary remains the source
of the frozen physical lift predicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def analyze(telemetry: Path, summary_path: Path, checkpoint: Path | None = None) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    starts: dict[int, dict] = {}
    ends: dict[int, dict] = {}
    physics: dict[int, list[dict]] = {}
    actions: dict[int, list[dict]] = {}
    record_counts: dict[str, int] = {}

    with telemetry.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            kind = row["record_type"]
            episode = int(row["episode"])
            record_counts[kind] = record_counts.get(kind, 0) + 1
            if kind == "episode_start":
                if episode in starts:
                    raise ValueError(f"duplicate episode start at line {line_number}")
                starts[episode] = row
            elif kind == "terminal_pre_reset":
                if episode in ends:
                    raise ValueError(f"duplicate terminal state at line {line_number}")
                ends[episode] = row
            elif kind == "physics_step":
                physics.setdefault(episode, []).append(row)
            elif kind == "policy_step_start":
                actions.setdefault(episode, []).append(row)
            else:
                raise ValueError(f"unexpected record type {kind!r} at line {line_number}")

    episodes = sorted(starts)
    expected = list(range(1, summary["episodes_requested"] + 1))
    if episodes != expected or sorted(ends) != expected or sorted(physics) != expected:
        raise ValueError("episode start, terminal, physics, and requested IDs differ")
    if summary["episodes_completed"] != len(episodes) or summary["incomplete_episodes"]:
        raise ValueError("playback summary is incomplete")
    if summary.get("error") or summary.get("playback_error"):
        raise ValueError("playback summary contains an error")

    reports = []
    for episode, recorded in zip(episodes, summary["episode_results"], strict=True):
        samples = physics[episode]
        start = starts[episode]
        terminal = ends[episode]
        dt = float(start["physics_dt_s"])
        if recorded["episode"] != episode or recorded["status"] != "completed":
            raise ValueError(f"episode {episode} is not recorded as completed")
        if len(samples) != recorded["physics_samples"] or len(samples) != recorded["contact_samples"]:
            raise ValueError(f"episode {episode} physics/contact sample count differs")
        if recorded["physics_step_gaps"] or recorded["contact_missing_samples"]:
            raise ValueError(f"episode {episode} has a recorded sample or contact gap")
        if not all(row["contact_valid"] for row in samples):
            raise ValueError(f"episode {episode} has invalid contact data")
        times = [float(row["episode_time_s"]) for row in samples]
        if any(b <= a or abs((b - a) - dt) > 2e-6 for a, b in zip(times, times[1:])):
            raise ValueError(f"episode {episode} physics timestamps are not contiguous")
        if not math.isclose(times[-1], recorded["duration_s"], abs_tol=2e-6):
            raise ValueError(f"episode {episode} duration differs from summary")
        if not math.isclose(terminal["episode_time_s"], times[-1], abs_tol=2e-6):
            raise ValueError(f"episode {episode} terminal timestamp differs")

        cube0 = start["cube_pose_w"]
        max_rise = max(row["cube_height_m"] - start["baseline_cube_height_m"] for row in samples)
        if not math.isclose(max_rise, recorded["max_cube_lift_m"], abs_tol=1e-7):
            raise ValueError(f"episode {episode} cube rise differs from summary")
        contacts = [[row["finger_contact_n"][i] for row in samples] for i in (0, 1)]
        first = [next((times[j] for j, value in enumerate(side) if value >= 0.1), None)
                 for side in contacts]
        bilateral_indices = [j for j, row in enumerate(samples)
                             if min(row["finger_contact_n"]) >= 0.1]
        first_right = next((row for row in samples if row["finger_contact_n"][1] >= 0.1), None)
        first_bilateral = times[bilateral_indices[0]] if bilateral_indices else None
        if first_bilateral != recorded["first_contact_time_s"]:
            raise ValueError(f"episode {episode} bilateral-contact time differs")
        negative_targets = [sum(row["processed_finger_targets"][f"panda_finger_joint{i}"] < 0
                                for row in samples) for i in (1, 2)]
        action_values = [value for row in actions[episode] for value in row["policy_action"]]
        reports.append({
            "episode": episode,
            "status": recorded["status"],
            "termination": recorded["termination"],
            "duration_s": recorded["duration_s"],
            "physics_samples": len(samples),
            "contact_coverage": recorded["contact_coverage"],
            "max_cube_rise_m": max_rise,
            "max_cube_xy_displacement_m": max(_distance_xy(row["cube_pose_w"], cube0) for row in samples),
            "end_cube_xy_displacement_m": _distance_xy(terminal["cube_pose_w"], cube0),
            "min_cube_to_hand_body_center_m": min(row["cube_to_hand_distance_m"] for row in samples),
            "first_left_right_contact_s": first,
            "left_right_contact_samples": [sum(value >= 0.1 for value in side) for side in contacts],
            "bilateral_contact_samples": len(bilateral_indices),
            "first_bilateral_contact_s": first_bilateral,
            "max_left_right_sensor_force_n": [max(side) for side in contacts],
            "first_right_contact_actual_finger_positions_m": first_right["robot_joint_pos"][-2:] if first_right else None,
            "first_right_contact_finger_targets_m": first_right["processed_finger_targets"] if first_right else None,
            "negative_left_right_finger_target_samples": negative_targets,
            "policy_action_components_abs_gt_1": sum(abs(value) > 1 for value in action_values),
            "policy_action_components": len(action_values),
            "physical_lift_pass": recorded["physical_lift_pass"],
        })

    return {
        "kind": "offline_approach_table_trace_review",
        "input_sha256": {
            "telemetry": sha256(telemetry),
            "recorder_summary": sha256(summary_path),
            "checkpoint": sha256(checkpoint) if checkpoint else None,
            "analysis_script": sha256(Path(__file__)),
        },
        "input_paths": {
            "telemetry": str(telemetry.resolve()),
            "recorder_summary": str(summary_path.resolve()),
            "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        },
        "sampling_mode": summary["physics_sampling_mode"],
        "record_counts": record_counts,
        "contact_threshold_n": summary["contact_threshold_n"],
        "lift_threshold_m": summary["lift_threshold_m"],
        "sustained_contact_threshold_s": summary["sustained_contact_threshold_s"],
        "episodes": reports,
        "physical_lift_successes": summary["physical_lift_successes"],
        "interpretation_limits": [
            "Finger-force and contact durations are sampled at logged physics steps; no between-sample contact claim.",
            "Hand-body center distance is not fingertip or collider clearance.",
            "A negative commanded finger target is not proof of a force mechanism or an applied joint position.",
            "This single-seed five-episode screen is not a reliability or cross-controller comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.telemetry, args.summary, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
