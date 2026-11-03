"""Compare bounded fingertip-pose diagnostics with saved progress and scripted traces."""

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


def records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def first(rows: list[dict], kind: str) -> dict:
    return next(row for row in rows if row["record_type"] == kind)


def right_contact(rows: list[dict]) -> dict:
    return next(row for row in rows if row["record_type"] == "physics_step"
                and row["finger_contact_n"][1] >= 1.0)


def relative_pose(row: dict) -> dict:
    cube = row["cube_pose_w"][:3]
    poses = row["fingertip_body_pose_w"]
    return {
        "hand_center_minus_cube_m": [a - b for a, b in zip(row["hand_pose_w"][:3], cube)],
        "left_finger_frame_minus_cube_m": [a - b for a, b in zip(poses["panda_leftfinger"][:3], cube)],
        "right_finger_frame_minus_cube_m": [a - b for a, b in zip(poses["panda_rightfinger"][:3], cube)],
    }


def analyze(run_root: Path, prior_root: Path, scripted_log: Path,
            import_provenance: Path, zero_action_dir: Path,
            failed_cli_log: Path | None = None) -> dict:
    imports = json.loads(import_provenance.read_text(encoding="utf-8"))
    if imports["isaaclab_head"] != "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4" or imports["isaaclab_status"]:
        raise ValueError("pinned Isaac Lab provenance differs")
    scripted = None
    with scripted_log.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "planned_waypoints" in row:
                scripted = next(waypoint for waypoint in row["planned_waypoints"]
                                if waypoint["label"] == "grasp")
                break
    if scripted is None:
        raise ValueError("scripted grasp waypoint not found")
    cube_start = [-0.55, 0.10, 0.28]
    scripted_fingers = sorted(
        [[value - cube_start[i] for i, value in enumerate(xyz)]
         for xyz in scripted["finger_pos_w"]], key=lambda xyz: xyz[1]
    )

    episodes = []
    input_hashes = {
        "scripted_positive_run_log": sha256(scripted_log),
        "post_run_import_provenance": sha256(import_provenance),
    }
    for seed in (8, 11):
        directory = run_root / f"seed{seed}"
        telemetry = directory / "ppo-telemetry.jsonl"
        summary_path = directory / "ppo-telemetry.summary.json"
        launch_path = directory / "launch.json"
        validation_path = directory / "validation.json"
        old_telemetry = prior_root / f"seed{seed}" / "ppo-telemetry.jsonl"
        saved = records(telemetry)
        prior = records(old_telemetry)
        start = first(saved, "episode_start")
        action = first(saved, "policy_step_start")
        contact = right_contact(saved)
        old_contact = right_contact(prior)
        old_action = first(prior, "policy_step_start")
        steps = [row for row in saved if row["record_type"] == "physics_step"]
        end = next(row for row in saved if row["record_type"] in
                   ("terminal_pre_reset", "episode_incomplete_end"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        for module, info in imports["modules"].items():
            relative = str(Path(info["path"]).relative_to(Path(launch["source_root"]).parent.parent))
            if launch["source_sha256"].get(relative) != info["sha256"]:
                raise ValueError(f"seed {seed} imported module hash differs: {module}")
        if not validation["validated"] or summary["episodes_requested"] != 1:
            raise ValueError(f"seed {seed} diagnostic did not pass validation")
        if len(steps) != validation["physics_samples"] or len(steps) > 120:
            raise ValueError(f"seed {seed} physics sample count differs")
        if action["policy_action"] != old_action["policy_action"]:
            raise ValueError(f"seed {seed} first policy action differs from prior screen")
        if contact["episode_time_s"] != old_contact["episode_time_s"]:
            raise ValueError(f"seed {seed} first right-contact time differs from prior screen")
        first_state_max_difference = max(
            abs(a - b)
            for key in ("cube_pose_w", "hand_pose_w")
            for a, b in zip(contact[key], old_contact[key])
        )
        if first_state_max_difference > 1e-5:
            raise ValueError(f"seed {seed} first-contact pose differs from prior screen")
        if sha256(telemetry) != validation["telemetry_sha256"]:
            raise ValueError(f"seed {seed} telemetry changed since validation")

        cube0 = start["cube_pose_w"]
        def xy_displacement(row):
            return math.hypot(row["cube_pose_w"][0] - cube0[0],
                              row["cube_pose_w"][1] - cube0[1])
        peak = max(steps, key=lambda row: row["finger_contact_n"][1])
        episodes.append({
            "seed": seed,
            "status": summary["episode_results"][0]["status"],
            "termination": summary["episode_results"][0]["termination"],
            "duration_s": summary["episode_results"][0]["duration_s"],
            "physics_samples": len(steps),
            "contact_coverage": summary["episode_results"][0]["contact_coverage"],
            "physics_step_gaps": summary["episode_results"][0]["physics_step_gaps"],
            "first_right_contact_ge_1n_s": contact["episode_time_s"],
            "first_contact_left_right_force_n": contact["finger_contact_n"],
            "first_contact_geometry": relative_pose(contact),
            "reset_geometry": relative_pose(start),
            "first_contact_actual_finger_joints_m": contact["robot_joint_pos"][-2:],
            "first_contact_finger_targets_m": contact["processed_finger_targets"],
            "first_contact_pose_max_abs_difference_from_prior_m": first_state_max_difference,
            "first_policy_action_matches_prior": True,
            "peak_right_force_n": peak["finger_contact_n"][1],
            "peak_right_force_time_s": peak["episode_time_s"],
            "peak_right_force_geometry": relative_pose(peak),
            "max_cube_xy_displacement_m": max(map(xy_displacement, steps)),
            "max_cube_rise_m": max(row["cube_height_delta_m"] for row in steps),
            "end_cube_position_w_m": end["cube_pose_w"][:3],
            "end_record_type": end["record_type"],
            "physical_lift_pass": summary["episode_results"][0]["physical_lift_pass"],
            "run_dir": str(directory.resolve()),
            "source_sha256_at_launch": launch["source_sha256"],
            "checkpoint_sha256": launch["checkpoint_sha256"],
        })
        for label, path in (
            ("telemetry", telemetry), ("summary", summary_path),
            ("launch", launch_path), ("exit", directory / "exit.json"),
            ("validation", validation_path), ("prior_telemetry", old_telemetry)
        ):
            input_hashes[f"seed{seed}_{label}"] = sha256(path)

    zero_telemetry = zero_action_dir / "ppo-telemetry.jsonl"
    zero_summary_path = zero_action_dir / "ppo-telemetry.summary.json"
    zero_validation_path = zero_action_dir / "validation.json"
    zero_rows = records(zero_telemetry)
    zero_start = first(zero_rows, "episode_start")
    zero_steps = [row for row in zero_rows if row["record_type"] == "physics_step"]
    zero_actions = [row for row in zero_rows if row["record_type"] == "policy_step_start"]
    zero_summary = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    zero_validation = json.loads(zero_validation_path.read_text(encoding="utf-8"))
    zero_launch = json.loads((zero_action_dir / "launch.json").read_text(encoding="utf-8"))
    if not zero_validation["validated"] or sha256(zero_telemetry) != zero_validation["telemetry_sha256"]:
        raise ValueError("zero-action validation or telemetry hash differs")
    for module, info in imports["modules"].items():
        relative = str(Path(info["path"]).relative_to(Path(zero_launch["source_root"]).parent.parent))
        if zero_launch["source_sha256"].get(relative) != info["sha256"]:
            raise ValueError(f"zero-action imported module hash differs: {module}")
    seed8_start = first(records(run_root / "seed8" / "ppo-telemetry.jsonl"), "episode_start")
    for key in ("cube_pose_w", "hand_pose_w", "robot_joint_pos"):
        if zero_start[key] != seed8_start[key]:
            raise ValueError(f"zero-action reset differs from policy reset: {key}")
    if zero_start["physics_sampling_mode"] != seed8_start["physics_sampling_mode"]:
        raise ValueError("zero-action and policy physics sampling mode differ")
    if len(zero_steps) != 120 or len(zero_actions) != 30:
        raise ValueError("zero-action control did not reach the one-second cap")
    if any(any(value != 0 for value in row["policy_action"]) for row in zero_actions):
        raise ValueError("nonzero action found in zero-action control")
    zero_episode = zero_summary["episode_results"][0]
    if zero_episode["status"] != "incomplete" or zero_episode["contact_coverage"] != 1.0:
        raise ValueError("zero-action control status or contact coverage differs")
    if zero_episode["physics_step_gaps"] or zero_episode["contact_missing_samples"]:
        raise ValueError("zero-action control has physics or contact gaps")
    zero_at_first = next(row for row in zero_steps if row["episode_time_s"] == 0.05)
    zero_cube = zero_start["cube_pose_w"]
    zero_result = {
        "status": zero_episode["status"],
        "duration_s": zero_episode["duration_s"],
        "physics_samples": len(zero_steps),
        "contact_coverage": zero_episode["contact_coverage"],
        "first_policy_contact_time_s": 0.05,
        "hand_y_minus_cube_y_at_0_05s_m": (
            zero_at_first["hand_pose_w"][1] - zero_at_first["cube_pose_w"][1]
        ),
        "max_absolute_hand_y_drift_m": max(
            abs(row["hand_pose_w"][1] - zero_start["hand_pose_w"][1]) for row in zero_steps
        ),
        "max_cube_xy_displacement_m": max(
            math.hypot(row["cube_pose_w"][0] - zero_cube[0],
                       row["cube_pose_w"][1] - zero_cube[1]) for row in zero_steps
        ),
        "max_left_right_sensor_force_n": [
            max(row["finger_contact_n"][i] for row in zero_steps) for i in (0, 1)
        ],
        "initial_state_equals_seed8_policy": True,
        "action_components_all_zero": True,
        "run_dir": str(zero_action_dir.resolve()),
    }
    for label, path in (
        ("telemetry", zero_telemetry), ("summary", zero_summary_path),
        ("validation", zero_validation_path),
        ("launch", zero_action_dir / "launch.json"),
        ("exit", zero_action_dir / "exit.json"),
        ("console", zero_action_dir.parent / "console.log"),
    ):
        input_hashes[f"zero_action_{label}"] = sha256(path)
    if failed_cli_log is not None:
        input_hashes["zero_action_failed_cli_parse"] = sha256(failed_cli_log)

    return {
        "kind": "fingertip_body_pose_contact_geometry_diagnostic",
        "scope": "two selected, at-most-one-second progress-policy playbacks; no training or EEG",
        "source_checkpoint_sha256": episodes[0]["checkpoint_sha256"],
        "post_run_import_provenance": imports,
        "scripted_grasp_body_frame_minus_cube_m_sorted_by_y": scripted_fingers,
        "cube_half_extent_m": 0.025,
        "episodes": episodes,
        "zero_action_control": zero_result,
        "input_sha256": input_hashes,
        "collector_sha256": sha256(Path(__file__)),
        "limits": [
            "Finger link body frames are not collision-surface or contact-point poses.",
            "The script's planned waypoint is a static reference, not a matched dynamic trajectory.",
            "Only two preselected seeds were screened; neither run estimates pickup reliability.",
            "Subsequent contact dynamics can diverge despite matching first actions and first-contact states.",
            "A 1-second cap can end with an explicitly incomplete episode and cannot score a 12-second pickup trial.",
            "The zero-action control isolates this reset/action comparison, not every policy seed or runtime variant.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--scripted-log", type=Path, required=True)
    parser.add_argument("--import-provenance", type=Path, required=True)
    parser.add_argument("--zero-action-dir", type=Path, required=True)
    parser.add_argument("--failed-cli-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_root, args.prior_root, args.scripted_log,
                     args.import_provenance, args.zero_action_dir, args.failed_cli_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
