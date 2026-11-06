"""Validate and summarize the frozen two-action near-table arm ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VARIANTS = ("full-v2", "joint1_off", "joint3_off", "joint1_joint3_off")
MODIFIED = {
    "full-v2": (), "joint1_off": (0,), "joint3_off": (2,),
    "joint1_joint3_off": (0, 2),
}
SOURCE_SHA256 = "0d2f8372fb803b2c32e17a83428562d9370a7d87c9a562e48efabebe30b470b0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def collect(root: Path, source_trace: Path, failed_cli_log: Path) -> dict:
    if sha256(source_trace) != SOURCE_SHA256:
        raise ValueError("saved policy trace SHA256 differs")
    source_rows = rows(source_trace)
    source_actions = [r["policy_action"] for r in source_rows
                      if r["record_type"] == "policy_step_start"][:2]
    source_physics = [r for r in source_rows if r["record_type"] == "physics_step"][:8]
    source_start = next(r for r in source_rows if r["record_type"] == "episode_start")
    input_hashes = {"source_policy_telemetry": SOURCE_SHA256,
                    "failed_cli_parse_log": sha256(failed_cli_log)}
    variants = []
    common_source_hashes = None
    full_match = None
    for name in VARIANTS:
        directory = root / name
        telemetry = directory / "ppo-telemetry.jsonl"
        launch = json.loads((directory / "launch.json").read_text(encoding="utf-8"))
        exit_record = json.loads((directory / "exit.json").read_text(encoding="utf-8"))
        validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "ppo-telemetry.summary.json").read_text(encoding="utf-8"))
        trace = rows(telemetry)
        start = next(r for r in trace if r["record_type"] == "episode_start")
        physics = [r for r in trace if r["record_type"] == "physics_step"]
        actions = [r["policy_action"] for r in trace if r["record_type"] == "policy_step_start"]
        expected = [[0.0 if i in MODIFIED[name] else value
                     for i, value in enumerate(action)] for action in source_actions]
        if (launch["variant"] != ("full" if name == "full-v2" else name)
                or launch["actions"] != expected or actions != expected):
            raise ValueError(f"{name}: action sequence differs from frozen source")
        if exit_record != {"exit_code": 0, "error": None}:
            raise ValueError(f"{name}: failed exit")
        if not validation["validated"] or validation["telemetry_sha256"] != sha256(telemetry):
            raise ValueError(f"{name}: telemetry validation or hash differs")
        if (validation["physics_samples"] != 8 or validation["contact_coverage"] != 1.0
                or validation["physics_step_gaps"] != 0
                or validation["ending_record"] != "episode_incomplete_end"):
            raise ValueError(f"{name}: missing physics/contact samples or end state")
        if summary["playback_error"] or summary["episode_results"][0]["status"] != "incomplete":
            raise ValueError(f"{name}: playback error or status differs")
        if common_source_hashes is None:
            common_source_hashes = launch["source_sha256"]
        elif launch["source_sha256"] != common_source_hashes:
            raise ValueError(f"{name}: code changed across ablation runs")
        for key in ("cube_pose_w", "hand_pose_w", "robot_joint_pos"):
            if start[key] != source_start[key]:
                raise ValueError(f"{name}: reset state differs: {key}")
        at_contact_time = next(row for row in physics if row["episode_time_s"] == 0.05)
        cube_y = at_contact_time["cube_pose_w"][1]
        variant = {
            "name": name,
            "modified_action_indices": MODIFIED[name],
            "duration_s": validation["duration_s"],
            "physics_samples": 8,
            "contact_coverage": 1.0,
            "end_status": "incomplete_at_two_action_cap",
            "first_left_right_contact_ge_1n_s": validation["first_left_right_contact_ge_1n_s"],
            "hand_y_minus_cube_y_at_0_05s_m": at_contact_time["hand_pose_w"][1] - cube_y,
            "cube_y_shift_at_0_05s_m": cube_y - start["cube_pose_w"][1],
            "left_right_finger_frame_y_minus_cube_y_at_0_05s_m": [
                at_contact_time["fingertip_body_pose_w"][body][1] - cube_y
                for body in ("panda_leftfinger", "panda_rightfinger")
            ],
            "left_right_sensor_force_at_0_05s_n": at_contact_time["finger_contact_n"],
            "run_dir": str(directory.resolve()),
        }
        if name == "full-v2":
            full_match = {
                key: max(abs(a - b) for measured, original in zip(physics, source_physics)
                         for a, b in zip(measured[key], original[key]))
                for key in ("cube_pose_w", "hand_pose_w", "robot_joint_pos")
            }
            if any(value > 1e-5 for value in full_match.values()):
                raise ValueError("full-action control did not reproduce saved state")
            if variant["first_left_right_contact_ge_1n_s"] != [None, 0.05]:
                raise ValueError("full-action first contact differs")
        variants.append(variant)
        for label, path in (
            ("telemetry", telemetry),
            ("summary", directory / "ppo-telemetry.summary.json"),
            ("validation", directory / "validation.json"),
            ("launch", directory / "launch.json"),
            ("exit", directory / "exit.json"),
            ("console", root / f"{name}.console.log"),
        ):
            input_hashes[f"{name}_{label}"] = sha256(path)
    return {
        "kind": "two_action_near_table_arm_component_ablation",
        "scope": "one seed, one reset, two saved open-loop actions, four variants, no training or EEG",
        "source_policy_checkpoint_sha256": "46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6",
        "source_telemetry_sha256": SOURCE_SHA256,
        "source_sha256_at_launch": common_source_hashes,
        "full_control_max_abs_difference_from_saved": full_match,
        "variants": variants,
        "input_sha256": input_hashes,
        "collector_sha256": sha256(Path(__file__)),
        "limits": [
            "The two actions are saved policy outputs, but the ablations are open-loop and load no policy checkpoint.",
            "Only the first 0.066667 s of one fixed reset is tested; absence of early contact is not pickup success.",
            "Finger body frames are not collider surfaces or contact points.",
            "Removing either action component also changes coupled downstream motion; this does not identify a unique reward defect.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-telemetry", type=Path, required=True)
    parser.add_argument("--failed-cli-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_root, args.source_telemetry, args.failed_cli_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
