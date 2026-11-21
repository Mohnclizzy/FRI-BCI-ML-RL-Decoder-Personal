"""Audit the one-episode early-arm filter against saved unmodified seed 8."""

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


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def episode(directory: Path) -> dict:
    summary = json.loads((directory / "ppo-telemetry.summary.json").read_text(encoding="utf-8"))
    trace = rows(directory / "ppo-telemetry.jsonl")
    start = next(row for row in trace if row["record_type"] == "episode_start")
    steps = [row for row in trace if row["record_type"] == "physics_step"]
    end = next((row for row in trace if row["record_type"] in
                ("terminal_pre_reset", "episode_incomplete_end")), None)
    result = summary["episode_results"][0]
    if summary.get("error") or summary.get("playback_error") or result["contact_coverage"] != 1.0:
        raise ValueError(f"error or incomplete contact coverage: {directory}")
    if result["physics_step_gaps"] or result["contact_missing_samples"]:
        raise ValueError(f"physics/contact gap: {directory}")
    if len(steps) != result["physics_samples"]:
        raise ValueError(f"physics sample count differs: {directory}")
    if end is None:
        raise ValueError(f"terminal or incomplete end missing: {directory}")
    return {
        "status": result["status"],
        "duration_s": result["duration_s"],
        "termination": result["termination"],
        "physics_samples": len(steps),
        "contact_coverage": result["contact_coverage"],
        "physics_step_gaps": result["physics_step_gaps"],
        "first_left_right_contact_ge_1n_s": [
            next((row["episode_time_s"] for row in steps
                  if row["finger_contact_n"][index] >= 1.0), None)
            for index in (0, 1)
        ],
        "max_left_right_sensor_force_n": [
            max(row["finger_contact_n"][index] for row in steps) for index in (0, 1)
        ],
        "max_cube_xy_displacement_m": max(
            math.hypot(row["cube_pose_w"][0] - start["cube_pose_w"][0],
                       row["cube_pose_w"][1] - start["cube_pose_w"][1])
            for row in steps
        ),
        "max_cube_rise_m": max(row["cube_height_delta_m"] for row in steps),
        "end_cube_position_w_m": end["cube_pose_w"][:3],
        "end_record_type": end["record_type"],
        "physical_lift_pass": result["physical_lift_pass"],
        "drop_count": result["drop_count"],
        "first_policy_action": next(row["policy_action"] for row in trace
                                    if row["record_type"] == "policy_step_start"),
        "run_dir": str(directory.resolve()),
    }


def collect(run_dir: Path, baseline_dir: Path) -> dict:
    launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    exit_record = json.loads((run_dir / "exit.json").read_text(encoding="utf-8"))
    if exit_record != {"exit_code": 0, "error": None} or not validation["validated"]:
        raise ValueError("filtered playback did not pass exit and validation gates")
    if sha256(run_dir / "ppo-telemetry.jsonl") != validation["telemetry_sha256"]:
        raise ValueError("filtered telemetry changed after validation")
    if sha256(run_dir / "input/model_249.pt") != launch["checkpoint_sha256"]:
        raise ValueError("isolated checkpoint copy hash differs")
    if sha256(Path(launch["original_checkpoint"])) != launch["checkpoint_sha256"]:
        raise ValueError("original checkpoint hash differs")
    measured = episode(run_dir)
    baseline = episode(baseline_dir)
    if measured["first_policy_action"] != baseline["first_policy_action"]:
        raise ValueError("first raw policy action differs from unmodified seed 8")
    if measured["status"] != "completed" or measured["end_record_type"] != "terminal_pre_reset":
        raise ValueError("filtered episode did not save a completed terminal state")
    input_hashes = {}
    for label, path in (
        ("filtered_launch", run_dir / "launch.json"),
        ("filtered_exit", run_dir / "exit.json"),
        ("filtered_validation", run_dir / "validation.json"),
        ("filtered_summary", run_dir / "ppo-telemetry.summary.json"),
        ("filtered_telemetry", run_dir / "ppo-telemetry.jsonl"),
        ("filtered_console", run_dir / "console.log"),
        ("isolated_checkpoint", run_dir / "input/model_249.pt"),
        ("exported_policy_jit", run_dir / "input/exported/policy.pt"),
        ("exported_policy_onnx", run_dir / "input/exported/policy.onnx"),
        ("exported_policy_onnx_data", run_dir / "input/exported/policy.onnx.data"),
        ("baseline_summary", baseline_dir / "ppo-telemetry.summary.json"),
        ("baseline_telemetry", baseline_dir / "ppo-telemetry.jsonl"),
    ):
        input_hashes[label] = sha256(path)
    return {
        "kind": "one_episode_near_table_inference_only_early_arm_filter",
        "scope": "one fixed seed-8 physical episode, one environment, 12-second cap, no training or EEG",
        "intervention": launch["intervention"],
        "checkpoint_sha256": launch["checkpoint_sha256"],
        "isaaclab_commit": launch["isaaclab_commit"],
        "source_sha256_at_launch": launch["source_sha256"],
        "filtered": measured,
        "unmodified_saved_seed8": baseline,
        "input_sha256": input_hashes,
        "collector_sha256": sha256(Path(__file__)),
        "limits": [
            "One selected seed cannot estimate pickup reliability or spatial generalization.",
            "The policy remains closed-loop after the modified first action, so later contact paths differ.",
            "Contact force fields are sampled filtered sensor proxies, not bilateral grasp force measurements.",
            "The terminal flag and process exit do not imply a lift; physical_lift_pass uses the frozen threshold.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_dir, args.baseline_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
