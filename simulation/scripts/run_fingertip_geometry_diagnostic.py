"""Run one bounded, opt-in fingertip-geometry PPO playback in a fresh directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PERSONAL = Path(__file__).resolve().parents[2]
ISAACLAB = Path("/home/ubuntu/IsaacLab-fri")
ISAACLAB_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
CHECKPOINT = PERSONAL / (
    "outputs/isaac/near-table-progress-256x250-seed8-20261102T2305Z/"
    "logs/rsl_rl/lift_franka/2026-11-02_23-03-47/model_249.pt"
)
CHECKPOINT_SHA256 = "46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6"
TASK = "FRI-Franka-Lift-NearTable-Progress"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def validate(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "ppo-telemetry.summary.json").read_text(encoding="utf-8"))
    if summary["episodes_requested"] != 1 or not summary["record_fingertip_body_poses"]:
        raise ValueError("diagnostic was not recorded with fingertip poses")
    if summary["physics_sampling_mode"] != "environment_physics_step":
        raise ValueError("unexpected physics sampling mode")
    if summary.get("error") or summary.get("playback_error") or len(summary["episode_results"]) != 1:
        raise ValueError("playback errored or episode result missing")
    episode = summary["episode_results"][0]
    if episode["contact_missing_samples"] or episode["physics_step_gaps"]:
        raise ValueError("contact or physics sampling gap")
    if episode["contact_coverage"] != 1.0 or episode["physics_samples"] > 120:
        raise ValueError("invalid sample coverage or step cap")

    counts: dict[str, int] = {}
    ending = []
    first_right_contact = None
    first_left_contact = None
    with (run_dir / "ppo-telemetry.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["record_type"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind in ("episode_start", "physics_step", "terminal_pre_reset", "episode_incomplete_end"):
                poses = row.get("fingertip_body_pose_w")
                if not isinstance(poses, dict) or set(poses) != {"panda_leftfinger", "panda_rightfinger"}:
                    raise ValueError("missing left/right fingertip body pose")
                if any(len(pose) != 7 or not all(math.isfinite(float(x)) for x in pose)
                       for pose in poses.values()):
                    raise ValueError("invalid fingertip body pose")
            if kind == "physics_step":
                if not row["contact_valid"]:
                    raise ValueError("invalid per-step contact")
                if first_right_contact is None and row["finger_contact_n"][1] >= 1.0:
                    first_right_contact = row["episode_time_s"]
                if first_left_contact is None and row["finger_contact_n"][0] >= 1.0:
                    first_left_contact = row["episode_time_s"]
            if kind in ("terminal_pre_reset", "episode_incomplete_end"):
                ending.append(kind)
    if counts.get("episode_start") != 1 or counts.get("physics_step") != episode["physics_samples"]:
        raise ValueError("start or physics sample count mismatch")
    if len(ending) != 1:
        raise ValueError("expected exactly one terminal or incomplete end-state record")
    expected_end = "terminal_pre_reset" if episode["status"] == "completed" else "episode_incomplete_end"
    if ending[0] != expected_end:
        raise ValueError("end-state record disagrees with episode status")
    return {
        "validated": True,
        "episode_status": episode["status"],
        "duration_s": episode["duration_s"],
        "physics_samples": episode["physics_samples"],
        "contact_coverage": episode["contact_coverage"],
        "physics_step_gaps": episode["physics_step_gaps"],
        "ending_record": ending[0],
        "first_left_right_contact_ge_1n_s": [first_left_contact, first_right_contact],
        "record_counts": counts,
        "telemetry_sha256": sha256(run_dir / "ppo-telemetry.jsonl"),
        "summary_sha256": sha256(run_dir / "ppo-telemetry.summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(8, 11), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.output_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {run_dir}")
    if _git_head(ISAACLAB) != ISAACLAB_COMMIT:
        raise RuntimeError("pinned Isaac Lab commit changed")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("progress checkpoint hash mismatch")

    source_paths = [
        PERSONAL / "simulation/src/fri_bci_sim/ppo_telemetry.py",
        PERSONAL / "simulation/src/fri_bci_sim/tasks/near_table.py",
        PERSONAL / "simulation/src/fri_bci_sim/tasks/near_table_progress.py",
        PERSONAL / "simulation/src/fri_bci_sim/tasks/__init__.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {str(path.relative_to(PERSONAL)): sha256(path) for path in source_paths}
    source_root = PERSONAL / "simulation/src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FRI_PPO_RECORD_FINGERTIP_BODY_POSES"] = "1"
    environment["FRI_PPO_MAX_STEPS"] = "30"
    environment["FRI_PPO_EPISODES"] = "1"
    environment["FRI_PPO_TELEMETRY"] = str(run_dir / "ppo-telemetry.jsonl")
    command = [
        str(ISAACLAB / ".venv/bin/isaaclab"), "play", "--rl_library", "rsl_rl",
        "--task", TASK, "--num_envs", "1", "--seed", str(args.seed),
        "--checkpoint", str(CHECKPOINT),
        "--external_callback=fri_bci_sim.ppo_telemetry.register",
        "physics=newton_mjwarp", "--viz", "none",
    ]
    run_dir.mkdir(parents=True)
    launch = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "seed": args.seed,
        "num_envs": 1,
        "max_policy_steps": 30,
        "episodes_requested": 1,
        "diagnostic_only": True,
        "training": False,
        "eeg_connected": False,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "isaaclab_commit": ISAACLAB_COMMIT,
        "source_root": str(source_root),
        "source_sha256": source_hashes,
        "process_environment_overrides": {
            key: environment[key] for key in (
                "PYTHONPATH", "FRI_PPO_RECORD_FINGERTIP_BODY_POSES",
                "FRI_PPO_MAX_STEPS", "FRI_PPO_EPISODES", "FRI_PPO_TELEMETRY"
            )
        },
        "command": command,
    }
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    result_code = None
    error = None
    try:
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            result_code = subprocess.run(
                command, cwd=run_dir, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=600, check=False,
            ).returncode
    except subprocess.TimeoutExpired as exc:
        error = f"TimeoutExpired: {exc}"
    (run_dir / "exit.json").write_text(
        json.dumps({"exit_code": result_code, "error": error}, indent=2) + "\n", encoding="utf-8"
    )
    if result_code != 0:
        raise RuntimeError(f"playback exited {result_code}: {error}; inspect {run_dir}")
    if {str(path.relative_to(PERSONAL)): sha256(path) for path in source_paths} != source_hashes:
        raise RuntimeError("source changed during playback")
    validation = validate(run_dir)
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **validation}, indent=2))


if __name__ == "__main__":
    main()
