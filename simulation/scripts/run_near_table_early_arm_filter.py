"""One frozen 12-second progress-policy episode with an opt-in early arm filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PERSONAL = Path(__file__).resolve().parents[2]
ISAACLAB = Path("/home/ubuntu/IsaacLab-fri")
CHECKPOINT = (PERSONAL / "outputs/isaac/near-table-progress-256x250-seed8-20261102T2305Z"
              "/logs/rsl_rl/lift_franka/2026-11-02_23-03-47/model_249.pt")
CHECKPOINT_SHA256 = "46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6"
SOURCE_TRACE = (PERSONAL / "outputs/isaac/fingertip-geometry-progress-20261103T0314Z"
                "/seed8/ppo-telemetry.jsonl")
SOURCE_TRACE_SHA256 = "0d2f8372fb803b2c32e17a83428562d9370a7d87c9a562e48efabebe30b470b0"
ISAACLAB_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path) -> dict:
    summary_path = directory / "ppo-telemetry.summary.json"
    telemetry_path = directory / "ppo-telemetry.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["episodes_requested"] != 1 or summary["max_policy_steps_per_episode"] != 360:
        raise ValueError("one-episode 12-second cap differs")
    if summary["early_arm_filter"] != "joint1_joint3_first_two":
        raise ValueError("early arm filter was not active")
    if summary["physics_sampling_mode"] != "environment_physics_step":
        raise ValueError("physics sampling mode differs")
    if summary.get("error") or summary.get("playback_error"):
        raise ValueError("playback error")
    if len(summary["episode_results"]) != 1:
        raise ValueError("episode result missing")
    episode = summary["episode_results"][0]
    if episode["contact_missing_samples"] or episode["physics_step_gaps"]:
        raise ValueError("contact or physics sample gap")
    if episode["contact_coverage"] != 1.0 or episode["physics_samples"] > 1440:
        raise ValueError("contact coverage or 12-second cap differs")
    counts = {}
    starts = []
    ends = []
    with telemetry_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["record_type"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "policy_step_start":
                starts.append(row)
                applied = row["applied_action"]
                raw = row["policy_action"]
                expected = [0.0 if row["policy_step"] <= 2 and i in (0, 2) else x
                            for i, x in enumerate(raw)]
                if applied != expected:
                    raise ValueError("applied action differs from frozen intervention")
            if kind in ("episode_start", "physics_step", "terminal_pre_reset", "episode_incomplete_end"):
                poses = row.get("fingertip_body_pose_w")
                if not isinstance(poses, dict) or set(poses) != {"panda_leftfinger", "panda_rightfinger"}:
                    raise ValueError("fingertip body poses missing")
            if kind == "physics_step" and not row["contact_valid"]:
                raise ValueError("invalid physics-step contact")
            if kind in ("terminal_pre_reset", "episode_incomplete_end"):
                ends.append(kind)
    with SOURCE_TRACE.open(encoding="utf-8") as stream:
        source_first_action = next(row["policy_action"] for row in map(json.loads, stream)
                                   if row["record_type"] == "policy_step_start")
    if len(starts) < 2 or starts[0]["policy_action"] != source_first_action:
        raise ValueError("policy's first raw action differs from frozen source")
    if counts.get("episode_start") != 1 or counts.get("physics_step") != episode["physics_samples"]:
        raise ValueError("start or sample count differs")
    expected_end = "terminal_pre_reset" if episode["status"] == "completed" else "episode_incomplete_end"
    if ends != [expected_end]:
        raise ValueError("terminal/incomplete end state differs")
    return {
        "validated": True,
        "status": episode["status"],
        "duration_s": episode["duration_s"],
        "physics_samples": episode["physics_samples"],
        "contact_coverage": episode["contact_coverage"],
        "physical_lift_pass": episode["physical_lift_pass"],
        "end_record": ends[0],
        "record_counts": counts,
        "telemetry_sha256": sha256(telemetry_path),
        "summary_sha256": sha256(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256 or sha256(SOURCE_TRACE) != SOURCE_TRACE_SHA256:
        raise ValueError("checkpoint or source trace SHA256 differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ISAACLAB, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ISAACLAB, text=True).strip()
    if head != ISAACLAB_COMMIT or status:
        raise ValueError("pinned Isaac Lab commit or clean status differs")
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_copy = directory / "input" / "model_249.pt"
    checkpoint_copy.parent.mkdir()
    shutil.copy2(CHECKPOINT, checkpoint_copy)
    if sha256(checkpoint_copy) != CHECKPOINT_SHA256:
        raise ValueError("isolated checkpoint copy SHA256 differs")
    files = [
        PERSONAL / "simulation/src/fri_bci_sim/ppo_telemetry.py",
        PERSONAL / "simulation/src/fri_bci_sim/tasks/near_table.py",
        PERSONAL / "simulation/src/fri_bci_sim/tasks/near_table_progress.py",
        PERSONAL / "simulation/src/fri_bci_sim/tasks/__init__.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {str(path.relative_to(PERSONAL)): sha256(path) for path in files}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PERSONAL / "simulation/src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["FRI_PPO_RECORD_FINGERTIP_BODY_POSES"] = "1"
    environment["FRI_PPO_EARLY_ARM_FILTER"] = "joint1_joint3_first_two"
    environment["FRI_PPO_MAX_STEPS"] = "360"
    environment["FRI_PPO_EPISODES"] = "1"
    environment["FRI_PPO_TELEMETRY"] = str(directory / "ppo-telemetry.jsonl")
    command = [
        str(ISAACLAB / ".venv/bin/isaaclab"), "play", "--rl_library", "rsl_rl",
        "--task", "FRI-Franka-Lift-NearTable-Progress", "--num_envs", "1",
        "--seed", "8", "--checkpoint", str(checkpoint_copy),
        "--external_callback=fri_bci_sim.ppo_telemetry.register",
        "physics=newton_mjwarp", "--viz", "none",
    ]
    (directory / "launch.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "seed": 8,
        "num_envs": 1,
        "episodes_requested": 1,
        "max_policy_steps": 360,
        "training": False,
        "eeg_connected": False,
        "intervention": "zero action components 0 and 2 only on policy steps 1 and 2",
        "original_checkpoint": str(CHECKPOINT),
        "checkpoint_copy": str(checkpoint_copy),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_trace_sha256": SOURCE_TRACE_SHA256,
        "isaaclab_commit": head,
        "isaaclab_status": status,
        "source_sha256": source_hashes,
        "process_environment_overrides": {
            key: environment[key] for key in (
                "PYTHONPATH", "FRI_PPO_RECORD_FINGERTIP_BODY_POSES", "FRI_PPO_EARLY_ARM_FILTER",
                "FRI_PPO_MAX_STEPS", "FRI_PPO_EPISODES", "FRI_PPO_TELEMETRY"
            )
        },
    }, indent=2) + "\n", encoding="utf-8")
    return_code = None
    error = None
    try:
        with (directory / "console.log").open("w", encoding="utf-8") as log:
            return_code = subprocess.run(command, cwd=directory, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         timeout=600, check=False).returncode
    except subprocess.TimeoutExpired as exc:
        error = f"TimeoutExpired: {exc}"
    (directory / "exit.json").write_text(json.dumps({"exit_code": return_code, "error": error},
                                                  indent=2) + "\n", encoding="utf-8")
    if return_code != 0:
        raise RuntimeError(f"playback exited {return_code}: {error}; inspect {directory}")
    if {str(path.relative_to(PERSONAL)): sha256(path) for path in files} != source_hashes:
        raise RuntimeError("source changed during playback")
    validation = validate(directory)
    (directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(directory), **validation}, indent=2))


if __name__ == "__main__":
    main()
