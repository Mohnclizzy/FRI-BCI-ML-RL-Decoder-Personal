"""Run one fresh-process fixed-open playback with physics-step telemetry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

TASK = "FRI-Franka-Lift-NearTable-Progress-FixedOpen"
CHECKPOINT_SHA256 = "46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    isaac = args.isaaclab_root.resolve()
    source = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 does not match the requested progress checkpoint")
    run_dir.mkdir(parents=True, exist_ok=False)
    telemetry = run_dir / "ppo-telemetry.jsonl"
    command = [str(isaac / ".venv/bin/isaaclab"), "play", "--rl_library", "rsl_rl",
               "--task", TASK, "--num_envs", "1", "--seed", str(args.seed),
               "--checkpoint", str(checkpoint), "--device", "cuda:0",
               "--external_callback=fri_bci_sim.ppo_telemetry.register", "physics=newton_mjwarp"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK, "seed": args.seed, "num_envs": 1,
        "max_policy_steps": 30, "episodes": 1, "physics_dt_s": 1/120,
        "policy_dt_s": 1/30, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "isaaclab_root": str(isaac),
        "isaaclab_commit": subprocess.check_output(["git", "-C", str(isaac), "rev-parse", "HEAD"], text=True).strip(),
        "personal_source_root": str(source),
        "personal_source_commit": subprocess.check_output(["git", "-C", str(source.parent.parent), "rev-parse", "HEAD"], text=True).strip(),
        "pythonpath_precedence": str(source), "command": command,
        "eeg_connected": False, "training": False,
        "diagnostic_only": True,
    }
    (run_dir / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source) + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["FRI_PPO_MAX_STEPS"] = "30"
    environment["FRI_PPO_EPISODES"] = "1"
    environment["FRI_PPO_TELEMETRY"] = str(telemetry)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=run_dir, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    (run_dir / "exit.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
