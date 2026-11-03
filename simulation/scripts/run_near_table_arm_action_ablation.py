"""Replay two saved near-table actions with selected arm components zeroed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path


SOURCE_TELEMETRY_SHA256 = "0d2f8372fb803b2c32e17a83428562d9370a7d87c9a562e48efabebe30b470b0"
VARIANTS = {"full": (), "joint1_off": (0,), "joint3_off": (2,), "joint1_joint3_off": (0, 2)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_actions(path: Path, variant: str) -> list[list[float]]:
    if sha256(path) != SOURCE_TELEMETRY_SHA256:
        raise ValueError("saved seed-8 policy telemetry SHA256 differs")
    with path.open(encoding="utf-8") as stream:
        actions = [row["policy_action"] for row in map(json.loads, stream)
                   if row["record_type"] == "policy_step_start"][:2]
    if len(actions) != 2 or any(len(action) != 9 for action in actions):
        raise ValueError("expected the first two nine-component policy actions")
    for action in actions:
        for index in VARIANTS[variant]:
            action[index] = 0.0
    return actions


def main() -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-telemetry", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("this frozen comparison requires --device cuda:0")
    actions = saved_actions(args.source_telemetry, args.variant)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_files = [
        root / "simulation/src/fri_bci_sim/ppo_telemetry.py",
        root / "simulation/src/fri_bci_sim/tasks/near_table.py",
        root / "simulation/src/fri_bci_sim/tasks/near_table_progress.py",
        root / "simulation/src/fri_bci_sim/tasks/__init__.py",
        Path(__file__).resolve(),
    ]
    (directory / "launch.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FRI-Franka-Lift-NearTable-Progress",
        "seed": 8,
        "num_envs": 1,
        "variant": args.variant,
        "modified_arm_action_indices": VARIANTS[args.variant],
        "source_telemetry": str(args.source_telemetry.resolve()),
        "source_telemetry_sha256": SOURCE_TELEMETRY_SHA256,
        "actions": actions,
        "action_source": "saved first two progress-policy outputs, open-loop",
        "max_policy_steps": 2,
        "policy_checkpoint_loaded": False,
        "training": False,
        "eeg_connected": False,
        "physics": "newton_mjwarp",
        "source_root": str(root / "simulation/src"),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in source_files},
    }, indent=2) + "\n", encoding="utf-8")
    os.environ["FRI_PPO_RECORD_FINGERTIP_BODY_POSES"] = "1"
    wrapper = None
    env = None
    report = {"exit_code": None, "error": None}
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import resolve_presets
        from fri_bci_sim.ppo_telemetry import _TelemetryWrapper
        from fri_bci_sim.tasks.near_table_progress import NearTableLiftProgressEnvCfg

        config = NearTableLiftProgressEnvCfg()
        config.play_mode()
        config.scene.num_envs = 1
        config.seed = 8
        config.sim.device = args.device
        resolve_presets(config, selected=("newton_mjwarp",))
        with launch_simulation(config.sim, args):
            try:
                env = gym.make("FRI-Franka-Lift-NearTable-Progress", cfg=config)
                if env.action_space.shape != (1, 9):
                    raise ValueError(f"expected [1,9] action shape, got {env.action_space.shape}")
                base = env.unwrapped
                telemetry = directory / "ppo-telemetry.jsonl"
                with telemetry.open("w", encoding="utf-8") as stream:
                    wrapper = _TelemetryWrapper(env, stream, base.step_dt, telemetry, episodes_limit=1)
                    wrapper.reset()
                    for action_values in actions:
                        action = torch.tensor([action_values], device=base.device)
                        with torch.inference_mode():
                            wrapper.step(action)
                        if wrapper.episodes_completed:
                            break
                    summary = wrapper.summary(episodes_requested=1, max_steps_per_episode=2)
                    summary["action_source"] = "saved first two policy outputs, open-loop"
                    summary["ablation_variant"] = args.variant
                    summary["playback_error"] = None
                    wrapper.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                report["exit_code"] = 0
            finally:
                if wrapper is not None:
                    wrapper.close()
                elif env is not None:
                    env.close()
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        (directory / "exit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(directory), "exit": report, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
