"""One-second zero-action control for near-table reset settling and contact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8)
    add_launcher_args(parser)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[2]
    source_files = [
        source_root / "simulation/src/fri_bci_sim/ppo_telemetry.py",
        source_root / "simulation/src/fri_bci_sim/tasks/near_table.py",
        source_root / "simulation/src/fri_bci_sim/tasks/near_table_progress.py",
        source_root / "simulation/src/fri_bci_sim/tasks/__init__.py",
        Path(__file__).resolve(),
    ]
    (directory / "launch.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FRI-Franka-Lift-NearTable-Progress",
        "seed": args.seed,
        "max_policy_steps": 30,
        "action_source": "zero tensor with the registered nine-action shape",
        "policy_checkpoint": None,
        "training": False,
        "eeg_connected": False,
        "physics": "newton_mjwarp",
        "source_root": str(source_root / "simulation/src"),
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in source_files},
    }, indent=2) + "\n", encoding="utf-8")
    os.environ["FRI_PPO_RECORD_FINGERTIP_BODY_POSES"] = "1"
    env = None
    wrapper = None
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
        config.seed = args.seed
        config.sim.device = args.device
        resolve_presets(config, selected=("newton_mjwarp",))
        with launch_simulation(config.sim, args):
            try:
                env = gym.make("FRI-Franka-Lift-NearTable-Progress", cfg=config)
                base = env.unwrapped
                if env.action_space.shape != (1, 9):
                    raise ValueError(f"expected [1,9] action shape, got {env.action_space.shape}")
                telemetry = directory / "ppo-telemetry.jsonl"
                with telemetry.open("w", encoding="utf-8") as stream:
                    wrapper = _TelemetryWrapper(env, stream, base.step_dt, telemetry, episodes_limit=1)
                    wrapper.reset()
                    action = torch.zeros(env.action_space.shape, device=base.device)
                    for _ in range(30):
                        with torch.inference_mode():
                            wrapper.step(action)
                        if wrapper.episodes_completed:
                            break
                    summary = wrapper.summary(episodes_requested=1, max_steps_per_episode=30)
                    summary["action_source"] = "zero tensor"
                    summary["checkpoint"] = None
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
