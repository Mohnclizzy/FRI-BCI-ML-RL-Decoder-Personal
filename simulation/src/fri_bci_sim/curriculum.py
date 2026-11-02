"""Plan or launch one explicit stage of the physical Franka lift curriculum.

Run a bounded pre-grasp startup pilot, inspect physical trajectories and saved
configuration, then transfer its checkpoint to the table-only task.  This CLI
never advances stages based on training reward alone.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from . import ISAACLAB_COMMIT, ISAACLAB_REF
from .launch import build_command, verify_runtime


STAGES = {
    "pregrasp": ("FRI-Franka-Lift-Pregrasp", "in_hand_pregrasp"),
    "table": ("FRI-Franka-Lift", "table_only"),
}


def build_stage_plan(root: Path, stage: str, num_envs: int, seed: int,
                     iterations: int, checkpoint: Path | None) -> dict:
    """Return an auditable command and stage contract without simulator imports."""
    if stage not in STAGES:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    if stage == "pregrasp" and checkpoint is not None:
        raise ValueError("Pregrasp must start without a checkpoint")
    if stage == "table" and checkpoint is None:
        raise ValueError("Table transfer requires an explicit pregrasp checkpoint")
    task, reset_protocol = STAGES[stage]
    command = build_command(root.resolve(), "train", num_envs, seed, iterations, checkpoint, task=task)
    command.extend(["--viz", "none"])
    return {
        "argv": command,
        "task": task,
        "stage": stage,
        "reset_protocol": reset_protocol,
        "source_checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "isaaclab_ref": ISAACLAB_REF,
        "isaaclab_commit": ISAACLAB_COMMIT,
        "eeg_connected": False,
        "measured_table_pickup_evaluation": False,
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path,
                        help="Absolute or relative pregrasp model_*.pt path, required for table transfer")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Default is dry-run only")
    args = parser.parse_args(argv)

    plan = build_stage_plan(args.isaaclab_root, args.stage, args.num_envs,
                            args.seed, args.iterations, args.checkpoint)
    plan["cwd"] = str(args.run_dir.resolve())
    plan["created_at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return

    verify_runtime(args.isaaclab_root.resolve(), plan["argv"])
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise ValueError("Transfer checkpoint file does not exist")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "launch.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    with (args.run_dir / "console.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(plan["argv"], cwd=args.run_dir, stdout=stream, stderr=subprocess.STDOUT)
    (args.run_dir / "exit.json").write_text(
        json.dumps({"exit_code": result.returncode}) + "\n", encoding="utf-8"
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
