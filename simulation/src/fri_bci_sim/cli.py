"""CPU-only replay/metrics plus opt-in launch of the pinned Isaac Lab runtime."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from . import ISAACLAB_COMMIT, ISAACLAB_REF
from .commands import CommandGate
from .launch import build_command, verify_runtime
from .metrics import Outcome, summarize
from .replay import load_trial_estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    replay = sub.add_parser("replay", help="Export offline trial requests; does not run a robot")
    replay.add_argument("--predictions", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--threshold", type=float, default=0.7)
    score = sub.add_parser("score", help="Summarize measured episode telemetry JSONL")
    score.add_argument("--outcomes", type=Path, required=True)
    for mode in ("train", "play"):
        launch = sub.add_parser(mode)
        launch.add_argument("--isaaclab-root", type=Path, required=True)
        launch.add_argument("--num-envs", type=int, default=16)
        launch.add_argument("--seed", type=int, default=0)
        launch.add_argument("--iterations", type=int, default=2)
        launch.add_argument("--checkpoint", type=Path)
        launch.add_argument("--run-dir", type=Path, required=True)
        launch.add_argument("--execute", action="store_true", help="Without this flag only print the plan")
    args = parser.parse_args()
    if args.operation == "replay":
        # One observation per isolated trial supplies no temporal persistence evidence.
        gate = CommandGate(threshold=args.threshold, persistence_s=0)
        estimates = load_trial_estimates(args.predictions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for estimate in estimates:
                request = gate.update(estimate, estimate.available_at_s)
                stream.write(json.dumps({"estimate": asdict(estimate), "request": asdict(request),
                                         "mode": "isolated_trial"}) + "\n")
        print(f"Wrote {len(estimates)} trial requests; no robot executed")
    elif args.operation == "score":
        outcomes = [Outcome(**json.loads(line)) for line in args.outcomes.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(summarize(outcomes), indent=2))
    else:
        root = args.isaaclab_root.resolve()
        command = build_command(root, args.operation, args.num_envs, args.seed, args.iterations, args.checkpoint)
        plan = {"argv": command, "cwd": str(args.run_dir.resolve()), "isaaclab_ref": ISAACLAB_REF,
                "isaaclab_commit": ISAACLAB_COMMIT, "task": "FRI-Franka-Lift",
                "reset_protocol": "table_only", "eeg_connected": False,
                "created_at": datetime.now(timezone.utc).isoformat()}
        print(json.dumps(plan, indent=2))
        if args.execute:
            verify_runtime(root, command)
            if args.operation == "play" and not args.checkpoint.is_file():
                raise ValueError("Checkpoint file does not exist")
            args.run_dir.mkdir(parents=True, exist_ok=False)
            (args.run_dir / "launch.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            with (args.run_dir / "console.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=args.run_dir, stdout=stream, stderr=subprocess.STDOUT)
            (args.run_dir / "exit.json").write_text(json.dumps({"exit_code": result.returncode}), encoding="utf-8")
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
