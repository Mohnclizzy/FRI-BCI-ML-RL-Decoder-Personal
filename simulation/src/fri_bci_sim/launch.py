"""Construct auditable upstream CLI invocations without launching compute by default."""

from pathlib import Path
import os
import subprocess
from . import ISAACLAB_COMMIT


def build_command(root: Path, mode: str, num_envs: int, seed: int, iterations: int,
                  checkpoint: Path | None = None) -> list[str]:
    if mode not in ("train", "play") or num_envs < 1 or iterations < 1 or seed < 0:
        raise ValueError("Invalid launch configuration")
    executable = root / ".venv" / ("Scripts/isaaclab.exe" if os.name == "nt" else "bin/isaaclab")
    cmd = [str(executable), mode, "--task", "FRI-Franka-Lift", "--rl_library", "rsl_rl",
           "--num_envs", str(num_envs), "--seed", str(seed)]
    if mode == "train":
        cmd.extend(["--max_iterations", str(iterations)])
    else:
        if checkpoint is None:
            raise ValueError("Playback requires an explicit checkpoint")
        cmd.extend(["--checkpoint", str(checkpoint.resolve())])
    cmd.append("physics=newton_mjwarp")
    return cmd


def verify_runtime(root: Path, command: list[str]):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != ISAACLAB_COMMIT:
        raise RuntimeError(f"Isaac Lab source must be pinned to {ISAACLAB_COMMIT}; found {commit}")
    dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True).strip()
    if dirty:
        raise RuntimeError("Isaac Lab has tracked modifications; use the pinned clean source")
    if not Path(command[0]).is_file():
        raise RuntimeError("Initialize Isaac Lab's .venv first; see simulation/README.md")
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-c",
        "import sys, importlib.metadata as m; "
        "assert sys.version_info[:2] == (3,12), 'Isaac runtime must use Python 3.12'; "
        "assert any(e.value == 'fri_bci_sim.tasks' for e in m.entry_points(group='isaaclab.tasks')), "
        "'Install simulation/ into the Isaac runtime'; "
        "print({p:m.version(p) for p in ('fri-bci-sim','isaaclab','torch','warp-lang')})"], check=True)
