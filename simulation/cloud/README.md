# Brev and later TACC bring-up

No provisioning, billing, cloud account changes or cluster submissions have been
performed by this scaffold. Use an already-provisioned GPU machine. Brev's Isaac
launchable can have a different release; do not assume its preinstalled runtime
matches this project.

## Existing Brev Ubuntu machine

1. If creating a machine in Brev, choose an Ubuntu 22.04+ x86_64 image and a
   CUDA-capable NVIDIA GPU supported by Isaac Lab 3.0 / PyTorch. Current Isaac Lab
   3.0 requirements specify Python 3.12 and Linux driver 580.65.06 or newer.
   Newton headless does not need RTX rendering. Check the displayed hourly cost
   and available disk before creating it; stop the instance from Brev when done.
   On the connected machine, check `nvidia-smi`, disk capacity and `uv --version`.
   See the versioned [Isaac Lab 3.0 requirements](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation)
   and [Brev quickstart](https://docs.nvidia.com/brev/getting-started/quickstart).
2. Clone this project's branch and set absolute paths. Use fresh destination
   folders; preserve existing checkouts.

```bash
git clone --branch codex/isaaclab-project-scaffold https://github.com/AL-DEV06/FRI-BCI-ML-RL-Decoder.git
export FRI_REPO="$(realpath FRI-BCI-ML-RL-Decoder)"
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab-fri
export ISAACLAB_ROOT="$(realpath IsaacLab-fri)"
git -C "$ISAACLAB_ROOT" rev-parse HEAD
# Expected e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
cd "$ISAACLAB_ROOT"
uv run isaaclab --help
uv run isaaclab train --help
uv pip install --python "$ISAACLAB_ROOT/.venv/bin/python" -e "$FRI_REPO/simulation" --no-deps
```

The `uv run` command initializes the pinned upstream environment and may download
large dependencies. It is a cloud installation step, not a CPU-local check.
Pin the upstream checkout rather than installing an unrelated PyPI package by
version number. Run project commands with that same `.venv` interpreter.

3. Confirm `train --help` exposes `--max_iterations`, `--num_envs`, `--seed`,
   and `--rl_library`. The actual smoke in the next step is the task discovery
   check: it must successfully load `FRI-Franka-Lift` and initialize Newton.
4. First print the plan (this does not start the simulator):

```bash
cd "$FRI_REPO"
"$ISAACLAB_ROOT/.venv/bin/python" -m fri_bci_sim.cli train \
  --isaaclab-root "$ISAACLAB_ROOT" --num-envs 16 --iterations 2 \
  --run-dir "$FRI_REPO/outputs/isaac/smoke-001"
```

After confirming the plan points to the pinned Isaac Lab checkout and a new
`smoke-001` directory, run it once with `--execute`:

```bash
"$ISAACLAB_ROOT/.venv/bin/python" -m fri_bci_sim.cli train \
  --isaaclab-root "$ISAACLAB_ROOT" --num-envs 16 --iterations 2 \
  --run-dir "$FRI_REPO/outputs/isaac/smoke-001" --execute
```

The launcher checks exact upstream source revision, tracked modifications,
Python3.12 and package/task registration before execution. Asset downloads and
reset-bank creation may still take time. A GPU smoke is the first integration
gate; until it passes, mark this task configuration unverified.

5. Review `console.log`, `exit.json`, `launch.json`, and upstream resolved configs
   under `logs/rsl_rl/`. Check table starts and contact before increasing the
   environment count or iterations. Then save explicit checkpoints for playback:

```bash
"$ISAACLAB_ROOT/.venv/bin/python" -m fri_bci_sim.cli play \
  --isaaclab-root "$ISAACLAB_ROOT" --num-envs 1 \
  --checkpoint /absolute/path/to/model_250.pt \
  --run-dir "$FRI_REPO/outputs/isaac/play-001" --execute
```

Playback uses upstream defaults and may run until interrupted; renderer/video
configuration is a subsequent machine-specific step. It is not automatically a
bounded scientific evaluation. Keep task-level metrics separate from playback.

## TACC later

Use `train.slurm` inside a site-approved environment containing the pinned
runtime. Supply allocation, partition, time and GPU resources via your site's
`sbatch` command. No partition, account, Docker support or GPU compatibility is
assumed. Do not install large environments in the batch script or on a login node.
Use site-approved containers or an existing environment after checking policy.

Set `FRI_REPO`, `ISAACLAB_ROOT`, `RUN_DIR`, and optional `NUM_ENVS`, `ITERATIONS`,
`SEED`; pass them to `sbatch --export=ALL` alongside actual site resource flags.
Keep output data/checkpoints on appropriate storage and upload only selected
source/configuration changes to GitHub.
