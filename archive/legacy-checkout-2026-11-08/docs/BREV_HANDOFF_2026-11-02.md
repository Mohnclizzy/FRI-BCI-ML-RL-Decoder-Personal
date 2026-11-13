# BREV handoff — 2026-11-02

## Status and stop point

The bounded table-approach run and its five-episode physical evaluation both completed before the later instruction to stop. The user then prohibited further training, evaluation, and GPU launches. No experiment process remains; the L4 is visible and currently idle. No output was deleted, no replacement evaluation episodes were run, and Jupyter was left running. EEG was disconnected throughout.

The handoff preserves the completed run as observed. It does not claim pickup: **0/5 physical lift successes**. The next learned-pickup policy audit is out of scope and was not started.

## Repository and provenance

- Destination: [Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal), branch `codex/robot-learning-design`.
- Reviewed base: `d8e6b22221bf99c2bb0c31a2383906dc8e64f5f4`, tree `de42c80eed8a20b2e0455c03202d52ddb8701a94`.
- The branch was checked immediately before publishing and still pointed to that reviewed base. The local shell could not authenticate Git transport, so the local checkout was reconstructed through the connected GitHub API; its full tracked tree hash exactly matched the reviewed tree. Its local root commit `6b1a010c209f54bc83315b52f544798f299f6f8e` is a synthetic snapshot, not the GitHub base commit object. The handoff was published by creating a child commit with parent `d8e6b222…` and a non-forced, expected-head-checked fast-forward ref update. No remote changes were overwritten.
- Only the Personal repository was updated. `/home/ubuntu/FRI-project-8b9db42` and its existing outputs were preserved; the source checkpoint there was read only. The Isaac Lab checkout was also left unmodified.
- GitHub handoff commit: see the final URL and SHA in the delivery message.

## Recorder repair and tests

`simulation/src/fri_bci_sim/ppo_telemetry.py` now records a pre-action reset baseline, captures terminal physical state before autoreset, samples contacts at physics-step cadence, records processed arm and finger targets, scores lift/drop dwell by elapsed simulation time, and emits incomplete/error summaries. Task and reset semantics were not changed.

A corrected one-episode validation completed in `outputs/isaac/ppo-telemetry-validation-20261102T201627Z/`: 360 policy steps / 12.0 s; 1,440 physics and contact samples; 100% contact coverage; no physics-step gaps; terminal timeout recorded. It did not lift the object. An earlier preserved validation at `outputs/isaac/ppo-telemetry-validation-20261102T201500Z/` exposed accumulated timestamp rounding (12.000959 s); the rounding issue was fixed before the passing validation and before training.

Commands used for recorder validation:

```bash
FRI_PPO_TELEMETRY=/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/ppo-telemetry-validation-20261102T201627Z/ppo-telemetry.jsonl \
FRI_PPO_MAX_STEPS=360 FRI_PPO_EPISODES=1 \
/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab play --rl_library rsl_rl \
  --task FRI-Franka-Lift-Approach --num_envs 1 --seed 8 \
  --checkpoint /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/ppo-telemetry-validation-20261102T201627Z/checkpoint/model_99.pt \
  --external_callback fri_bci_sim.ppo_telemetry.register physics=newton_mjwarp --viz none
```

Focused telemetry tests: **7 passed**. `git diff --check` passed. Full simulation unittest discovery reported 32 passing tests but could not import `test_physical_evaluation.py` because `pytest` is not installed (`ModuleNotFoundError`). No task-semantic tests were changed.

## Runtime, imported modules, and hashes

- NVIDIA L4 visible; PyTorch `2.11.0+cu130`, CUDA available.
- Isaac Lab commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; worktree clean.
- RSL-RL `5.5.1`; physics backend `newton_mjwarp`.
- `fri_bci_sim.ppo_telemetry`, `.curriculum`, `.tasks.approach`, and `.tasks.lift` were imported from `/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/simulation/src/fri_bci_sim/`.
- Project source SHA256: telemetry `0148706238cb7173a4e24add1e806f93a35ec32d78d5037e53547cc2655677b2`; curriculum `ef27cac3413c2ccb1acb3ef56c559baa09f9f23ac2aaaf51a52117d28ce74e84`; approach `b5e93ce61f79830f2d59391206d2aaeee7f1d927e269299715f6834bbaa2acc6`; lift `19187b0b45539a45a0364c70db1b1fca485dfce854fe3886c66bcc7f124318eb`; focused tests `302c6e2f18dac04342bfc19bbe6260a58a3c9f0fbab9f1fa6b68157a67886416`.
- Installed source hashes for Isaac Lab manager env / RSL-RL train / play / CLI / runner / PPO are recorded in the JSON handoff and original run report at `outputs/isaac/table-approach-64env-100-updates-seed8-20261102T201810Z/RUN_REPORT.md`.

## Training: 100 additional updates from original pregrasp checkpoint

The completed training output is `outputs/isaac/table-approach-64env-100-updates-seed8-20261102T201810Z/`; RSL-RL run: `logs/rsl_rl/lift_franka/2026-11-02_20-18-31/`. Process exit code was 0. The requested and completed budget was exactly 64 environments × 32 rollout steps × 100 updates = **204,800 transitions**. Logs contain 100 updates, from iteration labels `99/199` through `198/199`; final checkpoint iteration is 198. No expansion was launched.

Exact command:

```bash
/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab train \
  --task FRI-Franka-Lift-Approach --rl_library rsl_rl \
  --num_envs 64 --seed 8 --max_iterations 100 \
  --checkpoint /home/ubuntu/FRI-project-8b9db42/outputs/isaac/pregrasp-64x100-seed8/logs/rsl_rl/lift_franka/2026-11-02_00-39-11/model_99.pt \
  --resume_optimizer_state physics=newton_mjwarp --viz none
```

The original source checkpoint SHA256 is `6a9c5ff66ff813c5b3a72596b620f9729953f91febeb564a88421c59c564ee9b`. Optimizer resume was explicitly enabled. Installed RSL-RL `PPO.load` restores optimizer state and uses the saved optimizer group's learning rate; the runner resumes at the stored iteration and performs 100 updates. Actor observation-normalizer mean/variance/std/count were restored with actor state. Evidence in the completed run: one optimizer parameter's step rose from 2,000 to 4,000; normalizer count rose from 204,800 to 409,600. Configured LR was `0.001`, checkpoint optimizer LR was `0.00001`, and final adaptive LR was `0.00011390625000000005`.

Resolved config hashes: agent `76f18b1d5e211cbf38a6de83aef657d69380f5a38b76b71aa2ab214954fb0fad`; environment `87f66af1f3ac48488df46c577aaff7168b205e234c51fde18397814be25703c5`. Final `model_198.pt` SHA256: `4ca3e23b639bda131ec55df8d8ac16551d547b0ae03cae0f5685d91f2597bde7`.

## Five-episode physical evaluation

Evaluation output: `outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/`. It used one environment, seed 8, five consecutive 12-second episodes, 360 policy steps per episode, and the final checkpoint above. All five completed to timeout. Each episode has 1,440 physics/contact samples, full contact coverage, and zero physics-step gaps. No episodes were replaced. Frozen criteria: cube rise at least 0.08 m with both finger sensors at least 0.1 N sustained for 0.5 s; after acquisition, count a drop below baseline +0.04 m sustained for 0.25 s.

| Episode | Max cube rise (m) | Max left/right force (N) | First bilateral contact (s) | Acquired lift | Drops | Result |
|---:|---:|---:|---:|---|---:|---|
| 1 | 0.0000975 | 0.000 / 5.386 | — | No | 0 | Timeout, fail |
| 2 | 0.0001529 | 0.000 / 6.087 | — | No | 0 | Timeout, fail |
| 3 | 0.0001707 | 0.000 / 5.326 | — | No | 0 | Timeout, fail |
| 4 | 0.0002304 | 1.580 / 6.568 | 2.275 | No | 0 | Timeout, fail |
| 5 | 0.0001646 | 0.000 / 8.668 | — | No | 0 | Timeout, fail |

Measured result: **0/5 physical pickups**. `incomplete_episodes=0`; no recorder or playback error. Rewards and process exit were not used as pickup evidence.

Exact evaluation command and per-process environment:

```bash
FRI_PPO_TELEMETRY=/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/ppo-telemetry.jsonl \
FRI_PPO_MAX_STEPS=360 FRI_PPO_EPISODES=5 \
/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab play --rl_library rsl_rl \
  --task FRI-Franka-Lift-Approach --num_envs 1 --seed 8 \
  --checkpoint /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/checkpoint/model_198.pt \
  --external_callback fri_bci_sim.ppo_telemetry.register physics=newton_mjwarp --viz none
```

## Tracked handoff artifacts and retained raw outputs

Small sanitized/machine-readable records are tracked under `results/`:

- `results/ppo_approach_handoff_2026-11-02.json`: execution status, budgets, resume evidence, runtime, and per-episode outcomes.
- `results/ppo_approach_resolved_config_2026-11-02.json`: resolved task/run settings and config/checkpoint hashes.
- `results/ppo_approach_episode_results_2026-11-02.csv`: one row per measured episode.
- `results/ppo_approach_sha256_2026-11-02.json`: SHA256 manifest for source checkpoint, final checkpoint and evaluation copy, configs, source/tests, logs, launch/exit files, and recorder/evaluation telemetry.

Large checkpoints, TensorBoard event data, logs, and raw telemetry remain in the ignored local `outputs/isaac/` directories identified above. They were not uploaded to GitHub; backup status is **unverified**. The separate earlier validation directory and all existing archive outputs were retained. The sanitized tracked results are the durable index, not a backup of these large artifacts.

## Preserved earlier smoke attempts in the archive

The previously existing Approach smoke directories under `/home/ubuntu/FRI-project-8b9db42/outputs/isaac/` were inspected and retained:

| Directory | Result | Evidence |
|---|---|---|
| `table-approach-smoke-20261102/` | Interrupted during startup; no `exit.json`; log ends with `KeyboardInterrupt` during Warp array cleanup. | `launch.json`, `console.log` |
| `table-approach-smoke-ready-20261102/` | Failed before training, exit code 1: `RuntimeError: No CUDA GPUs are available`. | `launch.json`, `exit.json`, `console.log` |
| `table-approach-smoke-l4-20261102T1954Z/` | Completed 4-environment, seed-8, two-update smoke from the verified original `model_99.pt`, exit code 0; two updates (iteration 99 to 101), 256 transitions. | `launch.json`, `exit.json`, `console.log` |

The L4 smoke used the recorded command arguments:

```bash
/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab train --task FRI-Franka-Lift-Approach --rl_library rsl_rl --num_envs 4 --seed 8 --max_iterations 2 --checkpoint /home/ubuntu/FRI-project-8b9db42/outputs/isaac/pregrasp-64x100-seed8/logs/rsl_rl/lift_franka/2026-11-02_00-39-11/model_99.pt physics=newton_mjwarp --viz none
```

This smoke is distinct from and was not used as the source for the completed 100-update run, which resumed from the original checkpoint. All smoke artifacts remain in the archive. Their launch, exit, and log hashes (where present) are included in the SHA256 manifest.
