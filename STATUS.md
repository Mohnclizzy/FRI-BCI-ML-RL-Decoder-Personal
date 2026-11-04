# Current status: Phase 5 stable completion latch bounded failure — 2026-11-04

The one-way completion latch and 24 matched seed-0–2 physical controls are complete. Acquisition required 10 consecutive sampled bilateral-contact states plus >=0.08 m rise; thereafter the absolute end-effector pose and gripper-close command stayed fixed through step 400 regardless of requests. All HOLD/latch invariants and deadlines passed. Positive controls acquired in 21/21 but retained in **17/21**. Four runs fell below the 8 cm rise threshold despite continuing bilateral contact: approach 10-step seed 0, closure 10-step seeds 0 and 1, and closure 60-step seed 0. The [Phase 5 summary](results/robomimic_latch_2026-11-04/summary.md), [frozen protocol](results/robomimic_latch_2026-11-04/protocol.md), and [metrics](results/robomimic_latch_2026-11-04/metrics.json) give exact steps and telemetry. The physical gate failed, so the 120 s S006R12 recorded-EEG raw/confidence/temporal replay was **not run**. This is a bounded simulated failure, not completion from reward or exit.

Raw traces, source, logs and failure video are preserved under `/home/ubuntu/robomimic-native-phase5-20261104/` and in the private [Phase 5 Drive folder](https://drive.google.com/drive/folders/1_2inM732QxX0jLFS1Sgjpe19tGOS8A_P). Raw archive SHA256 `7d9dfe84f1729489c880f5bbfad85313cb65d88c030da0f4be742cd6b1cf0d26`; video SHA256 `c0992e6fa5de3687455be6562bcdbcf77016c038c083dc3ac0fa0750c3b8b5b1`. Both were downloaded and byte-verified after private Drive upload. No active job. Next measurable question is whether a predeclared higher acquisition height margin or compatible-runtime parity check can address the transient rise loss before a fresh qualification batch. Do not advance to EEG qualification on these data.

---

# Historical Phase 4 status: native BC-RNN HOLD/resume bounded failure — 2026-11-04

The continuous native BC-RNN enable/HOLD/resume adapter and 24 matched seed-0–2 controls are complete. Absolute end-effector goals and Panda gripper actuator commands stayed fixed in every HOLD; no learned call or recurrent-counter advance occurred during HOLD, the 20 Hz/500 Hz clocks and 400-step deadline continued, contact telemetry agreed with robosuite's grasp check, and frozen physical drift limits passed. The learned positive control acquired in 3/3 but retained in **0/3** at the terminal state. Brief carrying pauses retained **0/3** and prolonged carrying pauses **2/3**. Corrected closure pauses retained **0/3** brief and **1/3** prolonged. One reset-on-resume recurrent-state intervention retained **1/3** brief carrying and **0/3** brief closure. These results fail the handoff's no-drop/physical-resume gate; the end-to-end EEG-to-learned-robot replay was **not run**. The causal recorded EEG/gate replay port matched the existing contract in software across all 400 decisions for raw, confidence and temporal S006R12 gates. Full results: [summary](results/robomimic_continuous_2026-11-04/summary.md), [matched metrics](results/robomimic_continuous_2026-11-04/metrics.json), [follow-up metrics](results/robomimic_continuous_2026-11-04/followup_metrics.json).

No job remains active. The full raw archive, source, logs and representative video are preserved under `/home/ubuntu/robomimic-native-phase4-20261104/` and in the private [Phase 4 Drive folder](https://drive.google.com/drive/folders/1JVKaaJdDKM7GVnYNQABPGp129MvcOaK9). The archive SHA256 is `2160353974574251b2afdc51b45114667406bb352b0f127adb89e71f301767b1`; video SHA256 is `4a9dd9ad725dde01b4f73e53d832f17bd993a60ee33fdff9fd7cdcd2d45d4896`. The official checkpoint remains in the private Phase 2 folder with SHA256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`. All three were downloaded from Drive and byte-verified by SHA256; connector metadata showed owner-only permissions. Next measurable question: whether a compatible-runtime parity check or targeted retention demonstrations can remove the native post-acquisition drop before repeating fresh HOLD/resume controls. Do not advance to EEG qualification on reward or this diagnostic.

---

# Historical Phase 3 update — 2026-11-04

Bounded native BC-RNN acquisition HOLD diagnostic is complete. Five aligned starts retained the cube with sampled bilateral contact through all 400 steps (5/5), versus Phase 2 uninterrupted-policy retention 0/5. A single seed-0 HOLD-to-policy resume at step 201 also retained (1/1 exploratory). The one intervention fixed the controller end-effector pose at first bilateral-contact/8 cm acquisition and kept the Panda gripper closed; physics and the 20-second horizon continued. The fixed goal was invariant during HOLD. The resulting terminal rise margin was narrow (0.0038–0.0087 m above the 0.08 m threshold). This is a controller equilibrium, not a learned HOLD skill or EEG validation. See `results/robomimic_hold_2026-11-04/summary.md` for full protocol, trace diagnosis, and compatibility limits. Raw archive SHA256 `76b320138491df379f7ed2047711013ac2a1d447842e03f2e91edabf7c9ffdc4` is in the private [Phase 3 Drive folder](https://drive.google.com/drive/folders/1BjC9bhuEL52lAWsZ6D72gpQI-jInziNh); connector readback verified size and owner-only metadata, with download-hash verification outstanding. No active job or long training. Next research step, is held-out seeds and perturbations before reliability claims.

---

# Current execution status — 2026-11-04 04:25 UTC

## Phase and result

Phase 1 relevant artifact recovery is complete; CLI model-account execution and connector publication are verified. Phase 2 bounded native robomimic reproduction is complete. The official Lift PH low-dimensional BC-RNN reached native lift success at least once in 5/5 aligned 400-step starts, but all five dropped the cube before the 20-second terminal state; terminal retention was 0/5. Every start had at least 0.5 seconds of sampled bilateral contact and at least 8 cm rise. This is a retention failure, not a working learned pickup. See `results/robomimic_native_2026-11-04/summary.md`. The earlier Isaac unilateral PPO result remains 3/5 retained in its separate fixed near-table diagnostic; the half-arm rule remains rejected at 0/5. Learned HOLD/resume and EEG integration are still untested.

## Execution and preservation

- Canonical checkout: `/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal`, branch `codex/robot-learning-design`, local HEAD `b3422a0eef9105c4dcdb430474c6e3f3cf5b8828`. The checkout has pre-existing dirty/untracked work; do not reset or clean it.
- `origin` is `Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`; do not modify Angel's repository.
- NVIDIA L4 was idle at inspection (0% utilization, 0 MiB used). No training/evaluation process was observed.
- Best checkpoint original: `outputs/isaac/unilateral-50-seed8-20261103T2310Z/logs/rsl_rl/lift_franka/2026-11-03_23-11-39/model_347.pt`. SHA256 `37eb85242539f7d2f0c36a916f3aa8e44d2fe6a70030a9da18963d2506e673f7`.
- Fresh local continuity directory: `continuity/2026-11-04/`. It contains a hash-verified copy of the best checkpoint, source snapshot, tracked diff, local Git status, base commit and SHA256 manifest. Existing outputs and checkpoints were not replaced.
- Raw full-resolution evaluation telemetry remains in `outputs/isaac/unilateral-eval-5x12s-20261103T2310Z/`. Published compact metrics are in `results/unilateral_2026-11-03/`.

## Off-host recovery verified 2026-11-04

The connected Google Drive account's private [Phase 1 recovery folder](https://drive.google.com/drive/folders/1VG3gNnu3wvFrzyCtBlRLS3ODHK7vFmDy) contains these four uploaded files. Each was downloaded back to the fresh `continuity/2026-11-04/offhost/retrieved/` directory and matched byte-for-byte by SHA256:

| File | Drive file ID | Bytes | SHA256 |
|---|---|---:|---|
| `model_347.pt` | `1iCfCIQ9nGT-qVlWUL-WdmycIWMrKzhMF` | 19,382,755 | `37eb85242539f7d2f0c36a916f3aa8e44d2fe6a70030a9da18963d2506e673f7` |
| `unilateral_eval_raw_5seeds.tar.gz` | `1RiNi3cdVt_jV2ZTNVc3qU6_JhGwwhoGT` | 4,013,580 | `0cd5861d09fcebf31eb1f02289bbfbeb291deb3d3239d9ddaa5733934ade07f4` |
| `source_config_manifests.tar.gz` | `1qmcaGM9jcyXguFEKhBwbaf8kCeydhenI` | 670,741 | `787c68d748fb38b6db8399dbd5d7379d54296406d68506fbd73cf9d3dca0a188` |
| `newest_raw_telemetry_and_run_manifests.tar.gz` | `1dIh61yb6Y1noIm9fLceB_4v7oXWceY7x` | 2,580,654 | `04771c27141e0874d5e547a9e8623e0166c5a1ef099135f5f7d0df37abf227e3` |

The unilateral archive contains all five full-resolution `ppo-telemetry.jsonl` traces (seeds 8–12). The newest archive contains the later half-arm traces (seeds 8–12) and retention diagnostic trace (seed 11), plus their launch/summary files and half-arm manifest. The source/config archive contains the earlier local source snapshot, tracked patch, base commit, Git status and hashes, `configs/`, `simulation/pyproject.toml`, and unilateral training/evaluation launch and run summaries. The retrieved checkpoint loaded as an RSL-RL state dictionary with actor, critic and optimizer state in the matching Isaac Lab environment (`torch 2.11.0+cu130`); this is a load check, not a rollout. The training launch hashes for `near_table_unilateral.py`, `ppo_telemetry.py`, and `run_unilateral_training.py` match current local files. Isaac Lab remains pinned at `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.

Recovery is verified for these four files only. Other historical checkpoints, generated exports, and full logs were not copied off-host in this bounded pass. Drive folder access depends on the connected account; no public sharing was enabled.

## Active job and connectivity

No active job. Network/model execution and private Drive recovery work. Shell GitHub HTTPS has no credentials; the connected GitHub connector is the publication path. The Brev `codex-cli 0.166.0` completed a bounded model-account execution request under the continuing account. The mounted `/home/ubuntu/.codex` is read only; `/home/ubuntu/.codex-cli-runtime` is the writable runtime.

## Phase 2 artifacts and exact next action

Fresh isolated work area: `/home/ubuntu/robomimic-native-phase2-20261104`; no Isaac or Angel repo changes. robomimic `2804dd97dd1625ec861298a35cb677129d3bfacc`, robosuite `d9baa17d5328d7b0ec15ecda54c76c61bb983124`; official checkpoint SHA256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`. Raw all-attempt archive SHA256 `bbf92cc1d59d2bfe6c19d4c22235d67326e3c0cd84707f68b7f28c6e3cd04f5f`. Both are uploaded to the private [Phase 2 Drive folder](https://drive.google.com/drive/folders/1mCICv4tEWmoD1x2AGUZ7_FEiz55A-OrA); connector readback verified sizes and owner-only access, while download-hash verification is outstanding. The installed `mujoco-py 2.1.2.14` differs from the old robosuite source requirements; record and investigate this compatibility limitation before attributing retention failures to the checkpoint alone. Next: inspect the contact/action/drop traces and confirm dependency parity, then run one targeted retention intervention if justified. Do not claim a learned demonstration or advance to EEG qualification from native success alone.
