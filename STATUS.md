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
