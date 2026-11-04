# Current execution status — 2026-11-04 04:04 UTC

## Phase and result

Phase 1 continuity and preservation is in progress. No new robot experiment was run. The last verified learned result remains the unilateral-contact PPO checkpoint `model_347.pt`: 3/5 fixed near-table acquisitions retained at episode end in the five-seed diagnostic. The half-arm rule failed 0/5 and is rejected. There is no learned HOLD/resume or EEG integration, no farther-start pickup qualification, and no completed learned demonstration. See `docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md` on the published branch for the execution plan and `docs/HANDOFF_2026-11-03_AFTER_RETENTION.md` for the preceding evidence index. These two files are not in this local checkout.

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

No active job. Network access is now enabled in this executor: DNS resolution of both `chatgpt.com` and `github.com` succeeded on 2026-11-04. The earlier DNS/network blocker is stale. Shell GitHub HTTPS still has no credentials; use the connected GitHub tools for repository reads and publication. The existing Brev `codex-cli 0.166.0` reports ChatGPT login, but CLI model execution has not been reverified; do not treat login status as proof of model-account continuity. The mounted `/home/ubuntu/.codex` is read only here; `/home/ubuntu/.codex-cli-runtime` is the existing writable runtime.

## Exact next action

Phase 1 off-host recovery for the best checkpoint and newest raw telemetry is complete. Next, verify continuing CLI model-account execution and publication access independently, then proceed to the Phase 2 native robomimic reproduction in a fresh isolated environment. Do not start training in this turn.
