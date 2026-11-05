# Connected EEG→robot bottleneck — Stage 1 (Phase A + first causal read)

**Date:** 2026-11-05 (UTC) · **Author:** sole experiment writer · **Status:** bounded phase complete, honest partial results (S002/S003 causal read blocked on data/cred/mne).

## What this phase did (bounded, within existing authority)

1. **Inventory** of S001/S002/S003 raw EEG and split/decoder assets (structure + hashes only; **no S002/S003 run-12 outcome files opened** — none exist locally).
2. **Froze** a prospective evaluation protocol + sha256-pinned manifest: `docs/CONNECTED_BOTTLENECK_STAGE1_FREEZE_2026-11-05.md` + `manifest.json`.
3. **Verified the robot-replay execution path is live** (was blocked on a missing MuJoCo native lib): linked the already-local MuJoCo lib to `~/.mujoco/mujoco210`, set `LD_LIBRARY_PATH`; `robosuite 1.2.0` imports; ran **one** bounded control replay.
4. **Quantified the EEG-side activation bottleneck** from existing decision traces (S001, S006–S010).
5. **First causal data point** on the anchor subject (S001).

## Key result — the bottleneck is EEG-side activation, not the robot

Frozen native robomimic BC-RNN policy (lift), 400-step rollout @ 5 Hz latch, CPU.

**Anchor S001 (raw-gate duty ≈ 0.46):**
| arm | n | complete_retained | complete_unretained | latch/hold invariants |
|---|---:|---:|---:|---|
| EEG `raw` (seeds 0–2) | 3 | 3 | 0 | pass |
| EEG `confidence` (seeds 0–2) | 3 | 3 | 0 | pass |
| EEG `temporal` (seeds 0–2) | 3 | 3 | 0 | pass |
| **control `always-enable` seed 0 (this phase)** | 1 | **1** | 0 | pass |

→ Whenever the EEG arm is active, the policy acquires **and retains**; the robot side holds the cube (`complete_retained`, `max_hold_cube_drift_m = 0`). For the anchor, the robot is not the bottleneck.

**Activation bottleneck (grasp_lift command duty cycle, same frozen decoder+gates, 353 windows each):**

| subject | raw | confidence | temporal |
|---|---:|---:|---:|
| S001 | 0.46 | 0.30 | 0.43 |
| S006 | 0.25 | **0.012** | 0.20 |
| S007 | 0.51 | 0.51 | 0.51 |
| S008 | 0.67 | **0.565** | 0.67 |
| S009 | 0.68 | 0.47 | 0.56 |
| S010 | 0.49 | **0.101** | 0.39 |

The **confidence gate** collapses duty to ~0.01 (S006) / ~0.10 (S010) while the raw gate stays ~0.25–0.49 and the temporal gate stays ~0.20–0.39. That is a **50× subject-to-subject swing** in command rate from the frozen gate — the activation bottleneck, on the EEG side. A subject whose confidence gate collapses (as S006 does) will issue so few robot commands that the arm is gated-off, independent of robot policy quality.

## What is blocked (honest, not elided)

- **Definitive causal test (EEG-fails while always-on-passes on a held-out subject):** requires **S002/S003** raw run-12 EEG. **Blocked**: (a) no S002/S003 EDF locally or in the PhysioNet cache; (b) `physionet.username/password` absent (credentials must be provided, then `eegbci.download_dataset('EEGMMIDB')` is a large multi-hour/GB download); (c) `mne` is **not installed** in the Phase 2 venv (needed to decode S002/S003). Until then the causal read is inferred from S006–S010 duty cycles, not measured on a fresh held-out subject.
- **Phase C decoder comparison (CSP/Riemannian/EEGNet):** blocked on `mne` install (network) and the same S002/S003 data gap.

## Preserved (untouched)
All prior S006–S010 decision traces, S001 held-out replay results, the dirty canonical checkout, raw EDFs, and the frozen policy checkpoint. New writes only under `docs/CONNECTED_BOTTLENECK_STAGE1_FREEZE_2026-11-05.md`, `results/connected_bottleneck_stage1_2026-11-05/`, and the local `~/.mujoco` symlink (reversible).

## Next bounded steps (awaiting owner/data)
1. Provide PhysioNet credentials → fetch S002/S003 → install `mne` → run the **S002/S003** matched-control replay (EEG `raw`/`confidence`/`temporal` × seeds 0–2 + `always-enable` + `pause` controls) → **the** causal read.
2. Phase C decoder comparison on the identical frozen split.
3. Phase D (BC→PPO vs PPO-continuation) and Phase E (Ofner label/mapping) — later phases, separate bounded steps.

See `eeg_side_bottleneck.md`, `manifest.json`, and the freeze doc for the full frozen specification.
