# Phase 6 summary — +0.015 m world-z completion-equilibrium goal

**Result:** Phase 6 is complete. **Positive controls acquired 21/21 and retained 21/21** (terminal rise **0.09222–0.10259 m**); **all three always-HOLD negatives stayed negative (0/3 acquired**, terminal rise −0.01022 m). The four Phase 5 height-margin failures — `approach_10_seed0`, `closure_10_seed0`, `closure_10_seed1`, `closure_60_seed0` — were **recovered**; **0 regressions**; all 24 runs pass the 400-step deadline and every HOLD/latch invariant.

**Gate:** the predeclared physical-control gate for the exploratory EEG replay is **ESTABLISHED**. This is a **development result on reused seeds 0–2 — not held-out qualification and not a learned-EEG demonstration.**

**Command:** see [protocol6.md](protocol6.md) (retrospective). Short form: env `MUJOCO_PY_MUJOCO_PATH`/`LD_LIBRARY_PATH` → `runtime/mujoco210`, `MUJOCO_GL=egl`, venv Python 3.8 → `run_batch6.sh` (24 runs, `--latch-z-offset 0.015`) → `analyze6_batch.py`.

**Cross-check (independent):** `analyze6_batch.py` and the Codex `audit_saved_state.py` (→ `audit_saved_state_result.json`) **agree**: 21/21 positive retained, 0/3 negative acquired, all listed checks pass. (analyze6 recomputes HOLD/latch drift from telemetry; the Codex audit read the per-run invariant flags.)

## Per-run

| run | latch_step | terminal_rise_m | P5_retained | P6_retained | status |
|---|---:|---:|:--:|:--:|---|
| always-enable_seed0 | 58 | 0.09948 | yes | yes | stable |
| always-enable_seed1 | 54 | 0.10030 | yes | yes | stable |
| always-enable_seed2 | 47 | 0.09921 | yes | yes | stable |
| always-hold_seed0 | — | −0.01022 | no | no | negative control (held) |
| always-hold_seed1 | — | −0.01022 | no | no | negative control (held) |
| always-hold_seed2 | — | −0.01022 | no | no | negative control (held) |
| approach_10_seed0 | 69 | 0.09438 | no | yes | **RECOVERED** |
| approach_10_seed1 | 58 | 0.09641 | yes | yes | stable |
| approach_10_seed2 | 63 | 0.10065 | yes | yes | stable |
| approach_60_seed0 | 119 | 0.09604 | yes | yes | stable |
| approach_60_seed1 | 110 | 0.10259 | yes | yes | stable |
| approach_60_seed2 | 113 | 0.10094 | yes | yes | stable |
| carrying_10_seed0 | 58 | 0.09948 | yes | yes | stable |
| carrying_10_seed1 | 54 | 0.10030 | yes | yes | stable |
| carrying_10_seed2 | 47 | 0.09921 | yes | yes | stable |
| carrying_60_seed0 | 58 | 0.09948 | yes | yes | stable |
| carrying_60_seed1 | 54 | 0.10030 | yes | yes | stable |
| carrying_60_seed2 | 47 | 0.09921 | yes | yes | stable |
| closure_10_seed0 | 67 | 0.09482 | no | yes | **RECOVERED** |
| closure_10_seed1 | 70 | 0.09454 | no | yes | **RECOVERED** |
| closure_10_seed2 | 57 | 0.10211 | yes | yes | stable |
| closure_60_seed0 | 117 | 0.09222 | no | yes | **RECOVERED** |
| closure_60_seed1 | 114 | 0.09517 | yes | yes | stable |
| closure_60_seed2 | 107 | 0.10197 | yes | yes | stable |

Checkpoint (external pretrained BC-RNN): `lift_ph_low_dim_epoch_1000_succ_100.pth`, SHA256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605` (reused, unmodified).

## Protocol status (honest)
`protocol6.md` was authored **retrospectively, after the batch** (absent at the 10:21 UTC reconciliation inspection; `protocol_paths_observed: []`). It is labeled retrospective and **not backdated**; the intervention value/design and criteria were **predeclared** in [SESSION_WRAP](../../docs/SESSION_WRAP_2026-11-04.md) §"Exact resume point."

## Limitations
- **Reused development seeds 0–2** — not held-out; not a reliability/qualification claim.
- **20 Hz sampled contact** — substep (500 Hz) behaviour is not directly observed; the `always-hold` negatives settle to −0.0102 m, confirming they never latch.
- **Carrying-phase HOLD controls did not engage**: the one-way latch fires in the approach phase (step 47–58), before the carrying window, so `carrying_10`/`carrying_60` share the `always-enable` trajectory per seed. The discriminative pre-/acquisition HOLD controls are the **approach** and **closure** pauses.
- **Initial-state parity** is initial-cube/end-effector match, not full sim-state equality.
- **`mujoco-py 2.1.2.14`** differs from the original robosuite source requirement (recorded in Phase 2); not attributed to the checkpoint alone.
- **Not a learned-EEG demonstration and not held-out qualification.**

## Distinction (keep separate)
- **Learned acquisition** — the BC-RNN policy acquiring the cube (external pretrained checkpoint).
- **Controller completion latch** (this phase) — one-way frozen completion goal after acquisition.
- **Scripted 5-person EEG reference** — already complete; **separate** from learned replay.
- **Exploratory S006R12 learned-policy EEG replay** — next, **not yet run**.
- **Held-out qualification** — later, separate.

## Preservation
- Source + scripts (SHA256 in `metrics6.json` and `audit_saved_state_result.json`): `latch6_rollout.py`, `run_batch6.sh`, `analyze6_batch.py`, `metrics6.json`.
- 24 raw traces + 2 smoke cases + logs: `/home/ubuntu/robomimic-native-phase6-20261104/` (`matched_batch/`, `_smoke/`, `batch6.log`).
- Phase 5 baseline (17/21, four failures) **preserved unmodified** in `results/robomimic_latch_2026-11-04/`.

## Publication / backup (honest status)
Local preservation is complete. At the time of writing (2026-11-04, 10:21 UTC reconciliation), **GitHub publish and the private Drive backup were blocked from this host**: `gh` was not logged into any GitHub host, `git ls-remote`/`git push` had no credentials, and no GitHub or Google Drive connector was connected. No credentials were changed or exposed. The canonical checkout remained dirty at `b3422a0` with the untracked Phase 6 files preserved; no reset/clean/force-push was performed.

**Update (2026-11-04, after reconciliation):** GitHub publish is now **complete** — after `gh auth login` (account `Mohnclizzy`), this 7-file compact set plus STATUS.md were committed and pushed to `codex/robot-learning-design` (tip `8810978`, verified by `git ls-remote` and the GitHub contents API) via isolated clean-clone integration; the canonical checkout remains dirty at `b3422a0` with no reset/clean/force-push. The private Drive backup **remains blocked** (no Google Drive/Box connector connected on this host); the raw archive `phase6_raw_archive_20261104.tar.gz` (SHA256 `ecb0f0b4e7aeff25b69fca652a98e9bbb498102ec430e1ab1ddff58e43fc39a0`) and its 79-entry `phase6_raw_SHA256SUMS.txt` manifest are staged locally under `/home/ubuntu/robomimic-native-phase6-20261104/` for off-host upload.

## Next bounded task (gate established)
The predeclared **exploratory 120-second S006R12 recorded-EEG learned-policy replay** with always-enable/HOLD controls, a declared label-independent reset schedule, causal timestamps and fixed budgets — distinct from the complete scripted 5-person EEG comparison and from held-out qualification.
