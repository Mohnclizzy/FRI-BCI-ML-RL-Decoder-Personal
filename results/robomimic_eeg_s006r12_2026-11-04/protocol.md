# S006R12 recorded-EEG learned-policy replay — protocol (authorizing scope)

**Status:** executed 2026-11-04. See [summary.md](summary.md) for the result.

## What this is
One **bounded, exploratory** replay of the **native learned policy** (the external
pretrained BC-RNN checkpoint, `lift_ph_low_dim_epoch_1000_succ_100.pth`) driven by the
**recorded** S006R12 EEG decision trace, under the **established** Phase 6 physical-control
regime. It is the authorized next task after the Phase 6 physical-control gate was established
(21/21 positive retained, 0/3 negative acquired).

## Authorization basis
- `docs/bounded_replay_evaluation.md` — bounded-replay contract (120-second window).
- `docs/agents/HERMES_AUTONOMOUS_ENGINEERING_DIRECTIVE_V2.md` — "once the authorized
  physical-control gate is established, the existing next task is one exploratory 120-second
  S006R12 raw/confidence/temporal learned-policy replay with always-enable/HOLD controls, a
  declared label-independent reset schedule, causal timestamps and fixed budgets."
- `STATUS.md` and `docs/SESSION_PAUSE_2026-11-04_2110UTC.md` — next-task authorization.

## Scope and explicit non-goals
- **Exploratory only.** Not held-out qualification. S006R12 is subject **S006** (a Phase 2
  **training** subject; held-out set is S001/S002/S003) and the result is a within-subject
  developmental probe, not a generalization claim.
- **Not a live closed-loop demonstration.** The EEG is a **recording** (S006R12.edf); the
  decisions were pre-decoded offline into `S006R12-decisions.jsonl` (raw / confidence / temporal
  gates) and replayed causally.
- **Separate from the complete scripted 5-person EEG comparison** (`outputs/isaac/eeg-matched-5p-…`)
  — this drives the **learned** policy, not a scripted control sequence.
- **No new robot, no live EEG, no Franka.** Native robosuite/MuJoCo only.

## Harness and controls
- Harness: `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py` with
  `--condition eeg` (the Phase 6 harness, reused unmodified).
- **always-enable / HOLD switching:** `eeg` condition = policy running (enable) while the
  decoded trace says `grasp_lift`; frozen HOLD (pose/gripper frozen, no policy call, no RNN
  advance) while it says `hold`.
- **Completion latch** (+0.015 m world-z), one-way, fires once at acquisition — the Phase 6
  established gate regime. This is what makes retention work (Phase 4 without it: 0/3 retained;
  Phase 6 with it: 21/21).
- **Physical outcome checks:** terminal rise ≥ 8 cm, bilateral contact, post-acquisition
  retention (no drop), 400-step deadline.

## Causal timing (enforced, not assumed)
`DecisionReplay` (`simulation/src/fri_bci_sim/executor.py`) rejects at load:
- `issued_at_s < available_at_s` → causality error (executor.py:116–117);
- `window_end_s`/`available_at_s`/`issued_at_s` not strictly increasing → out-of-order error
  (executor.py:118–120); and a max-age staleness check at issue (executor.py:142).
A **single episode uses a single source trace** (executor.py:115), so no cross-episode leakage.

## Fixed budget
- **Native 400-step horizon** @ 20 Hz = **20 s** simulated per run (the fixed native rollout
  budget; the "120-second" window is the **recording** that produced the decision trace).
- **3 gates × seeds 0–2 = 9 runs**, each deterministic per seed.
- Fixed checkpoint, fixed task (`lift_ph_low_dim`), fixed latch offset (0.015 m).

## Decision-trace statistics (the EEG signal content)
| gate | decisions | grasp | first-20-s grasp | longest grasp bout (whole) |
|---|---:|---:|---:|---:|
| raw | 485 | 120 (25%) | 45/73 | 5.5 s (bouts 4.2–9.5, 11.8–14.8, 18.0–23.5 s) |
| confidence | 485 | 6 | 5/73 | 1.0 s (single 12.8–13.8 s bout) |
| temporal | 485 | 3 | 3/73 | 0.5 s (single 13.2–13.8 s bout) |

The raw gate is active enough in the first 20 s (sustained 3–5.5 s grasp bouts) to drive the
policy to a lift (acquires in ~2.6–3.2 s in Phases 5/6). The confidence/temporal gates are
near-all-HOLD for this subject (one brief bout each), so they enable the policy too briefly to
acquire — expected, not an error.

## Provenance
- Checkpoint: `lift_ph_low_dim_epoch_1000_succ_100.pth`, SHA256
  `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605` (reused, unmodified).
- Decision trace: `outputs/isaac/eeg-batch-inputs/S006R12-decisions.jsonl`, SHA256
  `ca51e6abb5685911ceae33ac9e1806e4907c10faf2210d7d2315c408ddfbeb62` (recorded in every summary).
- Env: `MUJOCO_PY_MUJOCO_PATH`/`LD_LIBRARY_PATH` → `runtime/mujoco210`, `MUJOCO_GL=egl`,
  venv Python 3.8 (mujoco-py 2.1.2.14, robosuite 1.2.0, robomimic 0.1.0, torch 1.13.1+cpu).

## Preservation
Phase 6 raw archive `/home/ubuntu/robomimic-native-phase6-20261104/` and the dirty canonical
checkout (`codex/robot-learning-design` @ `b3422a0`) are preserved unmodified; this run added
only new untracked files under `results/robomimic_eeg_s006r12_2026-11-04/`. No git
stash/clean/reset/pull.
