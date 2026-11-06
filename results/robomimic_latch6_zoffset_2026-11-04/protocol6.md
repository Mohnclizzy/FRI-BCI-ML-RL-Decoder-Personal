# Phase 6 protocol — completion-equilibrium goal intervention (+0.015 m world-z at acquisition)

- Phase: 6 (single-intervention test on top of the Phase 5 one-way completion latch)
- Date executed / analyzed: 2026-11-04
- Status: **complete** (24/24 matched seed-0–2 controls ran; analyzed; independently cross-checked)
- Attribution: executed by **Hermes (execution owner)**. Per `docs/agents/CODEX_HERMES_RECONCILIATION_2026-11-04.md`, the Codex CLI Phase 6 session was inspection-only and interrupted; all Phase 6 files in this directory are attributed to Hermes.

---

## ⚠ PROTOCOL STATUS — honest disclosure (retrospective, not backdated)

This standalone protocol file was authored **retrospectively, after the 24-run batch was executed and analyzed.**

- `analyze6_batch.py` referenced `protocol6.md` before it existed.
- The 2026-11-04 reconciliation inspection (10:21 UTC) recorded this file as **absent** — the companion audit result `audit_saved_state_result.json` reports `protocol_paths_observed: []`.
- This file is therefore a **retrospective reconstruction** of the executed protocol, written to fill the record. It is **NOT** a pre-frozen declaration and is **NOT backdated**.

Per the owner steering (2026-11-04): *"If no pre-run protocol exists, disclose that gap and document the reconstruction as retrospective; never backdate it."* That is exactly what this file does.

## Recovered pre-run declaration (what WAS declared before the batch)

The **intervention value and design** were genuinely **predeclared** in `docs/SESSION_WRAP_2026-11-04.md`, §"Exact resume point" (written before the batch):

> "In a fresh Phase 6 directory, freeze and test **one** higher completion equilibrium goal intervention, motivated by the four contact-stable height-margin failures. The intended pilot is **+0.015 m in world z once at acquisition, with a fixed goal thereafter**; this was proposed but **not run**. Judge the same 8 cm terminal rise, bilateral contact, drops, held targets and fixed 400-step deadline. Preserve Phase 5 as the unmodified baseline."

So the following were declared before execution: the **+0.015 m world-z** value; the **once-at-acquisition, frozen-thereafter** design; the **single-intervention** scope; the **fresh Phase 6 directory**; the **8 cm / bilateral / drops / held-targets / 400-step** criteria; and the **Phase-5-unmodified-baseline** rule. What was **not** predeclared as a standalone frozen file is the document that follows.

---

## Intervention (exact, single change)

Add **+0.015 m in world z** to the end-effector **latch-goal position**, applied **once at latch capture**, with the goal **frozen thereafter**. This raises the held completion equilibrium so the four Phase 5 height-margin failures keep ≥ 8 cm terminal rise.

The one behavioural change relative to Phase 5 (`latch_rollout.py`), at line 111 of `latch6_rollout.py`:

```python
# Phase 5 (latch_rollout.py:110)
ctrl.update(); latch_goal_pos=np.array(ctrl.ee_pos,copy=True); latch_goal_ori=np.array(ctrl.ee_ori_mat,copy=True)
# Phase 6 (latch6_rollout.py:111)  <-- +0.015 m world-z at acquisition, frozen thereafter
ctrl.update(); latch_goal_pos=np.array(ctrl.ee_pos,copy=True)+np.array([0.0,0.0,float(a.latch_z_offset)]); latch_goal_ori=np.array(ctrl.ee_ori_mat,copy=True)
```

Supporting (non-behavioural) additions:
- CLI flag `--latch-z-offset` (default `0.015`); `--latch-z-offset 0.0` reproduces Phase 5 exactly (verified by smoke A/B on `closure_10_seed0`).
- `latch_z_offset_m` added to the run record (audit trail).

**Single-change rule:** the full `diff` between Phase 5 and Phase 6 is exactly these hunks (docstring + CLI flag + the +z lift + the record field). All other logic is byte-identical. The latch is one-way: once acquired, `target_pos` is held at the frozen goal through step 400 regardless of requests.

## Runs / matrix (24)

`run_batch6.sh`, all with `--latch-z-offset 0.015`:

| Block | Seeds | Conditions | Runs |
|---|---|---|---|
| Negative control | 0,1,2 | `always-hold` | 3 |
| Positive baseline | 0,1,2 | `always-enable` | 3 |
| Pause controls | 0,1,2 | `pause` × {approach, closure, carrying} × {10, 60 steps} | 18 |
| **Total** | | | **24** |

Identical matrix to Phase 5, enabling a direct Phase 5 ↔ Phase 6 comparison on the same seeds.

## Gate and invariants (unchanged from Phase 5)

- **Acquisition:** ≥ 10 consecutive sampled bilateral-contact states **and** ≥ 0.08 m rise.
- **Retention (gate):** retained at step 400 (fixed 400-step deadline).
- **Invariants:** `bilateral_contact == builtin_grasp` throughout; `time_s = step/20` (20 Hz); latch goal pose/orientation/gripper **frozen** once locked; **no policy call / no RNN counter advance** during locked intervals.
- **Positive gate:** all positive controls acquired AND retained. **Negative gate:** all `always-hold` controls stay unacquired.
- **Failure policy:** any invariant breach → the run does not count as retained; a gate failure stops the predeclared EEG replay.

## Command (as executed)

```bash
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
bash results/robomimic_latch6_zoffset_2026-11-04/run_batch6.sh   # 24 runs
"$PYTHON" results/robomimic_latch6_zoffset_2026-11-04/analyze6_batch.py
```

## Execution order

1. **Smoke A/B** on `closure_10_seed0`: `--latch-z-offset 0.0` → `retained=False` (reproduces the Phase 5 failure, terminal rise 0.0798 m); `--latch-z-offset 0.015` → `retained=True` (0.0948 m). Confirms the offset is the operative variable and the script is otherwise faithful to Phase 5.
2. **Full 24-run batch** (`matched_batch/`), then `analyze6_batch.py`.

## Analysis

- `analyze6_batch.py` — reuses the Phase 5 gate/invariant logic, adds the Phase 5 ↔ Phase 6 per-seed comparison. Wrote `metrics6.json`.
- **Independent cross-check:** `docs/agents/reconciliation_2026-11-04/audit_saved_state.py` → `audit_saved_state_result.json`. Both agree: **21/21 positive retained, 0/3 negative acquired, all listed checks pass.** (Note: `analyze6_batch.py` recomputes HOLD/latch goal drift from telemetry; the Codex audit read the per-run `all_invariants_pass` summary flags.)
