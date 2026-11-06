# S001R12 matched controls — summary

Run: `results/robomimic_controls_s001r12_2026-11-06/` (2026-11-06)
Phase-B matched controls for the frozen Stage-1 order (after the S002/S003
decode + EEG arm). Same `latch6_rollout.py` harness, Phase-2 BC-RNN checkpoint,
Phase-6 completion latch; `--latch-z-offset 0.015`.

## Result (6 runs)

| Condition | Duty | Retained | First acquisition (s0/s1/s2) | `max_rise_m` |
|---|---|---|---|---|
| always-enable | 100% (grasp always) | **3/3** `complete_retained` | 58 / 54 / 47 | 0.107 / 0.102 / 0.104 |
| always-hold | 0% (grasp never) | **0/3** `contact_none` | none | 0.0 |

- **always-enable = upper bound:** the robot is always ready to grasp, so it
  acquires and retains the cube in ~47–58 steps on every seed.
- **always-hold = lower bound:** the robot never grasps → never contacts the
  cube (`contact_none`) on every seed.

These two conditions bracket the EEG-arm held-out results (S002/S003) in the
same batch family: S003's permissive-gate runs (raw step ~82–90) sit close to
always-enable; S002's strict-gate runs (`contact_none`) match always-hold.

## Scope / caveats

- **Matched controls, not the EEG signal.** `always-enable` / `always-hold` are
  the two extreme command streams — no EEG is decoded or replayed. Their purpose
  is to bound the completion metric the EEG arm is measured against.
- **`always-hold` is a sanity control, not a result claim:** it confirms the
  harness can fail to complete (0 duty cycle → no contact).
- **The pause-matched control is intentionally excluded** from this batch —
  the harness pause condition is inverted at `latch6_rollout.py` lines 85–86
  (see `docs/CONNECTED_BOTTLENECK_STAGE1_ADDENDUM_2026-11-06.md`). These two
  controls are `--condition control` / not the `pause` path, so they are
  unaffected by that bug.
- Native robosuite/MuJoCo on CPU; 20 s (400-step) horizon per run. Not a
  live-BCI, hardware-readiness, or clinical claim.

## Reproduction

- `run_controls_batch.sh` → 2 conditions × 3 seeds = 6 runs.
- `aggregate.py` → `controls.tsv` (subject / condition / seed /
  completion_state / first_acquisition_step / max_rise_m).
- Harness: `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
  --seed {0,1,2} --out {condition}_seed{n} --condition control
  --latch-z-offset 0.015` (CKPT hardcoded to the Phase-2 checkpoint
  `checkpoint_sha256 = 3ee222ca…`).
- Per-run `summary.json` + `replay.mp4` in each `{condition}_seed{n}/`.

> **Note on the batch marker:** `batch_run.log` ends with
> `CONTROLS_S001_BATCH_COMPLETE runs=15`. That number is a **cosmetic counter bug**
> in `run_controls_batch.sh` L55 — the `\\.` filters sit inside a double-quoted
> `$(…)`, so they match nothing and `grep -cv '^aggregate'` counts every entry except
> `aggregate.py`. The **true run count is 6** (the 6 rollout dirs / 6 `controls.tsv`
> rows above: 3 always-enable + 3 always-hold).
