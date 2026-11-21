# S001R12 held-out EEG learned-policy replay — protocol

## Purpose and status

Bounded held-out-subject qualification of the Phase 6 learned-policy
EEG replay. Subject **S001** is a **held-out** participant: it is **not** in the
Phase 2 / 5-person training set (S006–S010). The goal is to test whether the
same learned-policy replay regime that completed the **training**-subject
S006R12 replay also completes and retains the table lift when driven by a
**held-out** subject's recorded-EEG decision stream.

**Scope note (honest):** this is **1 of the 3 designated held-out subjects
(S001/S002/S003)**. Only S001 has an asset on this host (its pre-decoded
run-12 decision trace). S002 and S003 have **no** decision trace and **no**
raw EEG present locally, so the full 3-subject held-out qualification is
**partial** (see `summary.md`).

## Environment

Identical to the S006R12 training-subject replay: native robosuite/MuJoCo on
CPU (Python 3.8.20, torch 1.13.1+cpu, robosuite 1.2.0, robomimic 0.1.0). The
Brev Franka VM, Isaac Lab, and any live EEG were **not** started. **No new
robot, no live EEG, no Franka. Native robosuite/MuJoCo only.**

- Phase 2 venv + MuJoCo 2.1.0: `/home/ubuntu/robomimic-native-phase2-20261104/`
- Phase 2 learned policy (BC-RNN): `artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth`
  - `checkpoint_sha256 = 3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`
  - (same checkpoint used by the S006R12 replay)
- Harness: `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py --condition eeg`
- Batch driver: `run_heldout_s001_batch.sh`

## Source decision trace (pre-decoded, offline)

- File: `/home/ubuntu/S001R12-decisions.jsonl`
  - `decisions_sha256 = 84bab7c63594bf91dbbeb2eb0b20d9beed6d80664457878fae21383569871566`
  - 493 decision rows; window_end_s 2.0 → 125.0 s; episode/source `S001R12`
  - `grasp_lift` fires per gate: **raw = 227, confidence = 170, temporal = 128**
    (S001 triggers far more than the S006R12 trace, where raw = 120,
    confidence = 6, temporal = 3)
- **Provenance** (from `docs/end_to_end_pilot_2026-11-02.md`): held-out PhysioNet
  `S001R12.edf` recording; the **CSP artifact was calibrated for subject S001
  using runs 4 and 8**, with run 12 held out. Gates: raw argmax, confidence-only
  (0.7 threshold), temporal (0.7 threshold, 0.5 s persistence). Replay created
  493 probability estimates without reading labels.
- **Decoder caveat (honest, preserved from the pilot doc):** this is a
  **per-person calibrated** decoder (S001's own runs 4/8), with run 12 held
  out. It is **not** an uncalibrated-new-person / live-intent / grasp-label
  result. "Held-out" here means held-out **subject** (not in S006–S010) and
  held-out **run** (12 vs calibration runs 4/8).

## Regime and budget (matched to S006R12)

- Policy: Phase 2 learned BC-RNN (not the scripted skill).
- Switching: always-enable / HOLD, driven causally by the decision trace.
- Completion latch: **+0.015 m z** (Phase 6 regime), established in Phase 6.
- Horizon: native **400-step @ 20 Hz (20 s)** fixed budget, CPU/MuJoCo.
- Matched matrix: **3 gates × seeds 0–2 = 9 runs.**

## Command

```text
latch6_rollout.py --seed {0,1,2} --condition eeg --gate {raw,confidence,temporal}
  --decisions /home/ubuntu/S001R12-decisions.jsonl --latch-z-offset 0.015
  --out results/robomimic_eeg_s001r12_heldout_2026-11-05/<gate>_seed<seed>
```

## Success criteria

- `native_success_any: true`; `completion_state` `complete_retained`;
  `terminal_rise_m ≥ 0.08`; `terminal_contact` and `terminal_retained` true.
- Latch and HOLD invariants pass; no post-acquisition loss; contact proxy
  validated.
- **Honest labels (preserved):** native replay of a **held-out subject's**
  pre-decoded decision trace under the Phase 6 latch regime; **not** a live BCI,
  **not** an uncalibrated-new-person generalization claim, **not** a
  hardware/clinical-safety claim, and **not** the full 3-subject held-out set
  (S002/S003 absent).

## Out of scope

- S002 and S003 held-out replays (no local asset; would require obtaining their
  raw EEG and decoding with the frozen per-person decoder).
- Retraining the learned policy, new latch parameters, live EEG, a new robot,
  or a Franka.
