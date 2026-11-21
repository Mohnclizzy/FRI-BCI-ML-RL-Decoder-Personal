# S001R12 held-out EEG learned-policy replay — summary

## Result (all 9 runs)

**9 / 9 runs: `complete_retained` and `native_success_any: true`.** The Phase 2
learned policy completed the native table lift **and retained the cube** on the
held-out subject's recorded-EEG decision stream, across all three gates and all
three seeds, under the Phase 6 latch regime.

| Gate | Seeds completing (`complete_retained`) | `max_rise_m` (by seed) | Longest bilateral contact (steps) | First acquisition (step) |
|---|---|---|---|---|
| Raw | 3 / 3 | 0.0986 / 0.1014 / 0.0993 | 322 / 330 / 326 | 90 / 82 / 84 |
| Confidence | 3 / 3 | 0.0986 / 0.1014 / 0.0993 | 322 / 330 / 326 | 90 / 82 / 84 |
| Temporal | 3 / 3 | 0.0986 / 0.1014 / 0.0993 | 312 / 320 / 316 | 100 / 92 / 94 |

All runs: `terminal_rise_m ≥ 0.08` (max 0.1014), `terminal_contact` and
`terminal_retained` true, `post_acquisition_loss` false, `contact_proxy_validated`
true, latch invariants pass, HOLD invariants pass. Identical
`checkpoint_sha256 = 3ee222ca…` and `decisions_sha256 = 84bab7c6…` across all 9.
Seeds vary only slightly (initial cube placement); the outcome is stable.

## Contrast with the training-subject S006R12 replay

The **training**-subject S006R12 replay completed on **raw only** (3/3), with
confidence (0/3) and temporal (0/3) not completing. The **held-out** S001 replay
completes on **all three gates** (9/9). S001's trace also triggers far more
(`grasp_lift`: raw 227 / conf 170 / temp 128) than S006's (raw 120 / conf 6 /
temp 3), so the S001 result is a **higher-trigger-rate, all-gates** success — a
stronger held-out data point than the training-subject probe. This is consistent
with the latch regime being the condition under which the learned policy reaches
and retains a lift from a table start.

## Interpretation and honest limitations

- **Held-out subject, learned policy.** This is the first held-out-subject
  (not in the S006–S010 training set) replay of a recorded-EEG decision stream
  through the Phase 2 learned policy under the Phase 6 latch regime. It is a
  genuine held-out-subject generalization data point for the **replay/regime**.
- **Decoder caveat (preserved).** S001's decoder is **per-person calibrated** on
  S001's own runs 4/8, with run 12 held out. This is **not** an
  uncalibrated-new-person, live-intent, or grasp-label result. "Held-out"
  means held-out subject (vs S006–S010) and held-out run (12 vs 4/8).
- **Pre-decoded trace, offline.** No live EEG, no new robot, no Franka; native
  robosuite/MuJoCo on CPU. 20 s simulated horizon per run.
- **Not a full held-out set.** Only **S001 of S001/S002/S003** is executed (see
  below). This is **not** the full 3-subject held-out qualification.
- **Not** a live BCI, hardware-readiness, or clinical/physical-safety claim.
  Rest motion, drop rate, missed-command rate, and participant-level uncertainty
  are not measured here.

## S002 / S003 gap (why this is partial)

A broad search of this host (`find /home/ubuntu … *S001* *S002* *S003*`) found:

- **S001**: pre-decoded decision trace only — `/home/ubuntu/S001R12-decisions.jsonl`
  (the asset used above). No raw `.edf` present locally.
- **S002 / S003**: **nothing** — no decision trace, no raw `.edf`, in any
  checkout (`FRI-BCI-ML-RL-Decoder-Personal`, `FRI-project-8b9db42`, and the
  others). The repo's EEGMMIDB tree holds only S006–S010 raw recordings.

To complete the full 3-subject held-out qualification, S002 and S003 would each
require: (a) their raw EEG (e.g., PhysioNet EEGMMIDB S002R12/S003R12 + the
calibration runs 4/8), and (b) decoding with the frozen per-person CSP/LDA
calibration to produce their run-12 decision traces — a separate, larger
download + decode task, not yet performed.

## Reproduction

- Harness: `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py --condition eeg`
- Source trace: `/home/ubuntu/S001R12-decisions.jsonl`
  (`decisions_sha256 = 84bab7c63594bf91dbbeb2eb0b20d9beed6d80664457878fae21383569871566`)
- Checkpoint: `lift_ph_low_dim_epoch_1000_succ_100.pth`
  (`checkpoint_sha256 = 3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`)
- Batch driver: `run_heldout_s001_batch.sh`
- Per-run `summary.json` + `telemetry.jsonl` under
  `results/robomimic_eeg_s001r12_heldout_2026-11-05/{raw,confidence,temporal}_seed{0,1,2}/`
- Trace provenance: `docs/end_to_end_pilot_2026-11-02.md`; decoder split
  provenance: `docs/decoder_comparison_2026-11-01.md`.
