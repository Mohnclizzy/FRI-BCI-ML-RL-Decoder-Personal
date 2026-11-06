# S002/S003 R12 held-out recorded-EEG learned-policy replay — summary

Run: `results/robomimic_eeg_s002s003_2026-11-06/` (2026-11-06)
Continuation of the frozen Stage-1 order after S001R12 and the Stage-1 addendum.

## Headline result (18 runs)

**12 / 18 runs completed and retained the cube (`complete_retained`, `native_success_any`).**
The 6 non-completing runs are **all S002 under the two strict gates**
(confidence, temporal), which produced too few `grasp_lift` commands for the
robot to reach the cube within the 20 s budget (`contact_none`).

The split is **by subject, not by seed**:

- **S003** (near-always-on decoder): **9 / 9 retained** — all three gates, all
  three seeds.
- **S002** (recall-limited decoder): **3 / 9 retained** — only the permissive
  **raw** gate (3/3); confidence (0/3) and temporal (0/3) do not complete.

| Subject | Gate | Retained | First acquisition (s0/s1/s2) | `max_rise_m` |
|---|---|---|---|---|
| S002 | raw | **3/3** | 298 / 201 / 294 | 0.0976 / 0.0976 / 0.1057 |
| S002 | confidence | 0/3 | none | 0.0 |
| S002 | temporal | 0/3 | none | 0.0 |
| S003 | raw | **3/3** | 90 / 82 / 84 | 0.0986 / 0.1014 / 0.0993 |
| S003 | confidence | **3/3** | 155 / 151 / 152 | 0.1024 / 0.1029 / 0.1097 |
| S003 | temporal | **3/3** | 191 / 186 / 187 | 0.1033 / 0.1084 / 0.1000 |

Every retained run: `terminal_retained: true`, `post_acquisition_loss: false`,
`latch_invariants_pass: true`, `hold_invariants_pass: true`, `max_rise_m ≥ 0.08`.
Identical `checkpoint_sha256 = 3ee222ca…` across all 18. Per-subject
`decisions_sha256` is constant across that subject's 9 runs
(S002 `51412e3d…`, S003 `9a790543…`; see `eeg_arm.tsv`).

## Calibration (per-person, run 12 held out)

Frozen per-person CSP/LDA on **each subject's own runs 4/8**, run 12 held out
(same protocol as S001R12 and S006R12). 9 channels (C3/Cz/C4/C1/C2/FC1/FC2/
CP1/CP2), 160 Hz, 8–30 Hz, 2.0 s window / 1.0 s offset, 4 CSP components.
Held-out run-12 evaluation, 15 REST + 15 HAND_IMAGERY windows:

| Subject | Balanced acc | Macro-F1 | REST recall (prec) | HAND recall (prec) | Rest false-activation |
|---|---|---|---|---|---|
| S002 | 0.733 | 0.722 | 0.933 (0.667) | **0.533** (0.889) | 0.067 (1/15) |
| S003 | 0.533 | 0.403 | **0.067** (1.000) | 1.000 (0.517) | **0.933** (14/15) |

- **S002 is recall-limited:** good at saying REST, but HAND recall is only 0.53.
  It predicts REST for 21/30 windows → few `grasp_lift` commands.
- **S003 is precision-limited (near-always-on):** HAND recall 1.0 but REST
  recall 0.067 — it predicts HAND for 29/30 windows (`rest_false_activation`
  0.933). A genuine, documented finding, not a tuning artifact.

`probabilities_are_calibrated: false` for both (preserved caveat from the
Stage-1 freeze).

## Decode + gate analysis (run 12 decision trace)

`fri_bci_sim.cli decode-replay` → `compare-commands` (threshold 0.7,
persistence 0.5 s) → per-subject `decisions.jsonl` (causality-validated,
zero violations). Gate = `grasp_lift` command when the gated rule fires:

| Subject | Trace windows | grasp_lift (raw / conf / temporal) |
|---|---|---|
| S002 | 485 | **150 / 59 / 35** |
| S003 | 493 | **483 / 365 / 308** |

## Controls bracket (S001R12 matched controls, same batch family)

From `results/robomimic_controls_s001r12_2026-11-06/controls.tsv`:

| Condition | Retained | First acquisition (s0/s1/s2) | `max_rise_m` |
|---|---|---|---|
| always-enable (100% duty) | 3/3 | 58 / 54 / 47 | 0.107 / 0.102 / 0.104 |
| always-hold (0% duty) | 0/3 (`contact_none`) | none | 0.0 |

These are the **upper** (always ready to grasp) and **lower** (never grasps)
bounds. The EEG-arm results sit inside this bracket.

## Interpretation

**The gate is a trigger-rate throttle; completion has a minimum trigger rate.**
Read together:

- S002 raw → **150** `grasp_lift` in-window → completes (3/3); S002 conf →
  **59**, temporal → **35** → `contact_none` (0/3).
- S003 raw/conf/temp → **483/365/308** → completes (9/9).

Full cross-subject picture, same metric (in-window `grasp_lift` commands,
run-12 held-out, frozen gates):

| Subject | windows | grasp_lift raw / conf / temporal | retained |
|---|---|---|---|
| S001 (held-out) | 493 | 227 / 170 / 128 | all gates (9/9) |
| **S002 (held-out)** | 485 | **150 / 59 / 35** | raw only (3/3) |
| **S003 (held-out)** | 493 | **483 / 365 / 308** | all gates (9/9) |
| S006 (training) | 485 | 120 / 6 / 3 | raw only (3/3) |

The completion threshold is a **minimum in-window `grasp_lift` count**:
the highest that fails is **59** (S002 confidence), the lowest that succeeds
is **120** (S006 raw); S001 temporal (**128**) also succeeds → the threshold
sits in **(59, 120]**. Recall-limited decoders (S002, S006) cross it only on
the permissive **raw** gate; S003 (over-active) and the well-calibrated S001
cross it on **every** gate.
This is a **recall-limited vs. precision-limited** split:

- **S002 (recall-limited)** needs the permissive **raw** gate to supply enough
  triggers; the stricter confidence/temporal gates over-filter the few real hand
  signals and the robot never acts.
- **S003 (precision-limited / over-active)** supplies enough triggers under
  **every** gate, so it completes on all three — it just acts earlier the more
  permissive the gate (raw step ~82–90, temporal ~186–191).

**Honest caveats (preserved):**

- **Held-out run, per-person decoder.** "Held-out" means held-out **run** (12
  vs calibration runs 4/8) on the **same subject**; decoders are per-person
  CSP/LDA. Not an uncalibrated-new-person or cross-subject result.
- **Pre-decoded offline trace, no live EEG, no new robot, no Franka.** Native
  robosuite/MuJoCo on CPU; 20 s (400-step) horizon per run.
- **Two of the three held-out subjects (S002, S003);** S001R12 is the third and
  was already executed in the prior session (9/9). Together these are the three
  held-out subjects, but each is a **single held-out run**, not a full held-out
  run battery.
- **Not** a live-BCI, hardware-readiness, drop-rate, rest-motion, or
  clinical/physical-safety claim. `probabilities_are_calibrated: false` for
  both decoders.
- The **pause-matched control** is intentionally **excluded** from this batch —
  the harness pause condition is inverted at `latch6_rollout.py` lines 85–86
  (documented in `docs/CONNECTED_BOTTLENECK_STAGE1_ADDENDUM_2026-11-06.md`).
  This batch is `--condition eeg` only, which does not exercise that path.

## Reproduction

- Per-person calibration: `run_calibrate_s002s003.py` (MNE `eegbci` PosixPath
  workaround; writes `S002_model/`, `S003_model/` with `metrics.json` +
  `test_predictions.json`).
- Decode: `PYTHONPATH=./simulation/src python -m fri_bci_sim.cli decode-replay
  --artifact S{S}_model/S{S}_model.joblib --edf S{S}R12.edf --output
  S{S}R12_estimates.jsonl --chunk-samples 16 --hop-s 0.25 --decode-delay-s 0`
- Gate: `PYTHONPATH=./simulation/src python -m fri_bci_sim.cli compare-commands --trace S{S}R12_estimates.jsonl
  --output S{S}R12_decisions.jsonl --threshold 0.7 --persistence-s 0.5`
- Rollout harness: `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
  --condition eeg --gate {raw|confidence|temporal} --decisions
  S{S}R12_decisions.jsonl --latch-z-offset 0.015 --out S{S}_{gate}_seed{n}`
  (`run_eeg_s002s003_batch.sh`, seeds 0–2 × 3 gates × 2 subjects = 18 runs).
- Aggregate: `aggregate.py` → `eeg_arm.tsv`, `eeg_arm_agg.md`.
- Checkpoint: `lift_ph_low_dim_epoch_1000_succ_100.pth`
  (`checkpoint_sha256 = 3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`).
- Controls: `results/robomimic_controls_s001r12_2026-11-06/` (`controls.tsv`,
  `aggregate.py`).
- Stage-1 correction + pause bug: `docs/CONNECTED_BOTTLENECK_STAGE1_ADDENDUM_2026-11-06.md`.
