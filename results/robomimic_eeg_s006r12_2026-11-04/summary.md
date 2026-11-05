# S006R12 recorded-EEG learned-policy replay — result

**Scope:** the authorized bounded exploratory 120-second S006R12 raw/confidence/temporal
learned-policy replay, run under the established Phase 6 regime. See [protocol.md](protocol.md).

## Result
- **raw gate: 3/3 acquired and 3/3 retained** (`complete_retained`), acquired at steps
  **128–136**, terminal rise **0.0948–0.0988 m** (≥ 8 cm), **0 drops**, latch fired once at
  acquisition and held for the remaining 264–272 steps, all HOLD/latch invariants pass.
  **The recorded raw-gate EEG drives the learned policy to a successful retained lift.**
- **confidence gate: 0/3 acquired** (`complete_unretained`); the policy was enabled for only
  **24 of 400 steps** (a single ~1 s grasp bout) and HOLD for 376 steps → no lift.
- **temporal gate: 0/3 acquired** (`complete_unretained`); the policy was enabled for only
  **14 of 400 steps** (a single ~0.5 s grasp bout) and HOLD for 386 steps → no lift.
- **Overall: 3/9 acquired and 3/9 retained; 0 drops across all 9 runs; all 9 pass their
  HOLD + latch invariants.**

**Interpretation (honest):** this is a **within-subject, exploratory** result, **not held-out
qualification** and **not a live closed-loop demonstration**. It shows that **the recorded EEG
gate that actually produces sustained grasp intent (raw) drives the learned policy to a retained
lift**, while the **confidence and temporal gates are near-all-HOLD for this subject's recording**
(one brief grasp bout each), so they enable the policy too briefly to acquire. That is a signal
content / gate-conservativeness finding for S006, not a harness or policy defect — every run
completed its 400-step budget with clean control invariants.

## Per-run
| run | completion | acquire@ | latch@ | latch_steps | terminal_rise_m | term_contact | policy_calls | hold_steps |
|---|---|---:|---:|---:|---:|:--:|---:|---:|
| raw_seed0 | complete_retained | 136 | 136 | 264 | 0.09475 | yes | 50 | 86 |
| raw_seed1 | complete_retained | 128 | 128 | 272 | 0.09881 | yes | 42 | 86 |
| raw_seed2 | complete_retained | 130 | 130 | 270 | 0.09475 | yes | 44 | 86 |
| confidence_seed0 | complete_unretained | — | — | 0 | −0.01022 | no | 24 | 376 |
| confidence_seed1 | complete_unretained | — | — | 0 | −0.01022 | no | 24 | 376 |
| confidence_seed2 | complete_unretained | — | — | 0 | −0.01022 | no | 24 | 376 |
| temporal_seed0 | complete_unretained | — | — | 0 | −0.01022 | no | 14 | 386 |
| temporal_seed1 | complete_unretained | — | — | 0 | −0.01022 | no | 14 | 386 |
| temporal_seed2 | complete_unretained | — | — | 0 | −0.01022 | no | 14 | 386 |

All 9: `hold_invariants_pass=true`, `latch_invariants_pass=true`, `post_acquisition_loss=false`,
`contact_proxy_validated=true`, `latch_policy_call_violations=0`, `hold_rnn_counter_violations=0`.
Confidence/temporal settle to −0.01022 m (the held rest position), confirming they never latch.

## Provenance
- Checkpoint (external pretrained BC-RNN): `lift_ph_low_dim_epoch_1000_succ_100.pth`, SHA256
  `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605` (reused, unmodified).
- Decision trace: `outputs/isaac/eeg-batch-inputs/S006R12-decisions.jsonl`, SHA256
  `ca51e6abb5685911ceae33ac9e1806e4907c10faf2210d7d2315c408ddfbeb62`.
- Harness: `latch6_rollout.py --condition eeg --latch-z-offset 0.015` (reused unmodified).

## Reproducibility / determinism
Per (gate, seed) the outcome is **deterministic**: raw is identical across seeds 0–2
(retained, acq 128–136), confidence identical (no acquire), temporal identical (no acquire).
Runs are CPU/MuJoCo (torch 1.13.1+cpu); results are seed-deterministic, not wall-clock.

## Limitations
- **Within-subject / training-subject recording** (S006 is in the Phase 2 training set; held-out
  set is S001/S002/S003) — not a generalization or qualification claim.
- **Pre-decoded decision trace**, not live EEG; the decoder and its gates were not re-run here.
- **20 Hz sampled contact**; substep (500 Hz) behaviour is not directly observed.
- **Reused development seeds 0–2.**
- **`mujoco-py 2.1.2.14`** differs from the original robosuite source requirement (Phase 2 note).
- The confidence/temporal negatives are a **gate-conservativeness / signal-content** effect for
  this subject (one brief grasp bout each), not evidence about other subjects or time windows.

## Distinction (keep separate)
- **This:** exploratory recorded-EEG learned-policy replay (native BC-RNN, S006R12, within-subject).
- **Scripted 5-person EEG comparison** (`outputs/isaac/eeg-matched-5p-…`) — complete, separate.
- **Held-out qualification (S001/S002/S003)** — later, separate, not done here.
- **Live closed-loop** — out of scope, not done.

## Preservation
- 9 run dirs (telemetry + summary.json) + 8 batch logs + `run_eeg_batch.sh`:
  `results/robomimic_eeg_s006r12_2026-11-04/`.
- Phase 6 raw archive `/home/ubuntu/robomimic-native-phase6-20261104/` (tar.gz + SHA256SUMS)
  preserved unmodified; canonical checkout remains dirty at `b3422a0` (no reset/clean/pull).

## Publication / backup (honest status)
Local preservation is complete. Per Phase 6's honest status, **GitHub publish and the private
Drive backup are host-dependent**: Phase 6's GitHub publish succeeded (account `Mohnclizzy`,
branch `codex/robot-learning-design`) but this new compact set has **not** yet been committed or
pushed (no reset/clean was performed on the dirty canonical checkout), and the Drive backup has
no connected connector. Nothing here should be treated as "published" until committed + pushed
and (for the raw archive) backed up off-host.

## Next bounded task
**Held-out qualification** (S001/S002/S003) is the natural next step and is **separate** from this
exploratory replay — not authorized by this task and not run here. No new robot, no live EEG, no
Franka, and no further native work was started beyond this bounded replay.
