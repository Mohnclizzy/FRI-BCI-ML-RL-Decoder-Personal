# Approach table episode trace review — 2026-11-03

## Decision

The completed 100-update Approach run and five-episode screen require **no
repeat**. All five 12 s table episodes failed the frozen physical lift test.
The saved trace adds a useful diagnosis: the cube stayed at table height while
moving 3.7–6.8 cm horizontally, and sampled cube contact was overwhelmingly on
the right finger. This is evidence of a one-sided contact and displacement
failure, not proof of one unique cause. No new training or playback was launched
for this review. EEG remained separate.

## Inputs and checks

- Raw playback: `outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/ppo-telemetry.jsonl`.
- Recorder summary in the same directory: `ppo-telemetry.summary.json`.
- Checkpoint: the evaluation copy of `model_198.pt` in that directory;
  SHA256 `4ca3e23b639bda131ec55df8d8ac16551d547b0ae03cae0f5685d91f2597bde7`.
- Compact reproducible record: [`results/ppo_approach_trace_review_2026-11-03.json`](../results/ppo_approach_trace_review_2026-11-03.json).
- Offline collector: [`analysis/analyze_approach_table_trace.py`](../analysis/analyze_approach_table_trace.py).

The collector checked five distinct pre-action starts and five pre-reset
terminals, 1,800 policy actions, 7,200 physics samples, 1,440 valid contact
samples per episode, contiguous recorded 1/120 s sample timestamps, the saved
12 s duration and maximum rise for every episode, and zero recorded incomplete
episodes or playback errors. All five terminations were timeouts. The recorder
reported `environment_physics_step` for this run. The frozen threshold is a
cube rise of at least 0.08 m with both filtered object-contact sensors at
least 0.1 N for 0.5 s; no episode approached the height threshold.

| Episode | Max rise (m) | End horizontal cube shift (m) | Left/right contact samples | First right contact (s) | Bilateral samples | Max left/right sensor force (N) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000098 | 0.0482 | 0 / 123 | 0.775 | 0 | 0.00 / 5.39 |
| 2 | 0.000153 | 0.0447 | 0 / 168 | 0.683 | 0 | 0.00 / 6.09 |
| 3 | 0.000171 | 0.0367 | 0 / 226 | 0.925 | 0 | 0.00 / 5.33 |
| 4 | 0.000230 | 0.0679 | 3 / 185 | 0.558 | 1 | 1.58 / 6.57 |
| 5 | 0.000165 | 0.0459 | 0 / 397 | 0.583 | 0 | 0.00 / 8.67 |

The single bilateral sample in episode 4 was at 2.275 s, about 0.0083 s at
the logged cadence, far short of the 0.5 s dwell requirement. A contact-sample
count is not an uninterrupted-contact duration. All five cube rises were below
0.00024 m. The cube's horizontal shift is a more salient physical outcome than
its tiny vertical motion. A zero reward or a successful process exit adds no
pickup evidence.

## Mechanism clues and limits

The nine-action policy controls seven arm joints and two finger joints through
0.1-scaled relative position targets. At first right contact, actual finger
joint positions were 0.022–0.027 m. The saved target for the right finger was
−0.0070 to +0.0033 m, while the left target was 0.030–0.065 m. Over each
1,440-sample episode, the right target was negative
in 1,260–1,358 samples; the left target was negative in 79–656. This asymmetric
commanding is consistent with the observed right-only contact. The target is
not the actual finger position or an identified force mechanism.

The minimum recorded cube-to-`panda_hand` body-center distance was
0.118–0.125 m. That value cannot be used as fingertip clearance, because the
recorder did not save left/right fingertip body poses or collision geometry.
The later near-table progress policy's fixed-open diagnostic retained early
right-only contact even with both finger targets held at 0.04 m. Thus neither
the Approach target asymmetry nor a simple finger clamp is a sufficient
explanation across tasks. Those policies also have different starts and
weights, so their success rates must stay separate.

The contact values are filtered sensor magnitudes sampled at logged physics
steps. The reported 100% contact coverage means every recorded physics sample
had values; it does not prove contact continuity between samples. The review
does not establish parity with the scripted controller or the fused Newton
execution path. The original 100-update training budget, optimizer resume and
source provenance remain in [BREV_HANDOFF_2026-11-02.md](BREV_HANDOFF_2026-11-02.md).

## Next bounded action

Before allocating more PPO updates, inspect actual left/right fingertip body
poses and collision meshes at reset and around first right contact in a fresh,
short diagnostic. Record the fingertip-to-cube relative pose, arm path,
finger targets and actual positions, per-finger forces, contact coverage,
source/config/checkpoint hashes, and an explicit incomplete cap. Use the
existing fixed-open and ordinary progress traces as the comparison. Freeze
the diagnostic before launching it; do not interpret one-second contact
mechanism runs as pickup reliability. The corrected 120 s EEG scripted batch
is already complete in the Personal source of record and remains a separate
result. Keep the scripted enable/HOLD/resume reference intact.

Reproduce this offline review without a simulator:

```bash
python analysis/analyze_approach_table_trace.py \
  --telemetry outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/ppo-telemetry.jsonl \
  --summary outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/ppo-telemetry.summary.json \
  --checkpoint outputs/isaac/table-approach-eval-5x12s-seed8-20261102T202013Z/checkpoint/model_198.pt \
  --output results/ppo_approach_trace_review_2026-11-03.json
```

The record includes SHA256 hashes of the raw telemetry, summary, checkpoint
and collector. The raw data and checkpoint remain in ignored `outputs/`; their
off-host backup status is unverified in the original Brev handoff.
