# Recorded EEG to Franka pilot — 2026-11-02

> Historical engineering pilot: subsequent audit found the simulator adapter
> followed measured joint positions during HOLD, allowing drift. Cube/contact
> measurements remain recorded observations, but this run does not validate
> physical HOLD or isolate the effect of filtering. Command scores below cover
> 125 seconds; physical outcomes cover 24 seconds and must not be compared as
> if they used the same horizon.

## Purpose and setup

This bounded pilot connected causal replay of the held-out PhysioNet
`S001R12.edf` recording to the existing scripted Franka table-lift skill in
Isaac Lab 3.0 / Newton on the Brev L4. The CSP artifact was calibrated for
subject S001 using runs 4 and 8. Replay created 493 probability estimates
without reading labels; a separate comparison pass applied raw argmax,
confidence-only (0.7 threshold), and temporal (0.7 threshold, 0.5 s
persistence) gates to the same trace. Each condition used a fresh fixed-scene
run with the same 24 s simulated budget. The object remained a free rigid body.

## Results

The separate annotation evaluator scored the saved command trace against EDF
T0/T1/T2 intervals after decoding. This does not feed labels to the decoder or
gate. Its per-condition results were:

| Gate | Rest time falsely active | Active time | Imagery-bout detection | Mean detection delay | Missed bouts | Evaluator state switches |
|---|---:|---:|---:|---:|---:|---:|
| Raw argmax | 30.0% | 45.2% | 93.3% | 0.96 s | 1 | 35 |
| Confidence only | 21.7% | 34.2% | 93.3% | 1.24 s | 1 | 45 |
| Temporal | 17.1% | 25.6% | 86.7% | 1.75 s | 2 | 33 |

The evaluator counts the implicit initial HOLD transition, so its switch totals
differ by one from the estimate-to-estimate counts below for raw and confidence.
The observed timeline covered all 125 s of the recording.

| Command condition | Active command time | Command switches | Maximum cube center z | Sustained lift with bilateral contact | Pass (>= 0.5 s) |
|---|---:|---:|---:|---:|---|
| Raw argmax | 20.0 s | 34 | 0.4045 m | 1.08 s (run summary); 1.0 s sampled | Yes |
| Confidence only | 16.6 s | 44 | 0.2821 m | 0 s | No |
| Temporal | 14.0 s | 33 | 0.2849 m | 0 s | No |

Cube baseline center was 0.2799 m. A sustained lift requires >=0.08 m rise and
both finger contact forces >=0.1 N continuously for >=0.5 s. Raw reached a
maximum left/right contact force of 9.77/8.33 N. The confidence run had contact
peaks but no measured lift. Temporal filtering reduced active command time and
switches relative to raw on this trace, while confidence-only increased
switches. These are one recording and one scene per condition; they do not
establish that raw control is generally better or that a filter improves task
success. The outcome is consistent with a latency/coverage tradeoff that needs
matched repeated trials on additional held-out recordings.

## Interpretation and limitations

This establishes a first recorded-EEG-to-simulated-Franka execution: model
probabilities passed through the named gate, the resulting request controlled
the skill timeline, and telemetry recorded the Franka, cube, and finger
contacts. It uses a scripted arm skill as the low-level controller; it does not
demonstrate EEG control of the PPO policy. A separate measured PPO follow-up
found 5/5 lifts from privileged pre-grasp starts and 0/5 from table starts after
checkpoint transfer; see [the PPO evaluation](ppo_lift_evaluation_2026-11-02.md).
Those PPO results do not alter the scripted-skill table-start EEG comparison.

The pilot uses one previously inspected recording, uncalibrated probability
cutoff, and no scene/seed replication. Human-label timing is used only for
offline annotation scoring, not decoder input or command generation. It is not
evidence of a live BCI, new-subject generalization, hardware readiness, or
clinical/physical safety. Rest motion, drop rate, missed-command rate, and
participant-level uncertainty require additional validated evaluator runs.

## Reproduction artifacts

- Local ignored outputs: `outputs/isaac/S001R12-continuous.jsonl`,
  `outputs/isaac/S001R12-decisions.jsonl`,
  `outputs/isaac/S001R12-annotation-eval.json`, and the three
  `outputs/isaac/eeg-demo-*.jsonl` telemetry files.
- Simulator source used remotely: `simulation/scripts/controlled_lift.py`
  with `--decisions` and `--condition`; SHA256
  `C24CA67701FA4FCDA83B9123C0BA97BF06616228C88FC73F06C7A8F9C09D186B`.
- Per-condition remote logs and telemetry are under
  `/home/ubuntu/FRI-project-8b9db42/outputs/isaac/eeg-demo-{raw,confidence,temporal}/`.
- Decoder metrics and split provenance: `docs/decoder_comparison_2026-11-01.md`
  and `outputs/decoder_validation_001/`.
- Comparison figure: `outputs/isaac/S001R12_franka_comparison.png`.
