# Five-person EEG command replay before matched Franka runs — 2026-11-02

> **Physical-control validation failed on subsequent audit.** The original
> always-HOLD trial moved the hand about 0.141 m because the adapter replaced
> its target with measured joint positions every step. The negative control
> checked only absence of a cube lift. The measurements below are retained as
> historical debugging evidence; the claim that both physical controls passed
> is withdrawn. Command-only annotation scores remain usable, but effects on
> physical rest motion and task completion require a corrected executor rerun.

## Question

Does command filtering reduce rest-period activation across the five-person
PhysioNet confirmation cohort, and what is the coverage/delay tradeoff in the
same 24-second interval used by the Franka pilot?

## Protocol

Used each existing per-person CSP+LDA artifact for S006–S010, calibrated on that
person's runs 4/8 and replayed on run 12. The decoders received causal EEG chunks
only; annotation labels were read only by the post-hoc evaluator. Threshold 0.7,
0.5 s persistence, 1.0 s maximum estimate age, and the first 24.0 seconds were
fixed to match the existing one-person robot run horizon. No model was refit and
no threshold was tuned from these outcomes. These participants' decoder results
were already inspected in the earlier confirmation analysis, so this is an
additional exploratory command replay, not fresh confirmation.

## Per-participant command results

False activation is the fraction of annotated rest time with an active command.
Imagery detection counts the three imagery bouts touched by an active command;
mean delay is among detected bouts only. Active time and switches are command
trace metrics, not physical robot outcomes.

| Person | Gate | Rest false-active | Imagery bouts detected | Missed bouts | Mean detection delay (s) | Active time (s) | Switches |
|---|---|---:|---:|---:|---:|---:|---:|
| S006 | Raw | 39.0% | 3/3 | 0 | 0.07 | 14.7 | 8 |
| S006 | Confidence | 0.0% | 1/3 | 2 | 0.50 | 1.2 | 2 |
| S006 | Temporal | 0.0% | 1/3 | 2 | 1.00 | 0.7 | 2 |
| S007 | Raw | 29.4% | 3/3 | 0 | 0.60 | 10.0 | 14 |
| S007 | Confidence | 11.9% | 2/3 | 1 | 1.15 | 5.3 | 8 |
| S007 | Temporal | 5.6% | 2/3 | 1 | 1.65 | 3.5 | 6 |
| S008 | Raw | 71.5% | 3/3 | 0 | 0.00 | 20.3 | 7 |
| S008 | Confidence | 61.8% | 3/3 | 0 | 0.00 | 18.0 | 15 |
| S008 | Temporal | 47.2% | 3/3 | 0 | 0.07 | 14.2 | 13 |
| S009 | Raw | 61.8% | 3/3 | 0 | 0.30 | 17.7 | 8 |
| S009 | Confidence | 50.4% | 3/3 | 0 | 0.57 | 14.1 | 8 |
| S009 | Temporal | 43.1% | 3/3 | 0 | 0.73 | 12.1 | 8 |
| S010 | Raw | 22.8% | 2/3 | 1 | 0.85 | 6.3 | 8 |
| S010 | Confidence | 21.1% | 1/3 | 2 | 0.00 | 4.4 | 8 |
| S010 | Temporal | 14.6% | 1/3 | 2 | 0.50 | 2.6 | 6 |

| Gate | Mean rest false-active | Mean imagery detection | Median active time (s) | Mean switches |
|---|---:|---:|---:|---:|
| Raw | 44.9% | 93.3% | 14.7 | 9.0 |
| Confidence | 29.0% | 66.7% | 5.3 | 8.2 |
| Temporal | 22.1% | 66.7% | 3.5 | 7.0 |

The pattern is a coverage tradeoff: both filters reduced mean rest activation and
switches, while each detected fewer imagery bouts than raw and reduced active
time. Temporal filtering produced the lowest mean and median rest activation,
but did not win on every individual subject. These participant means are
exploratory and omit participant-level uncertainty; do not call them command
accuracy or robot task success.

## State and next step

This preflight proved the same three gating conditions can be formed on five
additional recorded EEG streams with one fixed parameter set. The matched
Franka batch is complete below; both controls passed their expected outcomes.

## Artifacts

Ignored local outputs preserve `S006R12`–`S010R12` continuous estimates,
three-gate command traces, manifests, and first-24-second annotation reports in
`outputs/isaac/`. Decoder artifacts are under
`outputs/decoder_validation_001/confirmation_models/`. The existing
`docs/end_to_end_pilot_2026-11-02.md` remains the separate pilot evidence for S001R12.
## Matched Franka results for S006–S010

Five participant-specific run-12 EEG streams were replayed through the same
scripted Franka skill, with raw, confidence, and temporal gates. All 15 episodes
used the fixed table-cube reset and a 24 s simulator-time budget. The two robot
controls ran first in the same scene: always-HOLD passed (cube z remained
0.27989 m; bilateral contacts 0 N), and scripted-positive passed (cube reached
0.45890 m; longest sustained bilateral-contact interval 5.08 s). Since both
controls behaved as required, the following EEG outcomes are interpretable.

Lift success requires cube center rise >=0.08 m and bilateral contact force
>=0.1 N continuously for >=0.5 s, measured by the simulator. `Active skill`
is accumulated simulator time for which the selected gate requests
`GRASP_LIFT`; the full scripted motion sequence requires about 21 s of active
skill time.

| Person | Gate | Cube rise (m) | Sustained bilateral lift (s) | Active skill (s) | Lift pass |
|---|---|---:|---:|---:|---|
| S006 | Raw | 0.0000 | 0.00 | 14.7 | No |
| S006 | Confidence | 0.0000 | 0.00 | 1.2 | No |
| S006 | Temporal | 0.0000 | 0.00 | 0.7 | No |
| S007 | Raw | 0.0085 | 0.00 | 10.0 | No |
| S007 | Confidence | 0.0104 | 0.00 | 5.3 | No |
| S007 | Temporal | 0.0000 | 0.00 | 3.5 | No |
| S008 | Raw | 0.1442 | 1.42 | 20.2 | Yes |
| S008 | Confidence | 0.0391 | 0.00 | 17.9 | No |
| S008 | Temporal | 0.0036 | 0.00 | 14.1 | No |
| S009 | Raw | 0.0266 | 0.00 | 17.7 | No |
| S009 | Confidence | 0.0000 | 0.00 | 14.1 | No |
| S009 | Temporal | 0.0006 | 0.00 | 12.1 | No |
| S010 | Raw | 0.0108 | 0.00 | 6.3 | No |
| S010 | Confidence | 0.0001 | 0.00 | 4.4 | No |
| S010 | Temporal | 0.0000 | 0.00 | 2.6 | No |

Task completion was 1/5 for raw and 0/5 for both filtered conditions. This
matches the command-replay tradeoff in the preceding section: filters lowered
rest false activation and active command time, but the resulting command stream
usually did not provide enough sustained skill time to complete the 21 s motion
sequence inside the fixed 24 s episode. The only measured EEG-driven lift was
S008 raw; it rose 0.144 m and maintained bilateral contact for 1.42 s. This is
a small, already-inspected cohort and one fixed scene per condition. It supports
an exploratory finding that the tested temporal gate reduced unwanted rest
activation at a task-completion cost under this 24 s budget; it does not support
a population-level superiority claim. The filter may behave differently with a
longer task budget or a different command/skill interface, which would need a
new frozen protocol.

This result points to command availability and task duration as the next
experiment-design issue; it does not establish that PPO is needed or that the
scripted Franka skill is unreliable. The scripted-positive control lifted
in the single matched control, and the pre-existing PPO table-transfer
pilot remains a separate 0/5 result.

## Reproduction

Batch command: `simulation/cloud/run_eeg_matched_batch.sh`. Inputs were the
saved `S006R12`–`S010R12` decision traces. The runner performs and physically
checks always-HOLD and scripted-positive before it launches the 15 EEG episodes,
then stops on any invalid process or missing telemetry. Source, command files,
input-trace hashes, case logs, TSV exit codes and JSONL simulator telemetry are
preserved in the ignored local folder
`outputs/isaac/eeg-matched-5p-20261102T054610Z/`. Runtime used Isaac Lab v3.0.0,
Newton, and an NVIDIA L4. The instance was stopped after copying the results.
## Post-hoc 60-second budget sensitivity

After seeing that most 24-second episodes supplied fewer than the skill's roughly
21 active seconds, the same five saved EEG traces were rescored over their first
60 seconds and a separate 60 s simulator batch was prepared. This extension was
chosen after inspecting the 24 s outcomes, so it is a post-hoc sensitivity
analysis. The 60 s annotation-only command summaries are:

| Gate | Mean rest false-active | Mean imagery-bout detection | Mean active-command time | Mean delay among detections |
|---|---:|---:|---:|---:|
| Raw | 41.7% | 88.6% | 27.4 s | 0.69 s |
| Confidence | 26.1% | 62.9% | 16.0 s | 1.07 s |
| Temporal | 19.9% | 51.4% | 11.9 s | 1.32 s |

The matching 60 s Franka batch is pending. It uses unchanged models, thresholds,
scene, command executor and success criterion; only the simulator budget is
longer. It cannot serve as fresh confirmation because these participant
recordings and decoder results have already been inspected.
