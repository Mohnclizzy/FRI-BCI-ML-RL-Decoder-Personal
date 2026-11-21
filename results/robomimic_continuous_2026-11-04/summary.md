# Native BC-RNN continuous HOLD/resume bounded result — 2026-11-04

The official pretrained robomimic Lift PH low-dimensional BC-RNN was used without retraining or checkpoint edits. The adapter runs in the preserved robosuite `offline_study` environment at 20 Hz control, 500 Hz physics and a fixed 400-action/20-second deadline. The checkpoint SHA256 is `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`. This is a simulated, exploratory native-policy diagnostic. The earlier 5/5 acquisition HOLD with a continually closing gripper is a separate diagnostic and is not included in these results.

## Contract and scoring

REST/HOLD freezes the measured end-effector pose as an **absolute OSC_POSE goal** and preserves the Panda gripper's current actuator command with zero gripper input. The controller and physics continue, and no learned-policy call occurs during HOLD. The default resume freezes the BC-RNN hidden state and 10-call recurrent counter through HOLD; the next call receives the current simulator observation. The model has no separate observation-history queue. A second, bounded intervention resets hidden state and counter on resume. Completion is evaluated once at the fixed terminal step and remains terminal. The adapter's Python 3.8 replay port exactly matched the existing `DecisionReplay` contract on 400 20 Hz decisions for each of raw, confidence and temporal gates using the same S006R12 recording trace. No EEG rollout was executed because the physical positive-control requirement below failed.

The [frozen protocol](protocol.md) sets acquisition at bilateral sampled fingerpad/cube contact plus 0.08 m rise from each episode's initial cube center; sustained contact is at least 10 consecutive 20 Hz samples. Terminal retention also requires no sampled contact or height loss after acquisition. Exact goal and gripper invariance tolerances are 1e-9, and HOLD physical drift tolerances are 0.035 m end effector and 0.025 m cube relative to each pause entry. The first always-HOLD software probe was excluded before freezing these limits. A 20 Hz contact sample cannot rule out breaks inside a 0.05-second control interval. Native reward and process exit were not used as success evidence.

## Matched 24-run result

All three seeds (0–2) had exactly matched starting states and action/state prefixes before their scheduled pauses. The first 24-run batch used 0.5-second and 3-second pauses at steps 15 (approach), 35 (nominal closure), and 70 (carrying). Step 70 was observed to begin with bilateral contact and 0.16–0.36 m cube rise for all three seeds. Step 35 preceded contact for all seeds and preceded actual gripper closure for seed 0; that phase-label limitation led to the corrective diagnostic below.

| Condition | Acquired | Retained at 400 | Post-acquisition loss |
|---|---:|---:|---:|
| Always HOLD | 0/3 | 0/3 | 0/3 (no acquisition) |
| Uninterrupted learned policy | 3/3 | 0/3 | 3/3 |
| Approach pause, 0.5 s | 3/3 | 0/3 | 3/3 |
| Approach pause, 3 s | 3/3 | 0/3 | 3/3 |
| Step-35 pause, 0.5 s | 3/3 | 1/3 | 2/3 |
| Step-35 pause, 3 s | 3/3 | 0/3 | 3/3 |
| Carrying pause, 0.5 s | 3/3 | 0/3 | 3/3 |
| Carrying pause, 3 s | 3/3 | 2/3 | 1/3 |

Every run completed 400 actions; direct bilateral contact agreed with robosuite's grasp check at every sampled state. All HOLD goals and gripper commands were invariant, policy/RNN counters did not advance in HOLD, and all measured drift stayed inside frozen limits. The largest HOLD end-effector and cube drifts were 0.0289 m and 0.0138 m, respectively. Every pause resumed once and the actor received the latest observation. An independent trace audit found no clock, contact, target, policy-call, matched-prefix or endpoint errors. Retention is still poor. Some failed runs put the cube below the table, as low as about 0.02 m world z; the historical robosuite source specifies `mujoco-py 2.0.2.9`/`2.0.2.13` in different files, whereas the runnable environment uses `2.1.2.14` with MuJoCo 2.1.0. That compatibility deviation remains a limitation, not an established cause of failure.

## Failure-directed follow-ups

Corrective closure starts were chosen from each uninterrupted trace's first closing command: steps 40, 32 and 34 for seeds 0, 1 and 2. With frozen recurrent state, a 0.5-second pause retained **0/3** and a 3-second pause retained **1/3**. All six ran the full deadline with fixed targets and bounded drift. These starts used inspected baselines and are descriptive, not independent qualification cases.

One bounded recurrent-state alternative reset the BC-RNN hidden state and call counter on resume while preserving the latest live observation and all controller/physical rules. A 0.5-second corrected-closure pause retained **0/3**; a 0.5-second carrying pause retained **1/3**, versus **0/3** carrying with frozen state. It did not pass the physical resume criterion. All 12 follow-up traces passed independent invariant and endpoint checks.

The Phase 3 handoff requires no pause-induced drops and physically successful resume before EEG integration. These matched controls fail that gate, and uninterrupted retention is 0/3, so the requested end-to-end EEG-to-learned-robot replay and five-person comparison remain **not executed**. The existing causal EEG/gate trace contract is connected and checked in software only. The next bounded research step is a measured native retention adaptation or compatible-runtime parity check, followed by fresh HOLD/resume controls; do not treat any of these 36 runs as a reliability estimate or fresh EEG confirmation.

## Artifacts and recovery

Raw traces, execution logs, one matched video repeat, source snapshot, package freeze and compact metrics are preserved in `/home/ubuntu/robomimic-native-phase4-20261104/`. The [private Phase 4 Drive folder](https://drive.google.com/drive/folders/1JVKaaJdDKM7GVnYNQABPGp129MvcOaK9) contains the raw archive and video. The official checkpoint is in the private [Phase 2 folder](https://drive.google.com/drive/folders/1mCICv4tEWmoD1x2AGUZ7_FEiz55A-OrA). All three were downloaded and matched to local SHA256:

| Artifact | SHA256 | Drive readback |
|---|---|---|
| Checkpoint | `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605` | Verified |
| [Phase 4 full raw archive](https://drive.google.com/file/d/1muXMwTkGT-BVEOxR728EAD8La_aL0OgR/view) | `2160353974574251b2afdc51b45114667406bb352b0f127adb89e71f301767b1` | Verified |
| Representative video | `4a9dd9ad725dde01b4f73e53d832f17bd993a60ee33fdff9fd7cdcd2d45d4896` | Verified |

Compact [matched metrics](metrics.json) and [follow-up metrics](followup_metrics.json) retain every condition and seed. The representative video is a repeat of seed 0's 3-second carrying pause; its initial and terminal state, retention result and policy-call count match the corresponding non-video run. No process remains active.
