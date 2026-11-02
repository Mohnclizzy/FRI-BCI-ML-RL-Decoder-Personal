# FRI EEG-to-Franka: current status

The project compares raw, confidence-filtered and temporally filtered recorded
EEG requests controlling the same simulated Franka skill. REST maps to HOLD;
fist imagery maps to GRASP_LIFT. The dataset does not label specific grasps.

## Established evidence

- Real PhysioNet replay and three gates exist. Five-person calibrated CSP
  confirmation achieved 59.33% balanced accuracy. That cohort is now inspected;
  additional experiments on it are exploratory.
- The scripted Franka physically lifts a free table cube in the fixed scene.
- PPO lifted from privileged pre-grasp starts in 5/5 measured episodes, but
  table transfer lifted in 0/5. EEG has not controlled the PPO policy.
- EEG replay reached the scripted executor and produced physical telemetry.

## Audit correction and current work

The original 24 s comparison did not validate physical rest behavior: HOLD
replaced targets with measured positions every step, and the always-HOLD hand
drifted about 0.141 m. Its negative control checked only absence of cube lift.
Historical counts (1/5 raw lifts, 0/5 each filter) are debugging evidence;
do not attribute physical differences solely to filtering or call those
controls passed. Command-only scores remain separate evidence.

The repaired executor and three physical controls are now verified on Brev.
A 120 s S006R12 temporal replay also completed with validated trace alignment,
but its gate admitted only 0.692 active seconds and produced no motion or lift.
The expanded matched batch was interrupted at the user's request; the full
corrected comparison is unfinished. No new PPO training was started.

Read [the stopping-point record](docs/stopping_point_2026-11-02.md) for measured
results and exact next actions, and [the completion protocol](docs/completion_protocol_2026-11-02.md)
for frozen controls and comparison endpoints. The PPO reward variant is still
an unvalidated training experiment, although actual coordinate observations
have now been checked against simulator state.

## Organization and limits

eeg_decoder/ holds decoding and annotation evaluation; simulation/ holds tasks,
executor and cloud scripts; docs/ holds protocols/findings; ignored outputs/
holds models and evidence. The parent folder is a workspace wrapper. Work
locally on the existing branch; do not push or create another branch. Use Brev
for bounded Isaac Lab jobs and stop it when idle. No second dataset, live EEG
or hardware integration is in the core.
