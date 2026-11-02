# Start here: current project state

This is a simulation-first EEG-to-robot project. The current goal is to compare
raw, confidence-filtered, and temporally filtered recorded EEG requests while
measuring physical lift completion, movement during REST, missed imagery bouts,
and delay. REST maps to HOLD; fist imagery maps to GRASP_LIFT. PhysioNet labels
fist imagery, not a specific grasp.

## Where to work and push

The independent private GitHub repository is
[`Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal).
The active branch is `codex/isaaclab-project-scaffold` and is the repository's
default branch. In this local checkout, `origin` points to that independent
repository and all project branches track `origin`. `upstream` points to
Angel's original repository for reference only. Push ongoing project work to
`origin`; do not push to `upstream` unless explicitly asked to contribute back.

## What is measured

- Five-person calibrated CSP confirmation reached 59.33% balanced accuracy.
  This cohort is now inspected; later analysis on it is exploratory.
- A fixed-scene scripted Franka controller lifted a free cube about 0.179 m
  with sustained bilateral contact. This was not learned-policy success.
- Corrected always-HOLD, uninterrupted-lift, and paused/resumed-lift physical
  controls passed in the fixed scene.
- A 120-second S006R12 temporal replay passed request/trace alignment and
  telemetry-coverage checks, but supplied only 0.692 active seconds, caused no
  motion or lift, and detected 1 of 15 imagery bouts.
- The corrected five-person, three-gate matched batch was interrupted and is
  unfinished. Earlier 1/5 raw and 0/5 filtered lifts are debugging history,
  not valid evidence of a filter benefit.
- Table-start PPO has not established a successful lift. EEG has not
  successfully controlled a learned robot policy.

## Immediate next step

Resume from `docs/stopping_point_2026-11-02.md` and follow
`docs/completion_protocol_2026-11-02.md` to finish the corrected matched batch.
Do not present partial results as the final comparison. The local `outputs/`
folder contains detailed artifacts and is ignored by Git; share it through
OneDrive separately if the other account needs those raw logs/checkpoints.

For the full history, see `docs/PROJECT_HISTORY.md`. For account sharing and
checkout instructions, see `docs/SHARING_AND_REPRODUCTION.md`.
