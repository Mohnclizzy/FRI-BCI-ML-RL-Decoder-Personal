# Start here: current project state

> **Find the data:** [Results index](results/README.md) · [Storage, backups and latest completion status](docs/DATA_AND_STATUS_MAP_2026-11-02.md).

> **Current check-in:** [Project status as of 2026-11-02](docs/PROJECT_CHECK_IN_2026-11-02.md).
> That report reconciles the older results and supersedes historical next steps.

An earlier snapshot of run artifacts and cached data is available as a private
GitHub release. See [artifact handoff](docs/ARTIFACT_HANDOFF.md). Later PPO and
pretrained-candidate runs are documented separately in the check-in.

This is a simulation-first EEG-to-robot project. Two required workstreams are
learning reliable table pickup and comparing raw, confidence-filtered, and
temporally filtered recorded EEG requests. REST maps to HOLD; fist imagery maps
to GRASP_LIFT. PhysioNet labels fist imagery, not a specific grasp.

## Where to work and push

The independent private GitHub repository is
[`Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal).
The active design branch is `codex/robot-learning-design`. In this checkout,
`origin` points to the independent repository. `upstream` points to Angel's
original repository for reference only; do not modify it.

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
- The bounded 100-update Approach PPO follow-up completed and yielded 0/5
  measured table pickups. A separate NVIDIA pretrained candidate achieved
  1/5 in five valid fresh-process table starts, with high forces and controller
  corrections; it is not a qualified controller or a comparable condition.
- EEG has not controlled a learned robot policy.

## Immediate next step

Follow the [current check-in](docs/PROJECT_CHECK_IN_2026-11-02.md): first audit
the pretrained port and saved traces, then choose a bounded learned table-pickup
experiment. The corrected scripted EEG matched batch remains incomplete and is
a separate reference experiment. Do not present its partial results as a final
comparison. Detailed artifacts under `outputs/` are ignored by Git.

For earlier history, see `docs/PROJECT_HISTORY.md`. For account sharing and
checkout instructions, see `docs/SHARING_AND_REPRODUCTION.md`.
