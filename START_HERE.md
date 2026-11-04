# Start here: EEG-to-Franka project

The current roadmap is the [execution handoff](docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md). It is self-contained for the continuing CLI/account and supersedes older next-step recommendations. The [latest evidence handoff](docs/HANDOFF_2026-11-03_AFTER_RETENTION.md) links the underlying runs. These documents describe the last verified state; writing the plan did not run new experiments.

## Objective and working repository

Build a reproducible simulation demonstration in which recorded EEG enables, pauses and resumes a learned Franka table-pickup skill. PhysioNet labels instructed rest and fist imagery, not grasp-specific intent. The required first interface is continuous enable/HOLD/resume. Pretrained policies, imitation learning and an isolated simulator change are authorized; preserve existing Isaac work and the frozen scripted reference.

Source of record: [Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal), branch `codex/robot-learning-design`. Inspect actual remotes/history before modifying a checkout. Angel's repository is reference-only; do not modify it.

## Last verified results

- Individually calibrated CSP/LDA achieved 59.33% balanced accuracy on S006-S010 confirmation recordings. These recordings are now inspected; subsequent analysis is exploratory.
- The corrected five-person, three-gate 120-second EEG-to-scripted-robot batch is **complete**: raw 5/5 lifts, confidence 3/5, temporal 2/5. Read unintended motion, missed imagery and delay alongside completion. [Summary and plots](results/corrected_eeg_2026-11-03-complete/summary.md)
- The frozen scripted always-HOLD, uninterrupted-lift and pause/resume controls passed.
- Original PPO lifted in 5/5 in-hand starts; the farther Approach table evaluation was 0/5. Different starts prevent treating those as a paired comparison.
- The strongest later learned checkpoint acquired and retained the cube in 3/5 fixed near-table diagnostic episodes. This is not a qualified farther-ready-pose pickup controller. The half-arm inference ablation failed 0/5 and was rejected.
- The previously tested NVIDIA native candidate produced 1/5 sustained lifts in fresh-process evaluations. The newly proposed robomimic candidate has **not** been reproduced here.
- EEG has not yet controlled a learned robot policy; learned HOLD/resume remains untested.

## Next actions

Follow the [execution handoff](docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md):

1. Preserve source/artifacts, verify recoverable off-host copies, and verify continuing model-account and GitHub access.
2. Reproduce the official robomimic low-dimensional Lift policy in its matching isolated native environment.
3. Test physical HOLD/resume early, then connect the frozen recorded-EEG pipeline.
4. Qualify the complete demonstration and package commands, video, plots and reproducible artifacts.

Do not reset/clean/force-push the dirty Brev checkout or overwrite old outputs. Latest off-host raw-artifact recovery and independent CLI publication are unverified. Keep one current STATUS.md as execution proceeds.

## Evidence and historical records

- [Results index](results/README.md)
- [Latest detailed evidence and artifact paths](docs/HANDOFF_2026-11-03_AFTER_RETENTION.md)
- [Research protocol](docs/research_protocol.md)
- [Sharing and reproduction](docs/SHARING_AND_REPRODUCTION.md)
- [Older artifact release](docs/ARTIFACT_HANDOFF.md) — does not establish backup of later runs
- [Project history](docs/PROJECT_HISTORY.md)

Older check-ins and startup instructions may describe then-incomplete work. Preserve them as history; use the current handoffs for status and execution order.
