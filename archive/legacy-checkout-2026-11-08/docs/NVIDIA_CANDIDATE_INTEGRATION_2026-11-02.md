# NVIDIA pretrained candidate integration — 2026-11-02

Status: additive integration implemented and bounded playback completed. No training,
fine-tuning, EEG connection, runtime upgrade, or replacement of project tasks.

**Correction after reset audit:** the five episodes below are raw saved outcomes,
but episodes 2–5 did not start from their recorded reset states in physics. The
first physics sample jumped back to the prior state. Only episode 1 is a valid
start, so the original `1/5` must not be used as a five-start performance estimate.
See [NVIDIA_CANDIDATE_RESET_DIAGNOSTIC_2026-11-02.md](NVIDIA_CANDIDATE_RESET_DIAGNOSTIC_2026-11-02.md)
for the independent fresh-process screen: **1/5 valid starts lifted**.

## Measured outcome

One startup episode passed the sampled physical predicate. The subsequent,
separately recorded five-episode run produced **1/5 raw passes**, with the reset
validity correction above. These are not pooled into a six-episode success rate.
Every episode was retained.

| Episode | Max post-action cube rise (m) | Best sampled sustained lift/contact (s) | Pass |
|---:|---:|---:|---|
| 1 | 0.671860 | 1.591667 | Yes |
| 2 | 0.668371 | 0 | No |
| 3 | -0.003108 | 0 | No |
| 4 | -0.003108 | 0 | No |
| 5 | -0.003108 | 0 | No |

The predicate is unchanged: cube rise >=0.08 m and each finger-to-cube normal
force >=0.1 N spanning >=0.5 s. Baseline is captured before the first action.
Negative post-action maxima mean the cube settled below that baseline; including
the initial baseline in a geometric maximum would give zero. Episode 2 illustrates
why height alone is not a grasp/lift success. Episodes 3–5 had zero recorded finger
contact, while the hand still moved roughly 0.33–0.36 m.

All episodes ended at the native timeout, about 2.983333 s: 179 policy steps at
60 Hz, with 358 recorded scene-update samples at 120 Hz. Contact data was available
in every sample. Initial states showed a table-height cube, open fingers, and the
hand separated from the cube. Terminal states were recorded before automatic reset.
There were no incomplete episodes or playback errors in the completed screen.
`verify_saved.py` independently rescored the serialized telemetry on Windows and
matched all five pass/fail outcomes and sustained-contact spans.

The native controller includes gripper-closure gating and explicit robot-state
limit corrections. The screen recorded 25 finger-clamp and 349 arm-velocity-clamp
events cumulatively. Maximum corrections were about 0.003139 m and 8.608315 rad/s.
Peak measured finger forces were high (up to about 272.74 N). These interventions
and forces are limitations requiring investigation; do not call this a robust,
unassisted, or hardware-qualified learned pickup controller.

## What was added and preserved

The implementation lives only under `simulation/candidates/nvidia_franka_cube/`:

- `prepare.py`: fixed official download/hash verification, safe extraction to a
  new working copy, and explicit compatibility patch manifest.
- `run.py`: CPU probe plus finite one- or five-episode inference, dedicated direct
  environment/contact adapter, pre-action/pre-reset recording, correction counts,
  failure summaries, and unchanged policy weights/normalization loaded in eval mode.
- `scoring.py`, `verify_saved.py`, `test_candidate.py`: timestamp scoring, offline
  serialized verification and nine focused tests.
- `README.md`: exact preparation/probe/playback commands and condition limitations.

No existing FRI task, scripted controller, PPO telemetry path, EEG decoder, or
protocol was edited. `/home/ubuntu/IsaacLab-fri` remains at
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` with a clean tracked worktree. No packages
were installed or upgraded in its `.venv`. Imports of the candidate are scoped to
the playback process. The separate CLI was explicitly left idle while direct SSH
performed this implementation; no second operator launched GPU work.

The original pregrasp checkpoint still hashes to
`6a9c5ff66ff813c5b3a72596b620f9729953f91febeb564a88421c59c564ee9b`.
The latest project Approach checkpoint still hashes to
`4ca3e23b639bda131ec55df8d8ac16551d547b0ae03cae0f5685d91f2597bde7`.
The original archive project and existing outputs were preserved. The CLI's
unfinished candidate draft was copied into the new run's `worker-wip/` directory
before replacement. Downloads, working copies and every startup failure were kept.

## Compatibility condition

This is a port of the NVIDIA policy to the existing pinned runtime, not an exact
reproduction of NVIDIA's declared older runtime. The weights are the official
`model_399.pt`, SHA256
`832d8262ede665b5698c557e79ef0aa4167ec0fae1b5d5d8b222e829effc24a4`.
Archive SHA256 is
`d0fdef74d06752b224e78402437de2b8aaf5ee76145f13338bd81049a9f75f67`.

The port selects the explicit legacy Franka configuration, retains separate
shoulder/forearm groups and static joint friction, and maps deprecated actuator
velocity/effort field names. The pinned scene importer requires cfg-owned assets:
the adapter declares the original robot/cube/table/ground/light configs and
transforms instead of using the removed manual clone API. Original action logic,
41 observations, eight actions, reset noise, gripper gate, reward code and default
horizon are retained. RSL-RL `handle_deprecated_rsl_rl_cfg` adapts model config;
only the actor/normalizer is loaded for inference. Installed RSL-RL was 5.5.1.

Finger contact sensors are an explicit instrumentation addition. The recorder
observes existing scene updates without switching private backend graph or
actuator flags. Sampling does not prove uninterrupted contact between samples.

Preserved startup history:

1. `startup-01`: obsolete `clone_environments` API, before policy execution.
2. `startup-02`: manual scene missing the current importer clone plan; no matching
   articulation, before policy execution.
3. `startup-03`: assets initialized, but missing RSL config compatibility conversion
   rejected legacy `stochastic` model arguments, before policy execution. An early
   draft also mapped joint friction incorrectly; it was corrected before any
   completed policy episode. The discarded drafts and their manifests remain.
4. `startup-04`: corrected port, one complete passing episode.
5. `eval-5ep`: five complete episodes, one pass, no training or replacement trials.

## Artifacts, backup and verification

Remote base:
`/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/candidate-integration-20261102/`.
The successful preparation is `compat-v3/`; the screen is `eval-5ep/`.
Invocation arguments are in each run's `manifest.json`; stdout is preserved in
adjacent `.console.log` files. New output directories are exclusive.

Tracked compact evidence is under `results/nvidia_candidate_2026-11-02/`: screen
summary/manifest, startup summary, preparation manifest, CPU probe, episode CSV,
serialized rescore and checksum index. Large raw artifacts remain ignored.

A compressed archive of this new run, including failed startup evidence and worker
draft, was copied to the Windows checkout's
`outputs/isaac/candidate-integration-20261102.tar.gz` and extracted there.
The remote and local archive SHA256 both equal
`0a19b51b72a2b2565264becf529885bff248be82d6618cb679959e7dfd9c8a0b`.
This verifies a second copy of the new candidate run; it does not establish backup
of every historical Brev output. No large archive or checkpoint was pushed into Git.

Validation: nine focused tests pass; Python compilation and `git diff --check`
pass; CPU checkpoint/config probe passes; one startup episode and five measured
episodes complete. Original runtime remains clean, original checkpoint hashes
match, and no GPU experiment process remains after this work. Jupyter/VM remain on.

## Interpretation and next gate

This demonstrates that the checkpoint can execute and achieve a measured table
lift in this compatibility condition. The corrected five fresh-process screen
has one lift and does not establish reliability. The scene, reset distribution,
controller assistance and short horizon differ from the project's prior 12-second
table test, so **1/5 versus 0/5 is not a controlled improvement estimate**.
No EEG or learned HOLD/resume result follows.

Stop automatic expansion here. Review the high forces/corrections, reset-dependent
failures and native-runtime/port equivalence before deciding whether to invest in
this candidate. Any retention, perturbation, HOLD/resume, or fine-tuning experiment
must be specified separately. The existing project path remains available unchanged.
