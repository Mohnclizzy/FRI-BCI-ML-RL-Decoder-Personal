# NVIDIA candidate reset diagnostic — 2026-11-02

## Decision

The first five-episode playback is **not a valid five-start evaluation**. Four
recorded reset poses were not applied to the next physical step. A replacement
bounded screen used five independent simulator processes, one episode each,
seeds 9–13. All five starts passed the reset-continuity gate, and **1/5 met the
sampled sustained two-finger lift criterion**. This is evidence of occasional
simulated table pickup, not reliable grasping or a controlled improvement over
the project's own PPO run.

Stop extending this candidate into retention, HOLD/resume or EEG until its
runtime/physics port and high contact forces are understood. Keep the project
PPO and scripted comparison intact. The next bounded work is a source-and-trace
fidelity audit of action scale, observations, reset and actuator behavior against
NVIDIA's declared runtime; no training or checkpoint transfer is justified yet.

## Reset finding

The old run logged a plausible new cube/hand pose at each `initial_pre_action`,
but on the first 1/120-s physics update the cube and hand returned near their
previous episode's terminal state. Initial-to-first-sample displacements:

| Old episode | Cube (m) | Hand (m) | Valid start | Raw sampled lift |
|---:|---:|---:|:---:|:---:|
| 1 | 0.0004 | 0.0028 | Yes | Yes |
| 2 | 0.7554 | 0.3456 | No | No |
| 3 | 0.4562 | 0.3316 | No | No |
| 4 | 0.4016 | 0.3539 | No | No |
| 5 | 0.4420 | 0.3305 | No | No |

The imported native reset code requests zero cube velocity and writes a new
cube and robot state. The trace proves a physics/data-state mismatch across
in-process resets; it does not identify which layer loses those writes. Invalid
episodes are unscorable as independent starts, not pickup failures.

`scoring.reset_continuity` now checks that the first physics sample is within
0.02 m of the reset cube pose and 0.05 m of the reset hand pose. Those are
generous one-tick validity bounds, not a success metric. `run.py` records the
check and rejects the full run if any start fails. `verify_saved.py` also
audits existing raw telemetry without editing it. The old audit is saved as
`results/nvidia_candidate_2026-11-02/reset_audit.json`.

## Five fresh-process starts

The original policy checkpoint and prepared source were reused unchanged in
five new one-episode processes. Each process had a fresh simulator, distinct
seed, complete contact coverage over its observed episode, and a valid first
physics update. Three episodes terminated before the configured ~3-s horizon.
The same >=0.08-m rise and >=0.1-N force on each finger for >=0.5 s was used.

| Seed | Valid | Duration (s) | Native ending | Max rise (m) | Best qualifying span (s) | Pass |
|---:|:---:|---:|---|---:|---:|:---:|
| 9 | Yes | 2.983 | Timeout | 0.561 | 2.042 | Yes |
| 10 | Yes | 0.567 | Terminated | 0.021 | 0 | No |
| 11 | Yes | 1.333 | Terminated | 0.106 | 0.033 | No |
| 12 | Yes | 2.983 | Timeout | 0.040 | 0 | No |
| 13 | Yes | 0.800 | Terminated | 0.034 | 0 | No |

Seeds 9–13 were specified before these outcomes were inspected; no failures
were replaced. This is a small screen, with one successful instance, not a
reliability estimate. Peak finger force still exceeded 100 N in several starts,
and the native controller's arm-velocity and finger clamps fired frequently.
Seed 11 rose above the height threshold briefly but failed the duration test.

Raw run directory on Brev and the local backup:
`outputs/isaac/candidate-reset-diagnostic-20261102Tnext/`. The compressed
backup's remote and local SHA256 is
`8a282ee1fb074506cffd9b2c515d68a709c5f81a61c891760e9eb2b3f53ae080`.
Compact verified evidence is in
`results/nvidia_candidate_2026-11-02/reset_diagnostic.json`.

All five subprocesses exited successfully; physical outcomes were scored from
saved positions and actual contact sensor samples, not those exit codes or
training reward. Eleven focused tests pass. Original project checkpoints,
candidate weights, Isaac Lab pin, archive project and prior outputs were left
intact. No GPU job remains running.
