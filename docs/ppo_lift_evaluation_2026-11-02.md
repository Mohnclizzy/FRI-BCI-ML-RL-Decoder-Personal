# PPO lift evaluation — 2026-11-02

## Question and protocol

The earlier 64-environment, 100-iteration pre-grasp pilot had weak training
success metrics and many out-of-bounds terminations. Before extending training,
the saved checkpoint was evaluated with the pinned Isaac Lab 3.0 / Newton
runtime on Brev's L4. Playback used Isaac Lab's RSL-RL checkpoint loader, one
environment, seed 8, 30 Hz policy steps, and direct simulator telemetry. The
cube remained a free rigid body. No command teleported or attached it.

Physical lift success was defined as cube center rising at least 0.08 m above
its episode start while both finger contact sensors each measured at least
0.1 N continuously for 0.5 s. The five-episode runs logged cube pose, finger
forces, finger joint positions, and episode boundaries on every step. Terminal
rows returned after Isaac Lab's automatic reset were excluded from episode
success calculations. A later table diagnostic also logged arm joint positions,
hand pose, policy actions, and hand-to-cube distance to localize the failure.

## Pre-grasp checkpoint result

The checkpoint was
`outputs/isaac/pregrasp-64x100-seed8/logs/rsl_rl/lift_franka/2026-11-02_00-39-11/model_99.pt`
(SHA256 `6a9c5ff66ff813c5b3a72596b620f9729953f91febeb564a88421c59c564ee9b`).
It ran five consecutive 12 s pre-grasp episodes. The reset puts the cube between
the fingers; this does not test reaching from the table.

| Episode | Start z (m) | Maximum z (m) | Rise (m) | Best sustained bilateral lift (s) | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | 0.3861 | 0.9534 | 0.5674 | 11.733 | Yes |
| 2 | 0.3855 | 0.9560 | 0.5706 | 11.733 | Yes |
| 3 | 0.3882 | 0.9584 | 0.5701 | 11.733 | Yes |
| 4 | 0.3851 | 0.9604 | 0.5753 | 11.733 | Yes |
| 5 | 0.3870 | 0.9611 | 0.5740 | 11.733 | Yes |

The checkpoint reliably closes and lifts from this in-hand reset in these five
episodes. This establishes a measured pre-grasp skill for this seed and reset
distribution, not table pickup or general robustness. It also shows that the
pilot's low training `Metrics/success_rate` did not predict this particular
checkpoint's pre-grasp physical outcome.

## Table transfer result

The pre-grasp checkpoint was fine-tuned for 100 additional PPO iterations with
64 environments, seed 8, on `FRI-Franka-Lift` table-only starts. The final
checkpoint was `outputs/isaac/table-64x100-seed8-from-pregrasp/logs/rsl_rl/lift_franka/2026-11-02_03-09-47/model_198.pt`
(SHA256 `48b7585d5c13f53c1a66836bf13c1e1ad4155493854a6ba9e4a8d2100993c009`).
The process exited successfully. Final training diagnostics reported zero
success, 0.1501 m object position error, time-limit termination on all recorded
episodes, and no out-of-bounds or abnormal-robot terminations.

That table checkpoint was played for five 12 s episodes from the corrected
table reset:

- Physical lifts: **0/5**. Maximum cube rise was 0.00018 m.
- Both finger sensors remained at 0 N; the gripper closed without touching the
  cube.
- In a detailed single-episode trace, hand-to-cube distance grew from 0.265 m
  to 0.604 m. The arm moved away from the cube while the object stayed at
  approximately 0.280 m.

The checkpoint loaded, produced policy actions, and moved the simulated Franka,
so this was not a missing checkpoint or a playback no-op. The table reset was
the intended fixed start, and a separate scripted controller can reach and
lift from it. The evidence localizes the failure to learned table approach and
grasp behavior after transfer. It does not by itself distinguish an observation
frame problem from inadequate shaping/exploration; increasing iteration count
without inspecting those terms would not be a targeted fix.

## Targeted PPO follow-up configuration

Inspection of the saved resolved config confirms the actor receives the cube's
64-point cloud in robot-base coordinates, fingertip state/contact, joint state,
the last action, and the fixed cube goal. The table replay therefore does have
object perception; replacing that observation layout would break the pre-grasp
checkpoint's input compatibility and is not the first intervention.

The inherited reward config gives fingertip-to-object distance weight **0.05**,
bilateral contact **0.75**, and per-finger contact count **0.1**, while object
goal progress is weighted **5.0**. Goal progress cannot provide an approach
gradient while the cube remains stationary on the table. That makes the dense
reach signal a likely bottleneck, although it does not prove reward weights
alone caused the arm to move away. The initial hand-to-cube distance was 0.265 m;
within 0.23 s of playback it had grown to 0.363 m, with zero finger contact.

A separate task, `FRI-Franka-Lift-Approach`, now preserves the original task,
resets, action space, observation dimensions, and checkpoint compatibility while
increasing only those approach/contact weights to **1.0 / 2.0 / 0.25**. This is
a targeted experimental variant, not a claimed improvement in success. The
first follow-up should be a short resolved-config smoke and one-episode physical
playback; if the arm still moves away, log the actor's point-cloud and command
observations against world-frame cube/hand positions before any longer PPO run.
Compare this variant against the unchanged table baseline and score both with
cube rise plus sustained bilateral contact.

## Project decision

Keep the two stages separate: pre-grasp PPO measured **5/5**, table-transfer PPO
measured **0/5**. Do not present the pre-grasp result as a table pickup. The
scripted Franka skill remains the working low-level controller for the recorded
EEG filtering comparison. Its raw, confidence-only, and temporal pilot is
reported separately in `docs/end_to_end_pilot_2026-11-02.md`.

For another PPO attempt, first inspect the table task's object/goal coordinate
observations and the gradient of its approach/contact rewards against the
observed hand motion. Keep the next run a bounded diagnostic. A successful
training exit or positive reward alone is not sufficient; require measured
cube rise and bilateral contact on table starts.

## Reproduction records

The local ignored output folders preserve the console logs, launch manifests,
resolved training configs, checkpoints, and per-step telemetry:

- `outputs/isaac/pregrasp-playback-5ep-seed8-20261102/`
- `outputs/isaac/table-64x100-seed8-from-pregrasp/`
- `outputs/isaac/table-playback-1env-5ep-seed8-20261102/`
- `outputs/isaac/table-playback-diagnostics-1env-seed8-20261102/`

The measurement callback is
`simulation/src/fri_bci_sim/ppo_telemetry.py`. Runtime: Isaac Lab `v3.0.0`,
commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Python 3.12, Newton
`newton_mjwarp`, NVIDIA L4. The project source was based on cloud archive
`8b9db42`; local checkout HEAD during this follow-up was `376927c` plus the
uncommitted callback. Brev was stopped after the bounded work.
