# Near-table learned-pickup curriculum — 2026-11-02

Status: additive reset task and launcher stage prepared. This document does
not report learned success or authorize treating a smoke test as training
evidence. The existing `FRI-Franka-Lift`, Approach, pre-grasp, scripted EEG
comparison, checkpoints and outputs remain separate.

## Why this rung exists

The project PPO checkpoint learned to lift a cube already between its fingers
(5/5 in-hand starts) but failed measured table pickup. The completed 100-update
Approach fine-tune spent 204,800 transitions from a ready arm/table cube reset
and still yielded 0/5 lifts. The next learning problem is acquisition, not
retaining a grasp the reset has already supplied.

`FRI-Franka-Lift-NearTable` inherits the Approach task's robot, free cube,
observations, 0.1-scale relative joint actions, reward terms, and RSL-RL PPO
agent. It changes only the robot's default reset arm pose. The cube remains
table-supported at the existing fixed location, and both fingers start open.
The arm pose is the measured scripted controller's pre-contact `grasp` joint
waypoint, recorded in the corrected positive control's log. In that reference,
the fingertip plane was about 0.0166 m above the cube top and the open fingers
were outside its sides. Task reset validation must confirm the same clearance
in the manager-based RL environment; a saved waypoint is a proposal, not proof
of a valid reset in a different execution path.

## Bounded startup gate before a long run

1. Launch from the Personal checkout's source via an explicit `PYTHONPATH` or
   verified editable install. The Brev venv's old archive import must not
   silently substitute an older task module. Record import paths, pinned
   Isaac Lab commit, source hashes, resolved task/agent configs and seed.
2. Run a new-directory 4-environment, 2-update smoke from scratch. Inspect
   reset bank creation, actual cube/hand/finger poses, open width, zero initial
   bilateral contact, table support, first physics-step continuity, and
   action/observation dimensions. A process exit or reward is only a software
   gate. Preserve failed attempts and original checkpoints.
3. If valid, benchmark 64/128/256 environments briefly on the L4 to select
   a sustainable headless rollout size. Do not assume NVIDIA's example
   4,096-environment configuration fits this task or hardware.
4. Freeze an initial learning budget before launch: for example, 256
   environments × 32 rollout steps × 500 updates = 4,096,000 transitions,
   with checkpoint and measured five-episode table screens every 100 updates.
   This is a first rung, not the final long campaign. Stop or redesign if
   measured contact and lift do not develop; do not extend merely because
   mean reward increases. A larger budget and farther-start transfer require
   a new explicit experiment record.

The first rung is deliberately fixed geometry. It can establish that a policy
learns contact/lift from a table-supported cube near the gripper. It cannot
establish full approach from the ready pose or spatial generalization.
Subsequent rungs must move the open gripper farther away and add frozen,
validated reset variation before final fixed-scene and perturbed evaluation.

## Measurements and interpretation

For each physical screen, save initial pose before action, first physics
sample, actual finger-to-cube forces, arm/finger targets, terminal state before
reset, drops, incomplete episodes, and all training/optimizer/normalizer
provenance. Apply the existing sampled predicate: cube rise >=0.08 m and each
finger force >=0.1 N spanning >=0.5 s. Use one episode per process if reset
continuity is unverified; report early terminations. The current project PPO
recorder's eager Newton actuation must be disclosed, and a parity check is
needed before claiming a controlled scripted-versus-PPO result.

No EEG is used for this stage. Continuous enable/HOLD/resume is tested only
after a learned robot-only table policy qualifies. The frozen scripted
comparison remains a reference experiment.
