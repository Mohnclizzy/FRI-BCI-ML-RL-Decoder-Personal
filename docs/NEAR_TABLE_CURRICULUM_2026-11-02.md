# Near-table learned-pickup curriculum — 2026-11-02

Status: additive reset task and launcher stage prepared. The 4-environment,
2-update startup smoke and one-environment reset geometry screen passed on
Brev. Neither reports learned success. The existing `FRI-Franka-Lift`,
Approach, pre-grasp, scripted EEG comparison, checkpoints and outputs remain
separate.

## Completed startup and reset gates

- Brev startup run: `outputs/isaac/near-table-startup-20261102T2245Z`, seed 8,
  4 environments, 2 updates, fresh policy, exit 0. This proves the task can
  construct and train briefly; the logged success reward was 0.
- Brev reset screen: `outputs/isaac/near-table-reset-screen-20261102T2305Z`,
  seed 8, 1 environment, no policy checkpoint. The preserved machine report is
  [`results/near_table_reset_screen_2026-11-02.json`](../results/near_table_reset_screen_2026-11-02.json).
  The cube center was z=0.2800 m on the table. Both finger joints were open at
  0.0400 m; finger body frames were 0.0166 m above cube top and 0.0400 m from
  cube center in the horizontal plane. Both initial finger contact sensors read
  0 N. A zero action moved the cube 0.00029 m and hand 0.00445 m in the first
  policy step. The 9-action, 100-policy-observation interface constructed.
- Runtime: pinned Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`,
  Newton MJWarp, NVIDIA L4, Personal checkout source selected with explicit
  `PYTHONPATH`; near-table source SHA256
  `a4fbdf45eb36a004d5769d15b20f623900f3802cb897882469351189a762d632`.
  The executed reset script SHA256 was
  `9d8a1ec6e63388091643bea43af8fcabb0b65e587cb9e9f2b542482690850d72`.
  The reset screen checks body-frame positions and contact sensors; it is not
  a collision-mesh proof or a pickup test.
- Two-update throughput probes in fresh directories all exited 0 on the L4:
  `near-table-bench-64-20261102T2320Z`,
  `near-table-bench-128-20261102T2322Z`, and
  `near-table-bench-256-20261102T2324Z`. Their warm second rollout collection
  times were respectively 0.460, 0.512, and 0.545 s; initial compilation and
  startup dominate a two-update run. This is a capacity screen, not a
  convergence claim. The 256-environment size was selected.
- Bounded stage-1 run launched from scratch in its own
  `outputs/isaac/near-table-stage1-256x250-seed8-20261102T2250Z`
  directory: 256 environments × 32 rollout steps × 250 updates = 2,048,000
  transitions. The runner's configured checkpoint interval is 250, so the
  final model is the first intermediate physical-screen candidate. Inspect
  `exit.json` and actual model path before use; a launch is not a result.

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
   action/observation dimensions. Completed above; preserve failed attempts
   and original checkpoints.
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
