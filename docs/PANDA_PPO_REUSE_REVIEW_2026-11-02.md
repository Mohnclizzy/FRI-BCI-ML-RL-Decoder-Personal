# Public Panda PPO reuse review — 2026-11-02

Source: [Robot-Nav/Panda_arm_PPO](https://github.com/Robot-Nav/Panda_arm_PPO),
main commit `200d3d16315899eb7a7c696b1272b083474b80fb`. This was inspected
read-only in a separate temporary clone. Nothing from it was merged into the
Personal project.

The repository offers three MuJoCo/Stable-Baselines3 PPO tasks: single-arm
reach to a target, single-arm obstacle avoidance, and dual-arm obstacle
avoidance. Its single-arm reach policy has seven joint actions and a
20-dimensional joint/end-effector/goal observation. The reach environment
controls the seven arm joints and has no cube grasp or gripper action. The
Personal Isaac Lab table task uses nine joint actions, a 100-dimensional
policy observation plus other groups, Newton MJWarp physics, and RSL-RL. Its
goal requires two finger contacts and physical cube rise. A reach actor's
weights are therefore not directly loadable into the table-pickup runner, and
even a converted reach actor would not have learned grasp/lift.

The README describes model checkpoint folders, but at the inspected commit
each of `assets/model/rl_reach_target_checkpoint`,
`rl_obstacle_avoidance_checkpoint`, and `rl_dual_arm_checkpoint` contains only
an empty `.gitkeep` file in the Git tree. No `.zip`, `.pt`, `.pth`, or `.pkl`
policy weights are published in that commit. This repo is useful as a source
of reach curriculum and training-loop ideas; it is not an available trained
table-pickup policy. Reassess if the authors publish weights and a matching
grasp task.

The more compatible reuse lane is an **Isaac Lab/RSL-RL Franka lift** actor
with published weights, matching robot asset, observations, action scaling,
gripper semantics, physics, reset scene, and normalization. The separate
NVIDIA candidate port in this project remains a measured candidate: one of
five fresh-process starts passed the sampled lift predicate under its own
scene; it is not a validated replacement for the project table policy.
Any new actor must first pass a source/config compatibility check, then the
same bounded physical five-start screen with complete telemetry. Continue the
near-table curriculum in parallel with that search because the first reset
geometry and startup checks now pass.
