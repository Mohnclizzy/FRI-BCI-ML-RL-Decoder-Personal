# Pretrained pickup policy audit — 2026-11-02

Status: source/checkpoint acquisition and static review complete; no candidate policy executed, trained, or physically validated by this review. Further project training is stopped pending review. Preserve the frozen scripted comparison and all historical runs.

## Decision

Prioritize a bounded native-environment playback of the NVIDIA Franka Cube checkpoint before expanding project PPO training. This is a concrete downloadable learned-policy candidate, not merely a task configuration. Native reproduction comes before porting its weights or changing the existing project runtime.

## Stopped-run handoff reviewed

Independently fetched Personal commit `d6e4ff3cf4d9b7788ee759b99c9b8cbc79385de0` and read `docs/BREV_HANDOFF_2026-11-02.md`, the per-episode CSV, and recorder implementation. The worker reports the 100-update / 204,800-transition diagnostic and five 12-second evaluations completed before the stop request. Published outcomes are 0/5 lifts; bilateral contact was recorded in one episode. Large raw artifacts remain on Brev, with backup unverified; the tracked summaries/checksum manifest are not a raw-data backup. No raw telemetry was independently rescored in this review.

Recorder review limitation: `_install_physics_step_sampling` switches private Newton flags to eager actuation and wraps `apply_action` to expose physics-step samples with fixed policy targets. This is a change to the evaluation execution path. Identical dt and targets do not by themselves demonstrate physical equivalence to the original fused path. No parity evidence was found in the published handoff. Interpret the result as measured under the documented eager recording mode; do not attribute a baseline difference solely to reward shaping or claim a learned/scripted comparison until equivalence/common execution is established. Native candidate playback should record such interventions explicitly and avoid concealed backend changes.

## Candidate 1: NVIDIA Franka Cube / Isaac Lab + Newton

Official setup: https://docs.nvidia.com/learning/physical-ai/getting-started-with-newton/latest/getting-started/setup.html

Official archive: https://developer.download.nvidia.com/Omniverse/learning/Courses/GettingStartedNewton/getting_started_with_newton_franka_cube.zip

Acquired locally on 2026-11-02. Archive SHA256:
`d0fdef74d06752b224e78402437de2b8aaf5ee76145f13338bd81049a9f75f67`.

The archive contains `franka_cube/checkpoint/model_399.pt` (1,276,899 bytes).
Checkpoint SHA256: `832d8262ede665b5698c557e79ef0aa4167ec0fae1b5d5d8b222e829effc24a4`.
These are independently calculated acquisition hashes, not publisher-signed checksums.

Archive source files inspected:
- README.md; source/franka_cube/config/extension.toml; setup.py.
- franka_cube_env_cfg.py; franka_cube_env.py; agents/rsl_rl_ppo_cfg.py.
- scripts/rsl_rl/play.py.

Source findings:
- Native task: `Template-Franka-Cube-Direct-v0`, Newton-only direct RL environment.
- README declares Isaac Lab `v3.0.0-beta2.patch1`, Python 3.12, RSL-RL >=5.0.1; the course website more generally says beta2. Compatibility with our pinned `e3f7d9f...` is not established.
- Concrete static incompatibility: the archive imports `FRANKA_PANDA_CFG` and indexes actuators `panda_shoulder` and `panda_forearm`. At our pinned commit, that config instead has `panda_arm`, `panda_hand` and `panda_finger2_passive`. Unmodified configuration evaluation would therefore encounter missing actuator keys. The pinned source offers a distinct legacy config, but selecting it alone is not a validated complete port. Preserve the pin and reproduce the declared native runtime or review an explicit candidate-side compatibility patch.
- Config declares 41 observations, 8 actions, MLP [256,128,64], observation normalization. Checkpoint ZIP metadata lists actor, critic and normalizer keys; tensors were not deserialized or executed.
- Seven arm actions set offsets from a fixed default joint pose using per-joint scales; eighth action controls both fingers. This differs from our current task interface, so direct weight substitution is inappropriate.
- Reset opens the gripper, adds arm noise and cube x/y noise, and restores cube default pose. Cube is a gravity-enabled free rigid body. Static inspection found cube pose/velocity writes in reset, not an attachment or ongoing cube-position controller. Actual table support and collision clearance still require physical startup verification.
- Controller is a learned policy plus engineered assistance: gripper closure is blocked until enclosure checks pass. Runtime helpers clamp excessive arm velocity and write corrected finger joint positions/velocities when limits are exceeded. Record these interventions; do not describe the native controller as an unconstrained policy acting alone.
- Physics dt 1/120 s, decimation 2 (nominal 60 Hz policy), five Newton substeps. Native horizon is 3 seconds. Contact tuning, asset, gravity settings and actuator settings differ from our scene.
- Bundled playback has an unbounded `while True` loop and exports beside the supplied checkpoint. Use a bounded harness and a working copy; preserve original download bytes.
- No executed pickup rate, native rollout evidence, or HOLD/resume evidence was established here. The official tutorial's reach/grasp/lift description is a candidate claim to test, not our result.

## Candidate 2: robomimic Lift BC-RNN

Official model zoo source reviewed at commit `d309eaecc18acf4152a830a895a6984b8ac71b05`:
https://github.com/ARISE-Initiative/robomimic/blob/d309eaecc18acf4152a830a895a6984b8ac71b05/docs/model_zoo/robomimic_v0.1.md

The maintainers list downloadable low-dimensional and image Lift BC-RNN agents with approximate native success around 100%, and explicitly require robomimic v0.1 / robosuite offline_study for those checkpoints. This rate is a publisher report, not independently reproduced here. Checkpoint downloads were not tested in this review.

This is a useful fallback learned-controller baseline, but would introduce a different simulator/controller stack and recurrent-state pause semantics. It is imitation learning, not our PPO result. Prefer the closer NVIDIA/Newton candidate first. A frozen external policy alone does not establish project-specific training; any required adaptation/training milestone must be reported separately.

## Next bounded experiment specification (not launched)

1. Finish and verify the stopped project's repository handoff and artifact manifest. No further project training.
2. In a separate candidate directory, verify the two download hashes, dependencies and API compatibility against the existing pinned runtime. Do not upgrade or patch that runtime in place. If incompatible, document exact failures and prepare an isolated compatible environment rather than silently migrating the project.
3. Add a finite native playback harness: one environment, fixed declared seed (8), initial state before action, terminal state before reset, physics timestamps, real contact measurement, actual joint targets and intervention counts/magnitudes. Validate that the cube starts supported outside the gripper and that contacts really refer to each finger versus cube.
4. First startup gate: one native 3-second episode, no optimization. Stop on invalid reset, missing assets, schema mismatch, unstable physics or incomplete recording. Do not keep retrying different settings without recording them.
5. If startup passes, run exactly five consecutive native 3-second episodes, no replacement of failures, no training. Evaluate measured rise >=0.08 m and bilateral force >=0.1 N spanning >=0.5 s; report drops and censored intervals. Separately report native reward/success definitions if they differ. Five episodes are a screen, not reliability certification.
6. A promising result justifies a separately labeled 12-second retention/drop evaluation and fixed-scene adaptation to our task. Changing native horizon/reset/controller makes a new condition. No cross-controller result is claimed until horizons, reset manifests and measurement coverage are matched.
7. Only after physical qualification implement continuous enable/HOLD/resume with latched executed arm/finger targets. Test pauses during approach, closure and loaded lift; policy normalization/history and native gripper-gate state must remain coherent. No EEG input during qualification.

Return source/config/checkpoint hashes, exact commands and versions, all episode outcomes, traces, intervention statistics and failure video if available. A successful load, reward or process exit does not establish a pickup. Any compute budget beyond this finite playback requires a new explicit experiment decision.

Pinned asset source supporting the compatibility finding:
https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab_assets/isaaclab_assets/robots/franka.py
