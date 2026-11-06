# EEG-to-Franka session wrap-up — 2026-11-04

Work paused at the user's request. The Brev CLI Phase 6 turn was interrupted before a new run directory or rollout was created. A process and GPU check found no active robot, training, or Codex job. No EEG-to-learned-robot rollout or qualification has been completed.

## Verified results and published records

| Stage | Physical finding | Published record |
|---|---|---|
| Native pretrained robomimic Lift BC-RNN | Native lift event 5/5; terminal 20-second retention 0/5 | [Phase 2](../results/robomimic_native_2026-11-04/summary.md) |
| Fixed pose after acquisition | Five aligned starts retained 5/5; this was a post-acquisition controller diagnostic | [Phase 3](../results/robomimic_hold_2026-11-04/summary.md) |
| Continuous enable/HOLD/resume adapter without completion latch | HOLD goals, clocks and drift passed; uninterrupted retention 0/3; pause schedules 0/3–2/3; recurrent reset did not solve brief closure failures | [Phase 4](../results/robomimic_continuous_2026-11-04/summary.md) |
| One-way completion latch at sustained bilateral-contact/8 cm acquisition | 21/21 enabled runs acquired; 17/21 retained at step 400. Four stayed in contact but settled below 8 cm. The physical gate failed. | [Phase 5](../results/robomimic_latch_2026-11-04/summary.md) |

The official checkpoint is an external pretrained policy, not a model trained by this project. The native runtime uses pinned robomimic and robosuite source revisions; its `mujoco-py` version differs from the historical robosuite requirement. Phase 4 verified the Python 3.8 port of the existing causal S006R12 decision contract for raw, confidence and temporal gates, but no EEG-driven learned rollout was run because the robot physical gate failed. The corrected scripted EEG reference remains separate.

## Artifact destinations and recovery

- Source, frozen protocols, compact metrics, analysis scripts, [STATUS.md](../STATUS.md), and this wrap-up: `Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`, branch `codex/robot-learning-design`, published through the connected GitHub app. The local checkout remains dirty at `b3422a0`; do not reset, clean or force-push it, and do not modify Angel's repository.
- [Phase 1 private Drive recovery](https://drive.google.com/drive/folders/1VG3gNnu3wvFrzyCtBlRLS3ODHK7vFmDy): best Isaac checkpoint, newest raw Isaac telemetry, and source/config manifests. All four uploaded files were downloaded and SHA256 verified.
- [Phase 2 private Drive](https://drive.google.com/drive/folders/1mCICv4tEWmoD1x2AGUZ7_FEiz55A-OrA): official BC-RNN checkpoint and native rollout archive. Checkpoint SHA256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`.
- [Phase 3 private Drive](https://drive.google.com/drive/folders/1BjC9bhuEL52lAWsZ6D72gpQI-jInziNh): post-acquisition HOLD traces/source, archive SHA256 `76b320138491df379f7ed2047711013ac2a1d447842e03f2e91edabf7c9ffdc4`. Upload metadata was checked; downloaded-byte verification remains outstanding.
- [Phase 4 private Drive](https://drive.google.com/drive/folders/1JVKaaJdDKM7GVnYNQABPGp129MvcOaK9): all continuous-control raw traces, logs, source, and video. Archive SHA256 `2160353974574251b2afdc51b45114667406bb352b0f127adb89e71f301767b1`; downloaded copies of archive, video and checkpoint were verified.
- [Phase 5 private Drive](https://drive.google.com/drive/folders/1_2inM732QxX0jLFS1Sgjpe19tGOS8A_P): all latch raw traces, source, logs and representative failure video. Archive SHA256 `7d9dfe84f1729489c880f5bbfad85313cb65d88c030da0f4be742cd6b1cf0d26`; video SHA256 `c0992e6fa5de3687455be6562bcdbcf77016c038c083dc3ac0fa0750c3b8b5b1`. Both were downloaded and verified.

Large native raw runs remain under `/home/ubuntu/robomimic-native-phase2-20261104/` through `phase5-20261104/`. Existing Isaac outputs remain under the canonical checkout. These directories were not replaced. The full source/config provenance and exact launch details are in the linked stage records and raw archives.

## Exact resume point

Read `docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md` on the published branch, [STATUS.md](../STATUS.md), and the Phase 5 protocol, metrics and failure traces. In a fresh Phase 6 directory, freeze and test **one** higher completion equilibrium goal intervention, motivated by the four contact-stable height-margin failures. The intended pilot is +0.015 m in world z once at acquisition, with a fixed goal thereafter; this was proposed but **not run**. Judge the same 8 cm terminal rise, bilateral contact, drops, held targets and fixed 400-step deadline. Preserve Phase 5 as the unmodified baseline. Only after physically passing continuous controls should recorded EEG replay and later held-out qualification proceed.
