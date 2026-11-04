# Brev Codex session wrap — 2026-11-04

## State at stop

Brev Codex reached its usage limit after bounded work through Phase 5. The live session proposed a Phase 6 experiment but did not create its run directory or launch it. No robot, training or evaluation job was active at the final status check. This note was reconstructed from the live Jupyter CLI output and the current repository STATUS; it is a handoff, not an additional experiment.

The current detailed record is [STATUS.md](../STATUS.md). All work belongs to the Personal design branch. Preserve the dirty canonical checkout, Isaac archive, and existing outputs; do not reset or clean them. The project's latest plan is [PROJECT_EXECUTION_HANDOFF_2026-11-03.md](PROJECT_EXECUTION_HANDOFF_2026-11-03.md).

## Bounded native-policy phases

- **Phase 2, official robomimic Lift PH BC-RNN:** aligned-start native lift event 5/5, 20-second terminal retention 0/5. The external checkpoint is pretrained; this project did not train it. Native success is not retained pickup.
- **Phase 3, fixed acquisition pose:** 5/5 exploratory trials retained sampled bilateral contact through step 400 after fixing the end-effector goal and closing the gripper at acquisition. The terminal cube-rise margin was only 3.8–8.7 mm above the 8 cm threshold. A single exploratory resume trial was not qualification.
- **Phase 4, continuous enable/HOLD/resume:** target invariants, clocks and deadlines passed, but retention gates failed. The EEG-to-learned-policy rollout was not run.
- **Phase 5, one-way completion latch with matched controls:** 21/21 positive controls acquired; 17/21 met terminal retention. Four fell below the 8 cm threshold despite continued bilateral contact. The physical gate failed, so the recorded S006R12 EEG raw/confidence/temporal learned-policy replay was not run.

Phase 5 artifacts: [summary](../results/robomimic_latch_2026-11-04/summary.md), [frozen protocol](../results/robomimic_latch_2026-11-04/protocol.md), and [metrics](../results/robomimic_latch_2026-11-04/metrics.json). Raw archive and failure video were uploaded to the [private Phase 5 Drive folder](https://drive.google.com/drive/folders/1_2inM732QxX0jLFS1Sgjpe19tGOS8A_P), downloaded back, and SHA256 verified. Hashes and local paths are recorded in STATUS.

## Where things stand

There is meaningful engineering progress: the official policy was reproduced in its native environment, a fixed-goal retention method was tested, HOLD clocks/targets were instrumented, 24 matched controls were recorded, and large artifacts were recovered off-host. There is **not yet** a completed learned EEG-to-robot demonstration or a qualified table-pickup policy. Keep the frozen scripted EEG reference and the Isaac PPO result (3/5 retained near-table diagnostic starts) as separate evidence.

## Resume point

Read STATUS.md and the Phase 5 protocol, summary, metrics and failure traces. Decide whether runtime parity or a predeclared higher acquisition-height margin directly addresses the four terminal-rise failures. The proposed +0.015 m world-z latch margin was **not run**. If justified, run only that bounded intervention in a fresh directory, freeze its protocol first, record all starts and failures, and reapply the physical retention gate. Do not proceed to EEG qualification unless the learned physical gate passes. The native Phase 2 archive's download-hash verification is still outstanding; other backup scope is listed in STATUS.md.
