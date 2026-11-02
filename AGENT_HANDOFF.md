# EEG-to-robot project handoff

## Objective

Evaluate whether temporal filtering of recorded EEG intent predictions improves
simulated grasp-and-lift outcomes compared with raw predictions, while measuring
unwanted actions and delay. Angel is collaborating on EEG; Mario is responsible
for simulation and integration. Use one public dataset and the Franka/cube
task first. A second compatible dataset is a future extension.

## Verified state

- Project branch: `codex/isaaclab-project-scaffold`; draft PR #3. Check the
  current Git revision and working tree before changing anything.
- A scripted controller lifted a physical simulated cube 0.179 m, holding it
  with bilateral contact for 5.075 s. This was one fixed-scene demonstration,
  not a trained policy. See `simulation/cloud/controlled_lift_2026-11-02.md`.
- The corrected task started on Brev with four environments for one PPO
  iteration. It has not learned a lift. The opt-in pregrasp curriculum and
  checkpoint transfer have CPU tests but have not been tested on the GPU.
- PhysioNet per-person CSP on five fresh people scored 59.3% balanced accuracy
  after subject-specific calibration; see `docs/decoder_comparison_2026-11-01.md`.
  EEG replay and offline command metrics exist, but do not control the robot.
- No video, validated hold/resume executor, simulator outcome recorder, or
  matched raw-versus-filtered physical experiment exists yet.

## Brev and safe working practice

Instance: `compact-green-alligator`, one L4. Isaac Lab checkout:
`/home/ubuntu/IsaacLab-fri`, expected commit
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, with `.venv` and Newton backend.
Prior project source is at `/home/ubuntu/FRI-BCI-ML-RL-Decoder`; it was copied
as an archive, not a Git checkout, and may lag GitHub. Inspect and preserve its
outputs before updating it. Logs and checkpoints are ignored by Git; do not
assume they were pushed. The user has paused work: check instance status before
starting it, document costs, and stop it when idle. Never expose credentials.

## Next bounded job

1. Inspect `simulation/README.md`, `docs/research_protocol.md`, and
   `simulation/src/fri_bci_sim/curriculum.py`. Preserve the baseline results.
2. After the user resumes GPU work, sync the current branch to a **new** remote
   directory and install its simulation package into the pinned Isaac runtime.
3. Run a two-iteration, 16-environment pregrasp smoke using the documented
   launcher. Save command, source revision, resolved configs, log and checkpoint.
   Verify task reset and contacts; report observed outcomes, not just exit code.
4. Before table-stage transfer, inspect whether restoring the pregrasp optimizer
   is desired; the current trainer restores it by default. Verify checkpoint
   loading and evaluate using physical table-start lifts with sustained contact.
5. Do not treat in-hand starts or training rewards as table-pickup success.
   Stop and report failures rather than silently launching long training.

Longer term: train/evaluate a reliable policy, implement hold/resume and measured
outcome recording, then compare raw, confidence-only and temporal filters on
matched EEG recordings and scenes. Keep human labels out of the controller.
