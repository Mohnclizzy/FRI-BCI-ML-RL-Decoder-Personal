# Connected EEG-to-robot bottleneck study — 2026-11-05

## Question and evidence boundary

Which limitation most reduces useful recorded-EEG control of the learned robot skill: decoder discrimination, command-gate duty cycle/timing, or the robot's approach/contact/retention policy? Answer with paired conditions on the same recordings and robot starts, then test targeted changes one at a time. This is a simulation and offline-replay study, not live EEG, hardware validation, or a clinical claim.

The current baseline is fixed evidence, not a fresh test: Phase 6 reused development seeds with 21/21 positive retention and 0/3 always-HOLD acquisitions under a completion latch; S006R12 is within-subject/training-subject exploration (raw 3/3 retained, confidence and temporal 0/3); S001R12 is one per-person-calibrated held-out-subject/run result (9/9 retained). Neither 21 runs nor 9 paired starts are independent people. The latch proves post-acquisition equilibrium, not arbitrary learned carrying pause/resume. Read STATUS.md and the saved protocols/summaries for exact provenance and limitations.

## Stage 1 — finish the existing system's evaluation

1. Inventory datasets, split manifests, decoder checkpoints, decision traces, robot start states, raw telemetry and hashes. Verify S002/S003 availability before promising a complete three-person held-out qualification. S001 has already been opened; do not present it as a fresh unopened confirmation under this new plan.
2. Write and freeze a prospective evaluation protocol before opening S002/S003 run-12 outcomes. Training/validation data may include already-inspected development subjects S006–S010. Define the per-person calibration budget separately from cross-person transfer. Whole recordings/trials are the split unit; overlapping windows cannot cross a split.
3. Declare the robot reset bank and any new perturbation/start set independently of EEG labels and gate outputs. Freeze horizon, seeds, checkpoint/source hashes, thresholds, physical acquisition and terminal retention rules, rest-motion tolerance, and maximum runs/resource budget. Include always-enable, always-HOLD and time-shuffled or duration-matched command controls. Do not tune after seeing the fresh test outcome.
4. Run only an existing-authority bounded batch at each gate. Preserve every attempt and failure, causal EEG/physics/policy timestamps, enabled steps, policy-call count, post-acquisition contact/rise, drops, misses, latency, unintended rest activation/motion and switches. Compare paired gate differences at the participant level; repeated scene seeds are not independent subjects.

## Stage 2 — make command duty cycle and timing causal variables

On development data, choose operating points for raw, confidence and temporal gates at matched accepted-time coverage OR matched rest false activation; report both axes and the tradeoff curve. Freeze points before fresh confirmation. Add an activation-duration-matched control and a time-shuffled control that preserve total enabled steps but break EEG timing. If a gate fails because it enables only 14 or 24 of 400 steps, report that as an exposure difference before attributing any change to decoding or temporal filtering. Keep reset schedule, learned checkpoint, control rate and budget identical across conditions. Labels are evaluator-only; neither decoder, gate nor actor may see current/future annotations.

## Stage 3 — decoder comparison

Implement and compare CSP/LDA (existing), a Riemannian covariance/tangent-space baseline, and EEGNet on exactly the same raw-data splits, preprocessing, channel order/reference, causal windowing and calibration allowance. Fit all preprocessing, feature selection, normalization, probability calibration and hyperparameters inside training/validation folds. Use already-inspected subjects to develop and select, then freeze code, checkpoint, thresholds and operating points before opening the fresh cohort. Report participant-level offline discrimination/calibration and resulting command traces; then test whether any difference changes robot outcomes under the same policy and physical starts. A decoder accuracy gain without a useful robot gain is a valid finding.

## Stage 4 — targeted robot-learning intervention

Treat the Isaac unilateral 3/5 near-table diagnostic and half-arm 0/5 as prior evidence of an approach/contact problem, not proof that PPO continuation will fix it. Inspect saved contact/action/approach traces. First validate that successful scripted demonstrations map into the policy's observation and action coordinates, controller scale, gripper semantics and timing; run a small replay parity test. Then compare (a) behavior cloning from those demonstrations followed by a bounded PPO fine-tune, against (b) a bounded continuation from the existing PPO checkpoint, with equal evaluation starts/budgets and at least distinct training seeds when resources permit. Keep the native pretrained BC-RNN result separately attributed. Do not infer learned carrying resume from the completion latch.

## Stage 5 — Ofner hand-action vocabulary

Treat Ofner rest/open/close motor imagery as a separate second-dataset experiment. First verify actual task labels, imagery versus execution, sample rates, channels, references, trials and subject splits from the source. Predeclare rest/open/close to HOLD/OPEN/CLOSE command mapping and a safe robot-state guard. Train and evaluate within that dataset before attempting integration. Separate "replication on a second dataset" from "frozen model transfers to a new dataset"; the latter requires no target-dataset retraining and its own protocol.

## Execution order and stop rules

Hermes is the sole experiment writer on Brev. Codex reviews evidence and design. Work in bounded phases: inventory and prospective protocol -> existing-system qualification and timing controls -> decoder comparison -> robot intervention -> Ofner extension. Stage 3/4/5 can be implemented after Stage 1 prerequisites, but do not run multiple competing experiment writers or silently increase resource budgets. At each phase, save source/checkpoint hashes, exact commands, all raw traces, metrics and limitations; update STATUS.md and publish compact evidence through an isolated Git checkout. Preserve the intentionally dirty canonical checkout at b3422a0, all prior results, and the Phase 6 raw archive; off-host backup remains pending until independently verified. If data or budget is missing, record the precise blocker and continue independent preparatory work.

## Immediate bounded task

Inventory S002/S003 raw EEG availability and the current decoder/split artifacts without inspecting their run-12 outcomes. Create the prospective Stage 1 protocol and frozen manifest, including matched-activation/timing controls and exact acquisition/retention endpoints. Verify the plan against docs/research_protocol.md and docs/literature_reference_summary.md. Publish the compact plan and update STATUS.md, then proceed only to a bounded data acquisition or dry-run step supported by current permissions. Do not rerun S001, S006, Phase 6, or open fresh outcomes before the freeze.
