# Execution handoff: EEG-controlled learned Franka demonstration

Prepared 2026-11-03 after review by two Astra agents. This is the current execution plan, not a record of new completed experiments. It supersedes the next-action recommendations in older handoffs; their results and limitations remain valid. If newer verified results exist, update the status and continue from them.

## Mission and authority

Deliver a reproducible simulation demonstration in which recorded motor-imagery EEG enables, pauses and resumes a learned Franka table-pickup skill. The robot must approach a table-supported cube, acquire it, lift it and retain it. The first interface is continuous enable/HOLD/resume, not a one-shot trigger that ignores later HOLD requests.

Robot learning remains required. A properly attributed pretrained policy or imitation-trained policy is allowed; our existing PPO training remains part of the learning evidence. Do not claim we trained an external checkpoint. A new algorithm, from-scratch PPO, live EEG, and real hardware are not prerequisites.

The user authorizes pretrained models, demonstration learning, simulator changes in isolated environments, and appropriate training. Brev GPU time is available; TACC Slurm is an optional resource, not a prerequisite. Preserve all existing work. Continue through routine implementation choices without returning to the main chat for approval. Use judgment where this plan leaves details open, record the decision, and keep moving toward the working demonstration.

The strongest immediate project is a shared-autonomy systems demonstration and measured filter comparison. A broader study with fresh EEG data, multiple training seeds and broader scene variation comes after that demonstration.

## Source, environment and preservation

- Source of record: [Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal).
- Design branch: codex/robot-learning-design.
- Canonical Brev checkout: /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal.
- Preserve /home/ubuntu/FRI-project-8b9db42 and every existing checkpoint/output directory.
- Inspect repository remotes and history before assigning other checkouts a role. Do not modify Angel's repository.
- Brev instance: compact-green-alligator, NVIDIA L4. The earlier session without CUDA access was superseded by successful GPU execution. Check current process/GPU state before launching work; do not infer it from old logs.
- Isaac Lab: /home/ubuntu/IsaacLab-fri, pinned to e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4.
- The older installed project imported from the archive. Later runs selected Personal simulation/src through PYTHONPATH. Record actual imported file paths and hashes, not just the intended checkout.
- The Personal Brev checkout has dirty/untracked work and divergent/synthetic local history. Snapshot it before reconciliation. Do not reset, clean, force-push, or overwrite it. Use a separate clean checkout when needed and apply reviewed changes deliberately.
- This account's GitHub connector previously published results because direct shell HTTPS authentication was unavailable. Independent publication by the continuing account is NOT verified.
- Newest full-resolution telemetry and checkpoints exist on Brev; an off-host recovery copy is NOT yet verified. An older private release is not proof that newer runs are backed up.

Read AGENTS.md, docs/literature_reference_summary.md, README.md, docs/research_protocol.md, and the latest handoff before changing the experiment. The current evidence index is [HANDOFF_2026-11-03_AFTER_RETENTION.md](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal/blob/codex/robot-learning-design/docs/HANDOFF_2026-11-03_AFTER_RETENTION.md). Historical protocol language describing unfinished work can be stale; reconcile against actual linked results.

## Verified starting point

| Component | Evidence | Meaning |
|---|---|---|
| EEG decoder | Individually calibrated CSP/LDA confirmation S006-S010: 59.33% balanced accuracy; rest false activation 29.33%, 150 windows | Imperfect offline decoding; these inspected recordings are now exploratory |
| Frozen scripted EEG reference | Corrected 120-second replay, five people, raw 5/5 lifts, confidence 3/5, temporal 2/5 | A completed scripted reference, not learned EEG control or evidence that temporal filtering improves completion |
| Original learned lift | 5/5 from in-hand starts | Does not establish table approach or grasp |
| Approach PPO | Two-update smoke completed; later 100 additional updates, 204,800 transitions, physical table evaluation 0/5 | Training execution/resume worked; physical pickup did not |
| Near-table learning | Progress and anti-push screens each 2/5 acquisitions, with a drop | Partial learning with substantial failures |
| Strongest current Isaac checkpoint | Unilateral-contact penalty: 3/5 fixed near-table acquisitions retained at episode end, no detected post-acquisition drops in those three | Diagnostic result from one training seed; not reliable farther-ready-pose pickup |
| Half-arm inference ablation | 0/5 acquisitions, three early terminations | Rejected; preserve result, do not revive without new evidence |
| Existing external candidates | NVIDIA native candidate 1/5 sustained lift; Robot-Nav/Panda_arm_PPO has reach/obstacle tasks, no usable pickup checkpoint | Already investigated; do not repeat a broad search through these dead ends |
| Learned EEG interface | Not integrated; learned HOLD/resume untested | The principal integration gap |

The corrected scripted reference and plots are in [corrected scripted EEG results](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal/blob/codex/robot-learning-design/results/corrected_eeg_2026-11-03-complete/summary.md). Read command activation, rest motion, missed bouts and latency alongside lift counts. Do not discard the frozen reference.

Important artifacts:
- Original pregrasp checkpoint: /home/ubuntu/FRI-project-8b9db42/outputs/isaac/pregrasp-64x100-seed8/logs/rsl_rl/lift_franka/2026-11-02_00-39-11/model_99.pt
  SHA256: 6a9c5ff66ff813c5b3a72596b620f9729953f91febeb564a88421c59c564ee9b
- Strongest current checkpoint: /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/unilateral-50-seed8-20261103T2310Z/logs/rsl_rl/lift_franka/2026-11-03_23-11-39/model_347.pt
  SHA256: 37eb85242539f7d2f0c36a916f3aa8e44d2fe6a70030a9da18963d2506e673f7
- Corresponding physical evaluations: outputs/isaac/unilateral-eval-5x12s-20261103T2310Z/seed8 through seed12.
- Retention diagnosis: docs/RETENTION_DIAGNOSTIC_RESULT_2026-11-03.md.
- Rejected ablation: docs/HALFARM_RESULT_2026-11-03.md.

These are last verified locations, not permission to replace files if paths have changed.

## Phase 1: make the handoff independent

1. Inventory current processes, source changes, remotes, runtime versions and available artifacts. Avoid duplicate jobs. Save exact source/configuration snapshots corresponding to important results.
2. Verify the continuing CLI is usable under the intended continuing model account and can complete a bounded execution request. Separately verify it can read the source and publish a small useful status commit through an authorized workflow. Repository access alone does not establish model-account continuity. Keep credentials out of logs. If authentication is blocked, record that specific blocker and preserve portable patches and local results; continue useful local work.
3. Copy irreplaceable checkpoints, raw telemetry and source/configuration manifests off the VM to available authorized persistent storage. Retrieve at least the current best checkpoint into a fresh location, verify its hash, and load it in the matching runtime. Record what is and is not recovered.
4. Keep this plan as the roadmap. Maintain one short STATUS.md containing the current phase, latest run/result, active job, unresolved issue and exact next action. Use a simple run ledger or existing equivalent for artifact paths. Do not build a general orchestration framework.
5. Update stale startup summaries. Commit documentation/code and compact metrics; place large artifacts in appropriate storage and link them with hashes.

Proceed when the execution source and preservation boundaries are clear. Publication access problems must not silently disappear from status or unnecessarily block independent analysis.

## Phase 2: reproduce one established learned pickup policy

Default candidate: robomimic Lift PH low-dimensional BC-RNN in its matching native robosuite environment.

- [Official model zoo](https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html)
- [Pretrained-model tutorial](https://robomimic.github.io/docs/tutorials/using_pretrained_models.html)
- [Native Lift implementation](https://github.com/ARISE-Initiative/robosuite/blob/offline_study/robosuite/environments/manipulation/lift.py)

The model zoo specifies robomimic v0.1 and robosuite offline_study. Its approximately 100% reported success has NOT been reproduced here. Use an isolated environment, record exact revisions/dependencies and checkpoint hash, and inspect checkpoint metadata for controller, action scales, observation ordering, normalization, policy rate, horizon and reset distribution. Do not transplant its weights directly into Isaac or impose Isaac's timing during native reproduction.

First verify loading, one complete native rollout and saved physical telemetry/video; then use a small multi-start diagnostic batch. Five trials can reveal failure modes but are not a mandatory 5/5 gate or reliability estimate. Keep all attempts.

Native success is only cube-center height above table height plus 0.04 m. It is not our sustained-grasp predicate. Report native success and independent acquisition/retention metrics separately. Preserve the trained runtime while instrumenting it; compare instrumented and original playback behavior before trusting altered execution paths.

A brief lift or positive reward does not satisfy the project. Pilot against the existing project target: at least 8 cm rise from the pre-action baseline, bilateral grasp contact sustained for at least 0.5 seconds, retention at the terminal state, and no post-acquisition drop. Keep detailed signals if simulator contact semantics differ. A substitute contact metric must be physically validated and explicitly documented before evaluation, not quietly chosen because it passes.

Use the candidate's native horizon for reproduction. Freeze the project's evaluation horizon after development pilots and before qualification. Do not silently change the historical 12-second Isaac protocol or pool results from different scenes, runtimes or horizons.

## Phase 3: prove continuous enable/HOLD/resume

Do this before substantial qualification or training.

Provide a small adapter with reset, enable, HOLD and resume behavior. While enabled, the learned policy controls the skill. During HOLD:
- Stop issuing learned motion actions.
- Latch and maintain the appropriate controller equilibrium goals and gripper state.
- Continue simulator physics, EEG playback, telemetry and fixed episode deadlines.
- Do not repeat a relative action, send zero joint targets, replay old torques, or continually redefine the goal as the current measured pose.
- Record policy calls, actual controller targets and measured state so target invariance can be checked.

Resume from current observations. For recurrent policies, explicitly manage hidden state, observation history and native update counters. Freezing history is the initial hypothesis, not a proven solution. Diagnose resume failures using traces before changing policy architecture.

Test always-HOLD, uninterrupted enable, and pause/resume during approach, closure and carrying. Start with three scene seeds across the three pause phases, adding brief and prolonged pauses where needed. Freeze target/drift tolerances using physical validation; account explicitly for settling transients. A pass requires invariant held goals, bounded physical drift, no pause-induced drops, continued clocks and physically successful resume. Completion should enter a defined stable state rather than restart the skill.

A native actor that picks up well but fails HOLD needs an adapter/state investigation, not immediate abandonment of its simulator or another long PPO run.

## Phase 4: connect recorded EEG and produce the complete demonstration

Reuse the current decoder and gate contracts. Do not begin by replacing the EEG model.

- REST maps to HOLD; accepted HAND_IMAGERY enables/resumes the learned skill.
- PhysioNet provides instructed rest and left/right fist imagery; it does not provide grasp-specific intent. Keep that mapping explicit.
- Preserve causal preprocessing, channel order, filter state, buffering and timestamps. Feed no annotations, future data or cue-boundary resets to the decoder, gate or actor. An explicitly labeled oracle control may use privileged information.
- Keep separate simulator/EEG time and enabled-policy-step counters. HOLD must not pause the EEG recording or grant unlimited episode time.
- Apply identical frozen estimates to raw, confidence and temporal gates; match initial simulator states, checkpoint, budgets and label-independent reset schedules.
- Use S006-S010/R12's existing 120-second replay for exploratory integration. Do not retune on those results and call them fresh confirmation.
- Include always-enable, always-HOLD and scripted pause requests through the SAME learned actor/adapter. A new scripted mechanical controller is unnecessary unless making a learned-versus-scripted comparison.
- Include a timing/distribution-matched shuffled-command control for the scientific comparison. Preserve its random seed and construction.

First complete one end-to-end replay with plots and video, then run the matched five-person/three-gate batch. Freeze the protocol before that batch. Compare task completion, instructed-rest motion, command false activation, imagery misses, acceptance delay, drops, switches and failure cases. A filter can reduce unwanted movement while preventing useful execution; report both.

Do not tune until temporal filtering wins. A negative comparative result is useful. Old Isaac and new native results must remain separate; a cross-simulator difference does not identify the effect of the learned controller.

## Phase 5: qualification and final package

After development, freeze the selected actor/adapter/configuration and a held-out robot evaluation list. Use 20 table-supported starts as an initial engineering screen. Default acceptance target: at least 18/20 sustained pickups retained through the final 0.5 seconds, with no post-acquisition drops among counted successes. Report every trial and termination; this project rule is not a statistical guarantee of 90% reliability.

The start must require approach and grasp rather than begin in-hand. Describe the actual reset distribution. Test modest cube position/yaw variation only after the fixed-task demonstration works; declare its range before testing. If variation fails, describe a fixed-scene demonstration honestly. Do not keep tuning on qualification cases and relabel them as held-out; create a fresh list after any revision.

The final package must include:
- One documented launch command and exact environment/configuration requirements.
- Robot-only, manual pause/resume and recorded-EEG execution modes.
- Checkpoints, attribution/licenses, reproducible source and environment manifests.
- Raw trace/compact metrics locations with SHA256 hashes and verified recovery instructions.
- A representative video with EEG/gate/controller state visible, plus documented failures.
- Graphs of robot acquisition/retention, per-person completion versus rest motion, delay/misses, and a synchronized EEG-command-object-height timeline.
- A short report distinguishing completed results, engineering acceptance, exploratory analysis and untested claims.
- A clean current status and no forgotten background jobs.

Record baseline state before the first action and terminal physical state before automatic reset. Include incomplete episodes, acquisition/drop times, contact coverage, held targets and both policy/physics sampling rates. The old Isaac recorder used 120 Hz physics and 30 Hz policy; new native rates may differ. Retain full-resolution traces; decimated charts cannot establish every short contact event. Do not compare eager versus fused simulator execution paths without checking parity.

Working-demo completion means the learned table skill, physical HOLD/resume and recorded EEG execute together reproducibly. Scientific comparison completion means the frozen matched evaluation is reported honestly, whether or not a filter improves outcomes. Neither implies live-user validation or real-robot safety.

## Failure-based fallback choices

Use the evidence to select the next action rather than automatically executing every route.

| Failure | Next useful action |
|---|---|
| Native runtime works, but pretrained weights are unavailable/unusable | Reproduce an official demonstration-learning recipe in that compatible environment; evaluate physical rollouts, not just loss |
| Native reproduction fails unexpectedly | Check checkpoint/runtime/controller/reset compatibility and traces before prescribing more training |
| Native success is reproduced, but the policy fails our stronger height, grasp or retention criterion | Use targeted demonstrations/adaptation for the measured mismatch or another justified route; repeating the same native training recipe may reproduce the same limitation |
| Native pickup works but pause/resume fails | One bounded controller/recurrent-state intervention, then targeted retest; consider pause-aware training only if the failure requires it |
| Legacy environment is fundamentally blocked after documented troubleshooting | Return to preserved Isaac and audit scripted-teacher imitation using the successful scripted skill |
| Imitation learns the task but fails after deviating from demonstration states | Consider targeted additional demonstrations/DAgger or limited fine-tuning, with a clear measured failure target |
| Further PPO is justified by evidence | Use a defined curriculum and explicit transition budget; avoid blind reward changes and unbounded update-count increases |

For any training: split whole demonstrations/episodes, retain validation separation, save checkpoints/configurations/seeds and evaluate saved policies physically. Start with a checkpoint-save/reload pilot, then run a meaningful published recipe or justified budget. Do not mistake a minibatch epoch, PPO update, environment step and physical timestep for the same quantity.

If resuming PPO, inspect the installed runner and load path to verify iteration semantics plus optimizer/normalizer restoration. Report additional updates and transitions explicitly: environments times rollout steps per environment times additional updates. The earlier Approach 100-update run used 64 times 32 times 100 = 204,800 transitions. Checkpoint numbering alone does not prove that budget.

A native BC/BC-RNN training reproduction with initial/final physical evaluation is a useful later strengthening milestone, not a prerequisite for crediting a properly attributed pretrained demonstration. Retain the project's actual PPO results. Use TACC only when a concrete training workload benefits; discover its allocation/partition/container constraints instead of assuming them.

## Operating rule for the continuing agent

Use the existing Brev CLI as the execution owner. Another account can inspect commits and results without becoming a required approval relay. Keep shared memory in files, not separate chat histories.

At each phase: inspect latest status, state the next measurable question, run one bounded experiment in a NEW directory, inspect actual artifacts, update status/run records, publish when available, and proceed to the next justified step. Implement simple restartable scripts and job tracking. After interruption, recover from saved files rather than rerun completed experiments.

Do not spend another long cycle searching catalogs, repeating rejected ablations, polishing infrastructure or enlarging training before resolving the current failure. Equally, do not abandon a promising route because of one diagnostic failure.

Stop a run when it is invalid, complete or at its declared budget. Continue the project when useful authorized work remains. If essential credentials, storage or external resources truly block progress, preserve everything and report the exact missing action. Do not claim completion merely because execution exited successfully or a budget expired.

The next concrete action is Phase 1 continuity/preservation, followed by native robomimic reproduction and early HOLD compatibility testing. No new experiment has been executed by writing this plan.

