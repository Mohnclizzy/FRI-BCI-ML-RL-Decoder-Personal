# Robot learning design decision — 2026-11-02

Status: selected implementation direction; no new training or validation performed by this review.
Scope: personal repository only. This document adds a future learned-controller protocol;
it does not replace the frozen corrected scripted comparison.

## Decision

Robot learning is a core project deliverable. Build a PPO-controlled Franka
table-pickup skill in the existing pinned Isaac Lab/Newton environment, validate
it independently, then connect recorded EEG through a tested HOLD/resume adapter.
The scripted controller remains a physical reference and completed-baseline target,
not a substitute for the learned-controller deliverable.

Keep continuous enable/pause as the first learned EEG interface. REST and stale
input retain HOLD priority. Defer discrete autonomous execution to a separately
specified experiment: the present binary EEG labels do not encode cancel intent,
and interpreting REST differently would change the scientific question.

Train the low-level policy from robot/object observations and clean commands;
do not train a single network to infer EEG and discover grasp mechanics together.
The existing actor need not receive EEG features or a new command observation:
the supervisor chooses whether the policy may dispatch actions.

## Evidence inspected

- README.md and docs/literature_reference_summary.md, as required by AGENTS.md.
- simulation/README.md and docs/ppo_lift_evaluation_2026-11-02.md.
- tasks/lift.py, tasks/pregrasp.py, tasks/approach.py under simulation/src/fri_bci_sim.
- curriculum.py, executor.py, ppo_telemetry.py in the same package.
- simulation/scripts/controlled_lift.py and simulation/cloud/smoke_ppo_approach.sh.
- Previously read stopping point and frozen completion protocol.

Reported evidence, not independently re-scored raw artifacts in this review:
pre-grasp model_99.pt lifted in 5/5 tested in-hand resets; transferred model_198.pt
lifted in 0/5 table episodes with no finger contact. The later observation audit
reported correct cube coordinates. These support focusing on approach/acquisition,
not claiming a proven reward fix. Five episodes and one fixed scene do not
establish general robustness.

Implementation findings:
1. The Approach task already changes three reward weights to 1.0 / 2.0 / 0.25.
   It preserves the baseline action/observation interface. It is a combined
   reward variant, not a single-term ablation.
2. The table configuration fixes reset poses, disables listed material/dynamics
   randomization and sets curriculum=None. Repeated seeds alone therefore do not
   create a meaningful spatial generalization test. Some prose in simulation/README.md
   still describes inherited randomization/curricula as active; resolved runtime
   configuration must determine the actual experiment.
3. controlled_lift.py generates fixed joint waypoints and a 24-active-second clock.
   That clock is specific to the scripted reference, not a requirement for PPO.
4. PPO playback and EEG execution are separate paths. There is no implemented
   learned-policy EEG adapter in the inspected files.
5. PPO telemetry samples environment/policy steps, unlike the scripted physics-step
   logger. It starts the height baseline after the first nonterminal step and
   excludes post-reset terminal states. It lacks the scripted recorder's complete
   HOLD/target/drop reporting. Harmonize this before cross-controller comparisons.
6. The cloud smoke script hard-codes an older checkout path and run directory.
   Do not execute it without checking the live instance and replacing stale paths.

## Work packages and gates

### A. Recover and establish the runtime

Verify the release ZIP and per-file manifest; restore data and outputs into a
separate checkout. Inspect current Brev status, GPU, free disk, running processes,
existing artifacts and the pinned runtime before launch. Check the two checkpoint
hashes against docs/ppo_lift_evaluation_2026-11-02.md. Preserve originals.

Use new run directories, record source commit, resolved environment/agent configs,
checkpoint hashes, optimizer-restoration behavior and total environment steps.
One operator owns cloud jobs. Copy results and verify them before stopping Brev.

### B. Bounded PPO diagnostic (first GPU learning job)

Use the existing Approach variant, original pre-grasp model_99.pt, seed 8,
64 environments and 100 additional iterations after a two-iteration smoke.
This preserves the already documented diagnostic comparison. Resolve the live
command first; iteration count is not a substitute for reporting rollout steps.

Evaluate five table episodes at the existing 12-second horizon with the frozen
physical definition: cube rise >=0.08 m, each finger contact >=0.1 N, sustained
>=0.5 s. Preserve the baseline table condition for comparison. Report distance,
first contact, bilateral acquisition, lift, drops and failure trajectories.
A successful process exit or training reward is not a gate.

If no contact remains, stop automatic training expansion. Inspect reward/action
traces and compare original pre-grasp initialization against a fresh-init
diagnostic under an explicitly budget-matched follow-up. The current launcher
requires a checkpoint for table stages, so fresh table training needs an explicit
new option and test; it is not already available.

### C. Learn the missing approach skill

If B does not yield table pickup, implement a separate approach curriculum:
cube stays on the table; initialize a collision-free open gripper near the cube,
then progressively increase approach distance toward the existing ready pose.
No in-hand starts count toward table success. Validate each reset with physics
and clearance checks. Advancement uses measured reach/contact/lift on development
episodes, not reward alone; freeze exact ranges and advancement thresholds before
running this new experiment.

PPO remains the primary learner. Demonstration initialization plus PPO fine-tuning
is a contingency if acquisition still fails. A demonstration exporter is not
currently implemented: absolute scripted joint targets cannot be copied blindly
as the PPO task's relative actions. Export aligned observations and correctly
transformed actions at the policy rate, then verify replay equivalence. Label
behavior cloning alone as imitation learning and BC-plus-PPO as such.

### D. Robot-only acceptance before EEG

Proposed engineering gate, to freeze before new evaluation:
- At least 18/20 successful table pickups under clean enable, with no post-acquisition
  drops, in the declared fixed-scene evaluation; report every episode and interval
  estimates. This is a readiness criterion, not proof of population robustness.
- Evaluate across at least three independent training seeds for the final learned
  result; do not select the best seed as the only reported result.
- Define a separate modest cube-position/robot-start perturbation suite after
  validating feasibility. Keep development and evaluation reset manifests separate.
  Report fixed-scene and perturbed results separately, even if the latter fail.
- Match controller evaluation horizons when comparing controllers; report the
  historical 12-second pilot separately if a future horizon changes.

A five-episode diagnostic pass can justify additional validation, not a final
robustness claim. No generalization claim follows from repeatedly resetting one pose.

### E. Learned-policy adapter and interruption validation

Reuse Estimate/Request, DecisionReplay watchdog and SkillExecutor semantics.
Add a dedicated policy adapter rather than replacing controlled_lift.py.

RUNNING: load checkpoint and normalization through the pinned RSL-RL path,
compute the current observation, dispatch the learned action through the task's
action manager, and record the processed actuator targets.
HOLD/PAUSED: bypass new learned targets and reissue the last executed arm and
finger position targets while physics continues. A zero relative action can track
the current pose and is not automatically equivalent to holding a fixed target.
COMPLETE: physical sustained-lift predicate causes absorbing hold until reset.
FAULT/TIMEOUT: record the reason and enter hold; no concealed reset or retry.

Keep observations/history and previous executed-action state consistent through
pauses. Inspect actual actor recurrence: recurrent state requires an explicit,
tested pause/resume policy; do not assume the network is stateless. Bound changes
on resumption using documented action limits and log any intervention.

Before EEG, test always-HOLD, uninterrupted policy execution, and pauses during
approach, closure and loaded lift. Use the frozen HOLD target/drift tolerances.
If interruption fails, train with injected pauses in a separately versioned
condition and repeat robot-only evaluation. Never assume scripted pause success
transfers to a learned policy.

### F. Learned EEG evaluation

Freeze a qualified policy checkpoint before comparing raw/confidence/temporal
gates on identical recordings and reset manifests. Ground-truth annotations remain
evaluator-only. Keep the existing gate settings for the direct exploratory
comparison; any tuned settings require a separate development protocol.

Measure physical completion, drops, rest motion, executed activation, misses,
latency, watchdog interventions, active time and telemetry coverage. Implement
physics-step recording for the learned path or explicitly establish and report
a common sampling scheme before comparing results. Capture reset baseline before
the first action, terminal physical state before reset, and incomplete episodes.

Retain continuous HOLD semantics. A future discrete-skill mode must specify
confirmation, fault stop, cancellation input, retrigger suppression and scoring.
Existing REST labels cannot be relabeled as a validated human cancel command.
It must not be pooled with the current continuous-enable experiment.

## Deliverables and order

1. Artifact/runtime preflight and evaluation-recorder corrections.
2. Bounded Approach PPO diagnostic on Brev and a measured decision report.
3. Approach curriculum or demonstration-assisted PPO only if indicated by (2).
4. Robot-only acceptance and pause/resume adapter.
5. Frozen-checkpoint EEG-to-learned-policy comparison and demonstration.
6. Final report with learning curves, sample budgets, seed variability, physical
   outcomes, failure videos, hashes and exact reproduction commands.

Finish the already frozen scripted comparison as a reference experiment, but
robot learning is now a required workstream and should not wait behind cosmetic
reporting. Schedule GPU work with explicit budgets and intermediate stop checks.

The final claim requires learned table pickup plus measured EEG integration.
If either fails, report partial completion honestly; scripted success cannot
close the learned-controller milestone.
