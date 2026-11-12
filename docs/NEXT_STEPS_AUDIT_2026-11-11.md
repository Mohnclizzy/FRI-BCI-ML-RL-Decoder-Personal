# Current evidence audit and next steps — November 11, 2026

This is the current forward-looking plan. Historical reports, submitted midterm material and experiment outputs remain preserved. Recommendations below are proposed work, not executed experiments or validated improvements.

## Recommended direction

Finish one reproducible, bounded recorded-EEG-to-robot study using the existing pretrained acquisition policy and completion latch. The highest-value work is accurate exposure accounting, matched timing controls and useful command delivery. More source pretraining, a new robot policy, and broad cleaning/model searches are lower priorities.

An honest exploratory engineering study is a viable finished deliverable. New independent EEG data are required for a new confirmation claim after development; obtaining another dataset is not a prerequisite for documenting the existing results accurately.

## What is verified

- Current personal branch: `codex/robot-learning-design`, local and remote HEAD `6f910c4`. Angel's reviewed branch remains at `46cff2a`. No latest-branch changes were discovered in this audit.
- November 11 EEG run completed 89-source-subject pretraining and 15-target transfer/scratch adaptation, calibration and authorized R12 evaluation. Mean balanced accuracy: transfer 63.78%, scratch 53.78%. This compares those two recipes, not EEGNet against CSP or Riemannian methods.
- CRC+hysteresis issued no commands for all 30 models on both R08 diagnostics and R12 evaluation. On R08, 29 models never reach the activation threshold; one crosses for one sample. The failure is therefore observable on development/calibration data.
- The previously measured robot system already connects recorded EEG to learned acquisition and controller-maintained retention. The one-way completion latch overrides EEG after acquisition; this is not continuous learned carrying control.
- Angel's five-subject offline cleaner and single-seed CPU PPO prototype reproduced. The cleaner gives no consistent individual-decoder improvement and uses future samples. His audit tools are useful infrastructure.
- All 155 entries in the current experiment's artifact checksum manifest verified without errors. The run folder is about 1.45 GB. New reports are untracked and raw outputs are gitignored; no verified November 11 off-host backup is established.

## Findings that change the plan

### Correct the command-count interpretation

The November 6 report infers a completion threshold of `(59,120]` commands from full-recording counts. The native robot harness instead queries EEG at `(step-1)/control_freq`, consuming the first 20 seconds during its 400-step, 20 Hz episode. These are different exposure intervals.

| Subject | Full-trace raw/confidence/temporal events | Events issued before 20 seconds |
|---|---|---|
| S001 | 227 / 170 / 128 | 66 / 54 / 47 |
| S002 | 150 / 59 / 35 | 22 / 7 / 3 |
| S003 | 483 / 365 / 308 | 72 / 43 / 26 |
| S006 | 120 / 6 / 3 | 44 / 5 / 3 |

These are decision-event counts, not robot steps or policy calls. The independent audit also resamples requested commands on the robot clock, before latch overrides; those values likewise must not be mislabeled as executed policy calls. See [machine-readable horizon audit](../results/project_audit_2026-11-11/replay_horizon_audit.json).

The `(59,120]` threshold is not established by this evidence and should not be repeated as a causal result. Even time-aligned exposure counts alone cannot identify a universal threshold: timing, interruptions, robot state and latching also matter. Recorded completion outcomes remain evidence; their interpretation needs this correction.

### Separate two incompatible meanings of pause

Commit `319a687` corrected the current harness to enable the policy **inside** a scheduled window. No completed corrected `pause_seed*` batch was found. Older Phase 6 batch scripts still use `pause` for **holding inside** the window. Reusing the changed historical harness can reverse those old controls.

Create distinct, versioned `enable-window` and `hold-window` modes, boundary tests and source hashes. Preserve the historical code/results needed to reproduce previous runs. `run_agent_task.sh` also masks failures with `|| true` and then prints a successful exit/completion message; it should not orchestrate the next scientific batch unchanged.

### All target R12 recordings are inspected

S011–S015 were the midterm's planned unopened cohort, but the authorized November 11 evaluation opened them. The notebook and forward-facing plan must stop calling them prospective/unopened. Preserve the submitted midterm and add this dated amendment. No later gate tuning may present those same recordings as fresh confirmation.

### Reproduction is currently fragile

- The shared `.venv` points to an older checkout. Tests already required an explicit local `simulation/src` import path.
- Notebook setup hard-resets an existing checkout to a moving branch. Published experiments need pinned code and preservation of local edits.
- The interim `build_report.py` replaces the final checksum manifest with only a subset of target artifacts and regenerates outdated gated-evaluation wording. Unify final report generation before rerunning it.
- Current Python lacks `robomimic`, `robosuite`, `mujoco_py` and `mujoco`. The policy checkpoint is present, but the native robot runtime is not ready in this environment. Verify/recover the known working isolated runtime before scheduling rollouts; GPU access is not the first blocker for the native CPU path.
- CRC traces are not yet an accepted gate choice in the native harness or simulator executor. Since they are all HOLD, routing them to the robot now would add no useful comparison beyond the negative control.

## Execution order and completion criteria

### 1. Preserve evidence and repair the reproduction path

Commit compact November 11 results and audit documents to a focused branch; archive the needed model/checkpoint/decision/log artifacts with hashes and verify a restore. Reconcile README, START_HERE, notebook and current STATUS navigation. Preserve historical reports rather than silently rewriting their original claims. Add local test import paths, safe pinned notebook setup and an idempotent final manifest/report builder.

Done when a fresh checkout can run the CPU tests and regenerate the final summaries without dropping hashes, importing another checkout or opening data unexpectedly.

### 2. Close the timing/control evidence gap

Use a new versioned harness, explicit control semantics and fail-fast execution. Join command requests and robot telemetry on the same declared 20-second interval; distinguish requested enable time, actual policy-call time and post-acquisition latch time. Reproduce one existing always-enable, always-HOLD and EEG case before expanding. Then freeze a small paired matrix with matched enabled-duration and time-shuffled controls, using the same robot starts, checkpoint and horizon.

Done when every condition has validated outputs and both completion/retention and REST-associated motion, misses and timing are reported. A hardcoded “typical” contiguous window is not an exact match to a fragmented EEG command trace.

### 3. Run a matched development-only decoder/gate comparison

Start with causal CSP/LDA and the frozen transfer EEGNet on the same participants, target R04 fitting allowance and R08 development scoring. Declare EEGNet's additional source-pretraining data explicitly. Add the existing Riemannian baseline only after the common harness is verified. Do not rank the old 70.7% CSP and current 63.8% EEGNet numbers directly: their cohorts and fitting budgets differ.

Before changing thresholds, measure R08 score distributions in class-interior versus transition windows and locate REST maxima relative to transitions. Classification trains on windows starting one second after cues, while replay includes continuous mixed-transition windows and carried commands; this mismatch is a hypothesis to investigate, not an established cause. Keep transition-associated errors in the headline command metrics.

Check training adequacy on development data as well: each target has 30 R04 windows, so batch size 64 and 25 epochs mean only 25 optimizer updates. This does not establish general scratch-CNN inferiority or prove underfitting.

Predeclare utility criteria that reject always-HOLD, and report REST activation, imagery delivery, onset delay, coverage and per-person variation together. Keep the failed frozen CRC configuration as a reference. Do not inflate the calibration count by subdividing dependent REST episodes. Source-subject-disjoint gate development requires retraining a source model that excludes those development subjects; the current checkpoint already used all 89 source participants.

Done when a small, predeclared candidate set has a transparent development tradeoff table and no parameter selection uses target R12. If no useful operating point exists, report that result rather than escalating into an unbounded sweep.

### 4. Finish the connected demonstration and report

Only advance a useful frozen gate to a small paired robot replay against the reference decoder, retaining the existing robot checkpoint/latch. Present task completion beside unintended REST motion and missed commands. Label subsequent analyses on inspected recordings as exploratory. An additional independent confirmation study requires genuinely new compatible data and a frozen protocol.

Package one reproducible demo, matched controls, current plots, source/runtime/config hashes and an evidence index. Main findings can include transfer versus scratch improvement, failure of conservative gating through total suppression, and the difference between classifier accuracy and command utility. Negative results are part of the study, not reasons to hide or rerun tests until they improve.

## Angel integration scope

Prioritize waveform/amplitude audits, coverage accounting, validation helpers/tests and explicit local test paths. Keep the cleaner optional and offline, with pinned dependencies and immutable output handling. Correct stale reserved-cohort claims before importing its validation runner. The CPU gripper is a teaching/interface fixture; neither it nor older shared documentation should replace the current learned-robot work.

## Audit scope and limits

The audit inspected local and remote branch tips, current and historical status/protocols, notebook setup, report source, experiment artifacts, gate/pipeline code, robot harness/executor/batch scripts and Angel's reproduced additions. Independent reviewers examined EEG, robot and delivery tracks. It did not inspect every binary, verify current cloud/Drive backups, rerun physics or train another model. No scientific configuration or frozen result was altered.
