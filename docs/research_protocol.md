# Research protocol and GPU readiness — working draft

Updated 2026-11-01. Freeze the protocol after validation and before confirmation;
record its Git commit with every experiment. This document does not establish
algorithmic novelty or report a completed robot experiment.

## Original question, now specified

**On identical recorded EEG and matched physical manipulation scenes, does
temporal command filtering improve task completion, unintended motion and delay
compared with raw predictions and confidence-only acceptance?**

The proposed contribution is a reproducible comparative systems evaluation.
PPO, EEG movement selection, confidence filters and simulation are established
components. New datasets, motion retargeting and synthetic EEG remain separate
extensions. A negative result is valid; a new robot-learning algorithm has not
been specified or claimed.

## Research comparison

| Primary source | Established work | Effect on our scope |
|---|---|---|
| [eeg-bci-arm-control, inspected f1d4ef7](https://github.com/elias-mueller/eeg-bci-arm-control/tree/f1d4ef721cc4cbf7c1a7885c7b335281eec44460) | Streaming decoding, LSL, calibration, confidence rejection and a fixed-velocity displayed arm joint. | EEG-to-visible-joint motion is an engineering baseline. Compare command methods on our matched physical task rather than unrelated classifier scores. |
| [Carrara & Papadopoulo, 2023](https://arxiv.org/abs/2308.11656) | Pseudo-online BCI evaluation addresses offline epoch limitations and idle-state handling. | Sliding windows are not a new contribution. Include instructed rest, preserve timing and use only past samples. |
| [Dillen et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39321842/) | MI/eye-tracking/AR shared control evaluates task performance; reported success changes with the completion-time allowance. | Fix time budgets and report misses and delay. Their added modalities/live users prevent direct numerical comparison with our EEG-only replay. |
| [Original project literature](literature_reference_summary.md) | Shared autonomy, subject variability and public-replay limits. | Keep decoding and skill execution separate; report per-person physical outcomes. |

These sources support the design. They do not establish that the exact proposed
comparison is an unexplored literature gap.

## Conditions and controls

Train the low-level skill on scripted requests first. Use balanced pure HOLD
and GRASP_LIFT trials and matched scene seeds to validate controller competence.
For a scientific comparison use at least three independent policy training seeds;
do not spend the full training budget before a small pilot passes.

Then freeze the policy checkpoints. Compare:

1. Raw REST/HAND_IMAGERY argmax.
2. Confidence-only hand acceptance.
3. The same confidence threshold plus sustained temporal evidence.
4. Oracle scripted requests: explicitly privileged positive control.
5. Always hold: detects misleading stability/low-motion scores.
6. Random/shuffled commands: fixed seeds and documented timing/distribution.

The first three command traces are implemented. The physical controls still
require the executor/recorder. No minimum-dwell or additional hysteresis policy
is silently included; specify those as explicit extra conditions if introduced.
All methods get the same probabilities, timing, scenes, control rates, policy
weights, time budget and reset schedule. Do not retrain each filter's controller
and attribute the resulting difference only to filtering.

## Splits and chronology

- Keep current PhysioNet meanings: task-instructed rest versus fist imagery;
  left/right are merged. A grasp request is our declared mapping.
- Subjects1–5/run12 have been repeatedly inspected. Current results on them are
  exploratory; do not tune on them and report them as fresh confirmation.
- Select decoder, thresholds, persistence and any probability calibration using
  training/validation only. Freeze fresh confirmation people/recordings before
  opening their results. Report individual failures and the calibration budget.
- A per-person calibrated decoder is not zero-calibration cross-subject transfer.
  Cross-dataset studies additionally require matched labels and channel semantics.
- Split whole recordings/trials before generating overlapping windows. Correlated
  windows do not count as independent people or trials.
- The decoder must not read annotations, reset at cue boundaries, use true labels,
  or read future samples. True instructions belong to the evaluator; only the
  named oracle gets privileged labels.
- Preserve trained channel order/reference/filter/rate. Keep SOS state across
  chunks and reset at recording boundaries. Include window/buffering delay,
  not just classifier runtime, in any latency claim.

## Outcomes and analysis

Choose horizons and physical tolerances in scripted/validation pilots, then
freeze them. Use a fixed simulator-clock scene schedule independent of labels
and filter decisions. Include rest and mixed-intent recording intervals.

Record actual simulator state and requests at every control step. Measure:

- completed grasp/lift tasks within the fixed budget and capped completion time;
- instructed-rest time with joint/object motion above frozen tolerances;
- drops after acquisition, aborted attempts and requests that never execute;
- instruction-transition-to-acceptance delay, plus missed requests separately;
- switch rates, acceptance coverage and watchdog events.

Success requires measured sustained lift/contact/target satisfaction, not command
acceptance or positive reward. HOLD must maintain appropriate joint/gripper
targets, including a carrying grasp. Zero joint targets or repeating a previous
relative action are not validated hold implementations.

A continuous robot episode may contain both rest and imagery. Do not assign it
one true class. Rest compliance needs ground-truth time intervals in a separate
evaluator. The simple Outcome summary currently handles pure scripted episodes,
not a complete mixed-timeline physical evaluation.

Report per-person/per-policy-seed results, paired method differences and
participant-level uncertainty. Explore tradeoff curves on validation; fix the
confirmation operating point. Zero switches/zero movement is not success.

## Readiness gates

**Local checks:** contracts, causal replay, equal-input command baselines, source
hashes, dry-run plans and tests. These permit a small cloud integration check.

**Before long clean-command training:** verify task discovery, table starts,
collision/contact, joints, rewards, assets, reset-bank cost, logs and checkpoint
save/reload on the selected GPU. Start with a short job and inspect the scene.

**Before full EEG experiment jobs:** implement physical hold/resume and the
simulator-state recorder; pass scripted positive/negative controls; validate
the continuous decoder and operating points; freeze splits, conditions, endpoints
and analysis. A manual evidence checklist is not physical safety certification.

**Before claims:** complete the paired evaluation and report failures. Recorded
replay cannot validate live user adaptation, clinical benefit or real-robot safety.

## Exploratory software check

On2026-11-01 the existing pooled model processed S001R12 without annotations.
Two-second windows,0.25s hop and16-sample chunks produced493 estimates. Raw argmax
issued13 active estimates; the illustrative0.7 confidence threshold issued none,
with or without0.5s persistence. No robot ran or new model was fitted, and the
threshold was not tuned. These are command counts, not physical task results or
false activations. This already-inspected recording is an exploratory software
check and illustrates the need for always-hold controls and missed-request metrics.
