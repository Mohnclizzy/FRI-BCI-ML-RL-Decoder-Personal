# Research follow-up: reliable EEG requests for an autonomous robot

## Status update — 2026-11-14

The Phase 3 dense causal-window ablation has now been completed as a conditional, exploratory R08 development run. It used the frozen EEGNet8 transfer architecture, three training seeds, all 15 target participants, and source R04/R08 training with source-only R12 validation. The dense-window result was 62.52% mean balanced accuracy (participant SD 10.23%), 42.07% raw REST activation, and 40.61% missed imagery. The historical cue-window recipe reported 68.15%, 40.11%, and 47.95%, respectively. This descriptive comparison suggests a modest reduction in missed imagery, with worse accuracy and no improvement in REST activation; it does not support more training or a positive command-quality claim. Full metrics and checkpoint bundle are in `results/dense_window_ablation_2026-11-14/`.

**Decision:** stop further EEG model-training expansion in this finish pass. The dense-window hypothesis did not solve the measured command bottleneck, and stronger gates still suppress most imagery coverage. Finish analysis/reporting of the already completed robot study, retain the dense ablation as a negative exploratory result, and identify genuinely unused data before any confirmatory modeling. R08 is not confirmation and historical target R12 remains unsuitable for that role.

## Objective and authority

The user wants a credible, compelling research contribution, not merely another completed training run. The recommended direction is to test whether a causal command controller can turn noisy recorded EEG into more reliably timed robot task requests. Preserve the completed study as the baseline. This document is a plan for a separate follow-up; its proposed methods and success targets are not achieved results.

Research question: **Can a controller reduce unintended task initiations while preserving detection of intended requests and acceptable delay, compared with strong threshold/persistence baselines, across participants and genuinely unused recordings?**

In the proposed task, HAND_IMAGERY is explicitly mapped to a request to initiate one autonomous lift. This is an experimental mapping, not evidence that the person imagined a grasp or wanted a particular robot action. Autonomous completion after initiation is a new command contract, different from continuous ENABLE/HOLD.

Execute the phases in order. Prepare and freeze each phase's concrete protocol before reading its outcomes. Use the decision rules below to control expansion. Do not launch a broad architecture search or GPU campaign. A negative finding is an acceptable deliverable; impressive claims must follow evidence.

## Authoritative starting point

- Repository: https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal
- Branch: `codex/robot-learning-design`; completed results and reporting corrections through `756a31c`, or a verified descendant. Preserve unrelated work and historical outputs.
- Local clone: `C:\Users\mario\.gemini\antigravity\scratch\FRI-BCI-ML-RL-Decoder-Personal`.
- Read `AGENTS.md`, `docs/literature_reference_summary.md`, and `README.md`. The README contains historical material; the latest result files take precedence for current status.
- Completed study: `results/bounded_eeg_robot_2026-11-14/summary.md`, `protocol.json`, `selection.json`, `participants.json`, `rows.json`, `SHA256SUMS.json`, and `robot/robot_results/aggregate.json` / `joined_metrics.json`.
- Training campaign: `results/gpu_campaign_2026-11-13/summary.md` and `development_summary.json`.
- Prior transition diagnostic: `results/decoder_gate_study_2026-11-11/summary.md`.
- Recovery/runtime handoff: `docs/BOUNDED_STUDY_HANDOFF_2026-11-14.md`.
- Drive study archive: https://drive.google.com/drive/folders/1ip1eEn_tlL7f5uR8xpkhD7qZxbhWcmG9
- Campaign models on Drive: https://drive.google.com/drive/folders/1SFqoYsksN2MsPckNeIRVDf6wrLIENv4S

Verify bytes against saved hashes before reuse. A file existing in Drive or having the expected size is not a successful restore test. Missing/mismatched artifacts must raise an explicit error. Extract archives into a new working directory, never over the frozen result files.

## What is already known

- The completed campaign contains 54 recipe groups and 2,430 scored checkpoints. The selected recipe is EEGNet8 transfer, learning rate .0001, epoch 300, seeds 42/123/2026.
- R08 development balanced accuracy: EEGNet 68.15%, CSP/LDA 62.44%, Riemannian/LDA 64.89%. EEGNet has additional source training; these are not architecture-only comparisons.
- The selected moderate gate passes the engineering utility screen for 7/15 people. Full-recording, training-seed-averaged REST activation is 27.9%; imagery coverage is 36.5%.
- The completed robot experiment used EEGNet training seed 42, the first 20 seconds, and three robot starts. Selected/raw/always-enable/always-HOLD retention counts were 42/45, 45/45, 45/45, 0/45. All three selected failures were S006 across robot seeds 0/1/2. The duration-matched contiguous control had the same failures.
- Selected REST end-effector path averaged .175 m versus .209 m for raw; the paired difference was negative for 10/15 participants. This is a descriptive development result, not an established statistically significant or independently replicated advantage.
- Selected commands request 9.32 enabled seconds on average, including 4.59 of the 11.8 REST seconds. Thus 38.9% describes the fraction of REST time enabled, not the fraction of all enabled time during REST.
- Always-enable acquires in 2.65 seconds on average and spends 17.35 of the 20 seconds under the completion latch. Later command errors can be invisible to terminal-retention counts. The 270 rollouts use only three distinct starting states, not 270 independent participants or environments.
- All target R12 recordings have been inspected in earlier work. Other eligible PhysioNet participants contributed to source training/validation. There is no untouched confirmation cohort in the current setup. The November 14 run did not load R12, but that does not restore its independence.

## Phase 1 — Resolve the bottleneck using saved outputs

Do this first, without retraining. Reuse `eeg_decoder/command_study_metrics.py::analyze_transition_vs_interior`, `bounded_study.py`, saved `scores/*.npz`, annotations and traces.

1. Verify the saved streams and reproduce the reported command metrics. Include all 15 participants and all three EEGNet seeds; average seeds within participant before cohort summaries.
2. Decompose errors into REST interior, imagery interior, imagery-to-REST transitions, and REST-to-imagery transitions. Use the existing two-second definition. Report durations/denominators, false activation events, episode detection, onset delay, offset delay, and gate carryover. Keep transition errors in headline results.
3. Plot per-person score separation and the detection/false-activation/delay tradeoff for the existing fixed gates. Do not search arbitrarily for a better operating point. Separate score errors from extra errors introduced by gate persistence or stale-score handling.
4. Add an explicitly identified annotation-oracle reference to the original 20-second robot contract. Use the instruction state to request ENABLE/HOLD on the same clock, starts, physical criteria and latch. Labels are allowed only in this reference and offline scoring. This is a diagnostic, privileged reference, not an attainable decoder or mathematical upper bound on task performance.
5. Write a decision memo: what would perfect command timing change; how much error remains in steady-state interiors; how much comes from trailing windows/transitions; which intervention is supported? Report uncertainty and participant variability.

In parallel conceptually, but without requiring extra agents, perform a bounded novelty check: review 5–8 closely related primary papers on asynchronous motor-imagery detection, sequential evidence/abstention, and task-level BCI shared autonomy. Record the task, baseline, held-out split, false-activation definition, latency, and specific difference from this proposal. Follow original sources; do not claim that evidence accumulation or shared autonomy itself is new. Start from the repository literature review and Akinola et al. (2017), while noting that its BCI uses SSVEP, not motor imagery.

Deliver `diagnostic.md`, per-participant JSON/CSV, two diagnostic plots, `novelty_matrix.md`, and a concrete frozen follow-up protocol. If no plausible command benefit or distinct contribution emerges, finish the current paper and report that conclusion rather than widening the search.

## Phase 2 — Test a repeated request task, initially without new training

Reuse the robot policy, simulator, physical acquisition/retention criteria, and completion latch. Implement a separate wrapper and protocol version; do not rewrite the old harness's meaning.

Default task design:

- Replay each complete causal EEG recording. For robot evaluation, use consecutive nonoverlapping 20-second slots starting at recording time zero. Include every complete slot; disclose the final incomplete remainder, which remains in full-stream EEG metrics. Do not pick offsets based on results or annotations.
- Reset the robot at each predetermined slot boundary using the scheduled robot seed. Preserve EEG filtering, prediction timestamps, and command-controller history across boundaries. A reset must not manufacture a fresh request.
- At most one accepted request may initiate a lift per slot. After acceptance, the existing policy completes the lift and the existing latch handles retention. Remain unavailable until the next scheduled slot. This preserves the current physical experiment while making repeated initiation observable.
- Re-arm only after at least .50 s of observed score below .50 and at least 1.0 s since the previous accepted request, plus robot readiness. Carry this state across slots. Never re-arm or reset using annotation boundaries. Scores older than 1.0 s cannot authorize an action; stale/gapped evidence clears pending accumulation.
- Reject requests emitted while unavailable; never queue them for automatic execution at the next reset. Every emitted event, including a rejected event, consumes its arm until the low-score re-arm criterion is met. Readiness becoming true must not itself create a rising edge or request. Log emitted, rejected and accepted events separately.
- Match each accepted request to at most one annotated imagery episode using half-open time intervals. Repeated requests in the same episode are duplicates, not additional detections. Count requests during REST as false starts. An imagery episode without an accepted request is missed, including when the robot is busy; also report a separate readiness-conditional metric so availability does not hide failures.
- For the oracle, use the same readiness and slot rules, with request evidence supplied by the annotations. Clearly disclose its privileged information. It cannot bypass the task scheduler.

Compare a small fixed set using the same request scheduler:

1. Raw .50 threshold rising-edge requests.
2. The existing moderate .55 hysteresis procedure converted to rising-edge requests, retaining its existing gate parameters.
3. One simple causal evidence-accumulation controller. Proposed defaults: clip probabilities to `[1e-6, 1-1e-6]`; evidence `e = clip(logit(p) - logit(.55), -2, 2)`; update `A = max(0, exp(-dt/1.0) * A + dt * e)` only when a new score arrives, using actual timestamp differences. Initialize with `A=0`; the first fresh score establishes the clock. Reset `A` after an emitted event, a gap greater than .50 s, or stale evidence; never fabricate updates from repeated robot-clock reads of one score. Emit only on a threshold crossing while armed; the common scheduler then accepts or rejects. Freeze exactly three candidate thresholds `{.5, 1.0, 2.0}` and the common re-arm rules. Treat these classifier scores as heuristic evidence, not calibrated likelihood ratios or a risk-guaranteed test. Do not vary architecture, decay and re-arm settings in this search. These are proposed engineering settings to test, not literature-established optimum values.
4. Annotation-oracle requests.
5. A fixed immediate-start-at-each-slot control, plus a no-request smoke control.
6. A predeclared non-EEG timing control: circularly shift the proposed controller's full event stream by exactly 10 seconds before passing it through the same scheduler. Record both submitted and accepted event counts, because readiness can change acceptance. This is not an exact-duration-matched control; do not label it as one.

First perform the complete score-level comparison without running every robot job. Use all three EEGNet seeds there; use training seed 42 for robot replay, chosen before seeing follow-up results. Keep all participants. Do not select each participant's best seed or gate on their evaluated recording.

Run native robot smoke cases and oracle/immediate-start/no-request checks before the full batch. The main exploratory batch uses robot seeds 0/1/2 for continuity and all complete slots. Freeze its manifest and total job count beforehand. It is larger than the original 270 jobs; measure smoke-run wall time and report the estimated cost before launch. Use the existing host, resumability and logs. Additional robot seeds are a later, separately frozen robustness extension, not a way to replace bad starts.

Primary outcomes:

- False accepted requests per minute of REST, with exact time denominator.
- Imagery-episode request detection, missed requests and duplicate requests.
- Request delay, including a failure-aware view with misses retained; delay among detected episodes alone is insufficient.
- Correctly initiated and physically completed lifts; premature or wrong-state starts are not successes merely because the robot lifts.
- Robot availability, accepted/requested counts, policy-call time, latch time and all REST-associated motion. Preserve the old motion measure as a secondary outcome even though autonomous completion has a different interpretation under this contract.

Keep the fixed moderate gate as the reference; show raw alongside it. A proposed **engineering advancement screen**, to freeze before computing follow-up outcomes, is: at least 50% fewer false starts than the reference; imagery detection at least 90% of the reference and at least 70% in absolute terms; mean participant-level restricted request delay no more than .50 s worse. Define restricted delay as delay to a valid request, assigning the remaining imagery-episode duration when no request occurs. Show the component metrics and per-person failures; this composite is a research improvement screen, not an operational usability or safety threshold. Absolute false-start rates must remain visible even if a relative reduction looks large. If reference false starts are zero, relative reduction is undefined and this screen cannot support a superiority claim.

These are ambitious proposed targets, not promised performance. Select development candidates on this fixed rule, then fewer false starts, then higher detection, then stable configuration name. R08 selection remains exploratory. If no candidate passes, stop controller search after the declared candidates and proceed only with the diagnosis-supported Phase 3 ablation, or deliver the negative result.

## Phase 3 — One conditional training ablation

Do this only when Phase 1 shows a plausible decoder/training bottleneck. Do not retrain simply because compute is available.

Hypothesis: training on clean cue-centered windows poorly matches continuous deployment. Current training uses one pure two-second window per annotation, starting one second after onset; replay scores two-second windows every .25 s, including mixed transition windows.

Compare the original training recipe against dense causal-window training using the same EEGNet8, channels, filtering, alignment rules, source/target roles, three seeds and matched optimizer-update budget. Train on deployment-timed windows from training recordings only, including transitions, with targets defined as instructed state at prediction time. These labels do not establish exact neural-intent onset. Balance sampling by participant/episode so overlapping windows do not dominate. Keep complete recordings together in every split; overlapping windows are not independent examples.

Freeze the request-controller procedure before comparing the two training recipes. Report command outcomes as well as balanced accuracy. Do not add a shorter window, new architecture, new frequency bands and new augmentation simultaneously. Preserve the original result as the matched baseline. This phase's development result is not fresh confirmation.

## Phase 4 — Independent confirmation and claims

Before claiming improved generalization, identify genuinely unused compatible recordings with suitable imagery/REST annotations and sampling/montage compatibility. Audit that they were absent from all source pretraining, validation, adaptation, threshold selection and result inspection. Specify allowed per-person calibration on separate recordings, then freeze the pipeline and open the confirmation set once.

If suitable data cannot be recovered in the available schedule, deliver the exploratory follow-up and an exact confirmation blocker. Do not substitute previously inspected target R12, source-pretrained subjects, random windows, or extra seeds as independent evidence. A nested participant-held-out rebuild is a possible future internal assessment, but does not erase project-level hindsight or count as external replication.

For confirmation, specify primary comparison, margins and sample-size rationale before opening outcomes. Use participant-level paired inference and confidence intervals; robot seeds, windows and episodes are repeated measurements. A point estimate passing the engineering screen is not by itself a statistically supported research claim. Report data-source and task limitations, including cued recorded imagery rather than live self-initiated interaction.

## Implementation map and verification

Reuse:

- `eeg_decoder/bounded_study.py`: saved-checkpoint verification, streams and seed aggregation.
- `eeg_decoder/command_study_metrics.py`: timing and transition diagnostics.
- `eeg_decoder/gpu_pipeline.py`: causal preprocessing, model loading, training and alignment.
- `simulation/src/fri_bci_sim/robomimic_harness.py`: pinned physical criteria, clock and latch.
- `simulation/scripts/run_frozen_robot_jobs.py` and `aggregate_frozen_robot_jobs.py`: existing batch/analysis patterns. They encode the old 270-job design; do not silently repurpose that manifest.

Create follow-up modules/configs and outputs with a new experiment ID. Verify causal availability at window end; 4 Hz to 20 Hz timing; stale-score rejection; no label access by ordinary controllers; no re-arm at annotation/slot boundaries; one request per opportunity; duplicate accounting; full-cohort inclusion; delayed/missed-event denominators; seed averaging; immutable manifests; resume provenance; and policy/latch invariants. Run relevant EEG/simulation tests after implementation. Synthetic tests establish implementation behavior, not EEG performance.

The working native environment is documented in the previous handoff. Saved connection endpoints may expire; inspect the active Brev environment rather than assuming that Jupyter being open or a saved hostname guarantees SSH access. Use existing compute before provisioning another instance. Never expose private credentials in logs or committed files.

## Deliverables and stopping rules

Deliver a short main report, frozen configs/protocol, exact commands, complete per-person metrics, all inclusion/exclusion accounting, compact paired tables, artifact hashes and recovery manifest, command-tradeoff and robot-outcome figures, and one synchronized video selected by participant/seed/slot order rather than best outcome. The video may show annotations as an evaluation overlay; they must not feed the controller.

Commit and push code and compact evidence. Put large artifacts on Drive and verify recovery hashes. Keep the completed November 14 study intact. Explain which result is historical, exploratory development, or independent confirmation. Keep current repository navigation pointed at the latest reports.

Stop expanding when the declared candidates fail, when confirmation data are unavailable, or when the deadline requires packaging. Do not exclude difficult participants, conceal transition errors, weaken the physical success criteria, remove the completion latch to manufacture difficulty, claim live control from replay, or relabel the old results to make them favorable. No new robot-policy training, simulator migration, finger-decoding claim, foundation-model campaign, or broad architecture sweep belongs in this handoff.

**Immediate instruction to the implementing model:** start with Phase 1's saved-output diagnostic and bounded novelty review. Produce the decision memo and frozen protocol before implementing the request wrapper or launching training. The first deliverable should explain which measurable failure the next change is intended to fix.
