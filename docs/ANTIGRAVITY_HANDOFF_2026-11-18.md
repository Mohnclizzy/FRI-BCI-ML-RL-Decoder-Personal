# Antigravity handoff — November 18, 2026

## Source of record and synchronization

Repository: https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal

Branch: `codex/robot-learning-design`. Completed training results are published through `44c270b`; use that commit or a verified descendant containing this handoff.

Shared Windows checkout: `C:\Users\mario\.gemini\antigravity\scratch\FRI-BCI-ML-RL-Decoder-Personal`.

An agent using this exact checkout can read these files immediately. A different checkout must fetch the branch and inspect its own changes before fast-forwarding. Preserve local modifications and untracked files; use a separate checkout if necessary. Do not reset or clean the existing workspace. This chat is not automatically available to another agent; this document carries the latest discussion into the repository. No running Antigravity agent has been contacted or confirmed to have read it.

## What uses robomimic

The completed 270-rollout EEG experiment uses the external pretrained **robomimic Lift PH low-dimensional BC-RNN**, in its compatible robosuite/MuJoCo runtime. The project did not train this robot checkpoint.

- Tracked checkpoint: `artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth`.
- Recorded SHA-256: `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`.
- Project harness: `simulation/src/fri_bci_sim/robomimic_harness.py`.
- Reproduction/runtime instructions: `docs/BOUNDED_STUDY_HANDOFF_2026-11-14.md`.
- Results: `results/bounded_eeg_robot_2026-11-14/summary.md`.

Current integration: recorded EEG -> trained EEG decoder -> explicit command gate -> pretrained BC-RNN -> engineered completion latch after acquisition. The latch is not learned. The project's own Isaac PPO is a separate historical learning experiment, not the policy used in these 270 rollouts. Its best cited near-table diagnostic was 3/5 acquisitions from one training seed, not broad robustness.

## Completed evidence

- EEG campaign: 54 complete recipe groups and 2,430 scored checkpoints. Selected EEGNet8 transfer recipe: target LR .0001, epoch 300, seeds 42/123/2026. R08 mean balanced accuracy 68.15%. See `results/gpu_campaign_2026-11-13/summary.md`.
- Bounded comparison: all 15 participants, three decoder families and six gates. Selected moderate EEGNet gate passed the utility screen for 7/15 people. Robot retention: selected 42/45; raw 45/45; always-enable 45/45; always-HOLD 0/45. There are only three robot starts, repeated across EEG participants; do not describe 45 independent robot starts.
- S006 selected requests supplied only 20 enabled steps (1 second), with no acquisition on any of the three starts. Same starts acquired under raw and always-enable; duration-matched contiguous also failed. This supports a delivery-budget explanation for those failures, not a general proof that the manipulation skill is robust.
- Dense-window ablation completed for three seeds and 15 targets: 62.52% balanced accuracy, 42.07% raw REST activation and 40.61% missed imagery. Source training stopped at 99/65/121 epochs; target adaptation ran 300 epochs. See `results/dense_window_ablation_2026-11-14/summary.md` and its manifest, logs and checkpoints. These results do not establish that all further ML is futile. Audit the actual runner and budget accounting before describing it as an exactly matched causal comparison.
- Saved-stream transition diagnostic: `results/next_research_diagnostic_2026-11-14/`.

All these EEG comparisons are exploratory development. Target R12 was inspected historically. Existing source participants are also not an untouched confirmation pool.

## Latest research direction discussed — proposed, not implemented

The user's objective is to improve the translation from uncertain EEG into useful robot decisions, including a policy trained by this project. The next candidate is a **learned supervisory request policy**, using the existing robot skill as its executor.

Candidate inputs: causal EEG scores or features, recent score history, score age, and robot readiness. Candidate actions: wait or request one task. Training must account for false starts during instructed REST, missed instructed imagery requests, and delay; rewarding cube lifts alone invites a controller that ignores EEG.

Use repeated opportunities so one early success and the completion latch cannot hide all later command errors. The detailed request-scheduler proposal is in `docs/NEXT_RESEARCH_HANDOFF_2026-11-14.md`; it changes the command contract and must be a separately labeled experiment. The learned supervisor has not been implemented, trained, or evaluated. No positive outcome or novel algorithm is established.

Before implementation, freeze one bounded protocol: controller architecture, training/development splits, allowed per-person calibration, reward/labels, baseline procedures, trial budget, and stopping/selection criteria. Compare with fixed gates, simple temporal evidence accumulation, and a control with no EEG. Guard against learning the periodic cue schedule, recording time, participant identity, annotation boundaries, or leaked labels. Source training and decoder fitting must be audited across any held-out partitions; splitting windows from the same recordings is insufficient. Keep training seeds nested within participants in analysis.

Instruction labels describe cued imagery, not observed self-initiated robot requests. They may supervise training or score evaluation but must not be inputs to the evaluated controller. Existing recordings can support an exploratory replay study; independent generalization requires genuinely unused compatible data, and live interaction needs a subsequent study.

This recommendation updates the overly broad statement that no further learning is worthwhile: the dense-window run provides evidence about that implementation, not every potential learned controller. It does not authorize silently changing the completed study's results or claims.

Related primary research discussed in chat (starting points, not a novelty determination):

- Lee et al., *Brain–computer interface control with artificial intelligence copilots* (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12599856/
- Wang et al., *Error-related potential-based shared autonomy via deep recurrent reinforcement learning* (2022): https://pubmed.ncbi.nlm.nih.gov/36541532/ . This concerns error-related potentials and navigation, not our motor-imagery dataset.

## Artifact availability

Git tracks the robomimic checkpoint, code, compact results, command-trace archive, robot-result archive, representative video, and dense-window checkpoint/log bundle. Expanded robot rollout folders and some local caches are untracked or ignored; a clone does not receive every loose local file.

Recover expanded evidence from the committed archives and follow their manifests. Large earlier campaign artifacts have Drive recovery instructions in `docs/BOUNDED_STUDY_HANDOFF_2026-11-14.md`. Do not infer that dependencies, credentials, remote runtime state, or private Drive access accompany a Git clone. Verify hashes on recovery and inspect live compute before reusing historical connection details.
