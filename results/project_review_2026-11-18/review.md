# Engineering and research review, November 18

Reviewed branch `codex/robot-learning-design` at `679a453`. The completed study supports a descriptive comparison of how recorded EEG command schedules affect one pretrained simulated lift skill. The proposed learned supervisor needs two protocol changes before an outcome would answer the user's research question.

## Findings in order of impact

1. **Elapsed time can supply the entire instructed state.** For each of S001â€“S015, the binary instruction intervals in R04 exactly match those in R08, even though their source EDF hashes differ. Applying the participant's R04 schedule to the R08 clock correctly classified 7,278 of 7,278 annotated 4 Hz endpoints (3,588 REST, 3,690 imagery). The other 21 endpoints had no R08 annotation. This is a diagnostic of the dataset. The existing EEG gate used decoder scores. Its 68.15% cue-window balanced accuracy was measured on a different set of windows. A supervisor that can infer elapsed time, including through recurrent state or periodic robot resets, could appear to read intent from EEG. [Clock control and denominators](cue_clock.json)

2. **The proposed acceptance screen exceeds scheduler capacity.** With at most one accepted request in each complete 20-second slot, the 15 recordings have 90 slots for 225 imagery episodes. Even a controller with exact annotations and instantaneous decisions can accept requests in at most 90 distinct imagery episodes (40%). Counting only episodes overlapping the robot horizon gives 90/222 pooled (40.54%). The proposed 70% full-episode acceptance criterion is therefore unattainable. The original 270 rollouts used a different continuous ENABLE/HOLD contract. [Capacity calculation](request_capacity.json)

3. **The November 14 dense-window comparison changed the training budget.** Each of 178 source training recordings has 30 cue windows. The historical cue runner pooled all 5,340 windows and used 42 optimizer steps per epoch at batch size 128. The historical dense runner rounded each recording up separately, then drew 178 batches of 128 (22,784 examples). On target R04, its sampler silently drew 14 examples for a nominal batch of 15. The R08 measurements still describe the completed runs (62.52% dense, 68.15% historical cue balanced accuracy). They cannot identify how much of the difference the window choice caused. The local draft fix pools cue counts, matches odd and partial batch sizes, and tags new checkpoints to prevent resuming an old run under the new budget. Early stopping can still give different total source updates. Use commit `44c270b` for the original run and its file hashes. Corrected training remains unmeasured. [Historical results at their recorded revision](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal/tree/44c270b/results/dense_window_ablation_2026-11-14)

4. **The robot evidence shows a tradeoff on three simulated starts.** Across 270 unique jobs and six conditions, selected EEG, raw EEG, always-enable, always-HOLD, contiguous, and shifted schedules retained 42, 45, 45, 0, 42, and 40 lifts respectively out of 45. Selected EEG reduced average end-effector travel during REST from 0.209 to 0.175 m versus raw, with lower retention. All three selected failures were one participant (S006). Its selected schedule delivered one second of ENABLE on each start. A contiguous one-second control also failed on those starts. The evidence supports studying command delivery. The robot policy is the externally pretrained robomimic BC-RNN and the completion latch maintains the lift after acquisition. [Bounded study](../bounded_eeg_robot_2026-11-14/summary.md)

## Engineering decisions and next experiment

Use the completed robot study as the present project result. For a supervisor pilot, first score **emitted requests** across the full recording, before robot availability is applied. Count false starts per REST minute, distinct imagery episodes detected, duplicates, and onset delay with misses included. Separately count accepted requests, correct initiations and completed lifts under the robot's fixed capacity. Record missed opportunities while busy. Freeze the definitions and feasible success criteria on the new protocol before inspecting controller outcomes.

Replay on existing recordings can exercise the code and compare engineering behavior. For evidence that the supervisor uses EEG to improve the timing of real requests, freeze a clock-only and a simple temporal EEG baseline and test on independent recordings with irregular cue or request timing, with the intended request annotated. Randomly splitting overlapping windows or hiding an explicit time input leaves the cue schedule intact. Closed-loop user benefit would require measuring performance during interaction.

Run the two preflights from the repository root. The clock diagnostic runs from tracked evidence. The capacity calculation uses the local cache to obtain recording durations from signal headers. Its published JSON also contains the durations and counts for each participant.

```powershell
python analysis/audit_cue_clock.py
python -m analysis.request_capacity --cache outputs/local_scratch_2026-11-11/cache/7d3de6fbf853d44a --output request_capacity_check.json
```

A local draft of the dense summarizer accepts the trainer's per-seed folders and the original published file layout, validates all 45 participant/seed records, and calculates report values from the supplied data. Generate a new report in a copy of the results directory to preserve the historical files.
