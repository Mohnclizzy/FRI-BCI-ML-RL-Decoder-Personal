# Bounded EEG command comparison (R04 to R08)

**Status:** Frozen EEG comparison and all 270 matched robot simulation rollouts complete.

All 45 selected EEGNet checkpoints (15 people × 3 seeds), their source checkpoints, shard manifests, and completion manifests were recovered from Drive. The runner verified hashes, checkpoint provenance, source-model linkage, R04 alignment, and epoch 300. Cue balanced accuracy reproduced the saved per-checkpoint values. All 15 target R04/R08 caches were used; this run did not load R12.

The frozen comparison evaluated three decoders and six gates on each of 15 participants. EEGNet seed metrics were averaged within participant before utility scoring. R08 is both the development threshold source and the comparison set, so these results are exploratory and optimistic after selection.

Cue-window balanced accuracy, reported separately from continuous command utility, averaged **68.15% (SD 11.46%)** for EEGNet (three seeds averaged within participant), **62.44% (SD 12.79%)** for CSP/LDA, and **64.89% (SD 12.22%)** for Riemannian/LDA.

| Decoder / gate | Participants passing / 15 | Mean REST activation | Mean imagery coverage |
| --- | ---: | ---: | ---: |
| **EEGNet / moderate hysteresis 0.55** | **7/15** | **27.9%** | **36.5%** |
| Riemannian / R08 REST q80 hysteresis | 6/15 | 25.1% | 33.9% |
| Riemannian / confidence 0.70 | 6/15 | 33.4% | 43.3% |
| Riemannian / raw 0.50 | 6/15 | 38.8% | 47.4% |
| EEGNet / R08 REST q80 hysteresis | 5/15 | 20.0% | 26.7% |
| Riemannian / moderate hysteresis 0.55 | 5/15 | 33.0% | 41.1% |
| EEGNet / raw 0.50 | 5/15 | 40.1% | 52.1% |
| Riemannian / temporal 0.70, 0.50 s | 4/15 | 26.9% | 34.1% |
| CSP / raw 0.50 | 3/15 | 37.9% | 45.8% |
| CSP / moderate hysteresis 0.55 | 3/15 | 31.8% | 38.4% |
| EEGNet / temporal 0.70, 0.50 s | 1/15 | 10.3% | 13.5% |
| All CRC references | 0/15 | 0.0% | approximately 0% |

The selected gate passes for 7 participants; it fails the predeclared criteria for the other 8. Threshold and gate selection used these same R08 recordings. Nothing here establishes independent generalization, prospective risk control, or physical robot safety.

## Robot stage

The frozen 270 rollouts ran on the recovered Brev L4 with the existing native robomimic/robosuite 1.2 + MuJoCo 2.1 runtime, the versioned completion-latch harness, and the original robot checkpoint. Always-enable and always-HOLD seed-0 smoke cases passed their established invariants before the batch. All jobs completed; the result archive, per-job logs/telemetry/summaries, aggregate metrics, and one representative video are retained. No R12 recordings were loaded.

Each condition contains 15 participants × 3 matched robot seeds. Group means below are over the 45 rollouts (seeds describe simulator variability, not additional EEG participants). Requested enable time is the command schedule before the completion latch; executed policy calls and the latch time are measured separately.

| Condition | Retained | First acquisition (step / 20 Hz) | Requested enable (s / 20 s) | Actual policy calls (s) | Post-acquisition latch (s) | Requested REST enable | REST EEF path (m) | REST moving (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Selected EEGNet gate** | **42/45** | 153.4 / 7.67 | 9.32 | 2.22 | 11.51 | 38.9% | 0.175 | 1.94 |
| Raw EEGNet commands | 45/45 | 139.3 / 6.97 | 11.83 | 2.29 | 13.03 | 51.0% | 0.209 | 2.04 |
| Always-enable | 45/45 | 53.0 / 2.65 | 20.00 | 2.65 | 17.35 | 100.0% | 0.387 | 3.37 |
| Always-HOLD | 0/45 | — | 0.00 | 0.00 | 0.00 | 0.0% | 0.029 | 0.58 |
| Duration-matched contiguous | 42/45 | 53.0 / 2.65 | 9.32 | 2.54 | 16.19 | 50.7% | 0.374 | 3.25 |
| Selected schedule shifted 10 s | 40/45 | 125.4 / 6.27 | 9.32 | 2.42 | 12.82 | 42.5% | 0.266 | 2.68 |

Paired against raw on the same participant and robot-seed starts, the selected gate had **6.7 percentage points lower terminal retention**, **28.9 steps later acquisition among acquired rollouts**, **50.3 fewer requested enable steps**, **1.5 fewer actual policy-call steps**, and **0.034 m less mean REST end-effector path**. The participant-averaged paired REST path difference was negative for 9 of 15 participants. The gate meaningfully reduces requested REST activation, but the robot's completion latch already suppresses most policy calls after a successful lift; consequently the policy-call difference from raw is small. Always-HOLD has zero successes and is not a useful controller. The duration-matched contiguous control reaches the task earlier but produces more REST motion; shifted schedules are intermediate. These are simulated results for a recorded-EEG command replay, not a live-control or safety result.

The selected-gate criteria passed for 7/15 participants in the R08 development analysis, and robot retention was lower than raw. R08 was used for gate selection and evaluation, making this a development result with optimistic selection bias. The study does not show independent generalization, prospective risk control, physical safety, or clinical benefit. The measured bottleneck is command timing/coverage: only 2.22 seconds of policy calls occur on average in the 20-second horizon because the completion latch ends active control after acquisition; the selected trace requests about 9.32 seconds, of which 38.9% overlaps annotated REST.

## Reproduction and artifacts

Run from the repository root:

```powershell
python -m eeg_decoder.bounded_study `
  --campaign-dir outputs\campaign_recovered `
  --cache-dir outputs\local_scratch_2026-11-11 `
  --output results\bounded_eeg_robot_2026-11-14
```

`protocol.json`, `selection.json`, `groups.json`, `participants.json`, `rows.json`, `robot_jobs.json`, `artifact_verification.json`, `command_tradeoff.png`, score NPZs, `robot/robot_results/aggregate.json`, `robot/robot_results/joined_metrics.json`, and `robot/robot_results/robot_outcomes.png` hold the compact evidence. `command_traces.tar.gz` contains the causal JSONL traces. `robot_results_bundle.tar.gz` contains all robot job outputs; `robot/robot_results/S001_seed0_selected/video.mp4` is the representative video selected by fixed participant/seed order. `SHA256SUMS.json` records the delivered artifact digests. Reproduction commands and exact runtime provenance are in `docs/BOUNDED_STUDY_HANDOFF_2026-11-14.md`. The 99 recovered Drive inputs remain under ignored `outputs/campaign_recovered/`; their SHA-256 values and checkpoint provenance are recorded in the verification manifest. The compact evidence and complete 11 MB robot result bundle are archived in the [Drive study folder](https://drive.google.com/drive/folders/1ip1eEn_tlL7f5uR8xpkhD7qZxbhWcmG9).
