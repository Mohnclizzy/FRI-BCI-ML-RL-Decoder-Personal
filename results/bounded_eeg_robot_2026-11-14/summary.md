# Bounded EEG command comparison (R04 to R08)

**Status:** EEG development comparison complete; robot execution blocked by unavailable Brev host.

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

The exact 270-job manifest (15 R08 traces × 3 robot seeds × 6 conditions), selected/raw causal traces, and duration-matched controls are prepared. No rollout was started: both saved Brev SSH endpoints at `compact-green-alligator` closed/timed out, and the local Windows environment has neither the native robomimic checkpoint nor the compatible simulator runtime. The existing robot checkpoint path is absent locally. The always-enable/always-HOLD smoke cases therefore could not be reproduced on the intended runtime. Resume with the pinned runtime and checkpoint; do not use a substitute policy or simulator.

## Reproduction and artifacts

Run from the repository root:

```powershell
python -m eeg_decoder.bounded_study `
  --campaign-dir outputs\campaign_recovered `
  --cache-dir outputs\local_scratch_2026-11-11 `
  --output results\bounded_eeg_robot_2026-11-14
```

`protocol.json`, `selection.json`, `groups.json`, `participants.json`, `rows.json`, `robot_jobs.json`, `artifact_verification.json`, `command_tradeoff.png`, and score NPZs hold the compact evidence. `command_traces.tar.gz` contains the causal JSONL traces. The complete uncompressed working directory is retained locally. The 99 recovered Drive inputs remain under ignored `outputs/campaign_recovered/`; their SHA-256 values and checkpoint provenance are recorded in the verification manifest. New artifacts have not yet been backed up to Drive.
