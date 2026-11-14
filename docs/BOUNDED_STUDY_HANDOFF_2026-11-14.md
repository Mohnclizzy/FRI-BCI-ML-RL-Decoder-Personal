# Bounded EEG-to-robot finish run handoff

The November 13 R08 campaign summary is in `results/gpu_campaign_2026-11-13/`. This run uses the selected `EEGNet8 / transfer / lr 0.0001 / epoch 300` recipe, all three target-training seeds, R04 target fitting, R08-only development, and S001-S015. It does not open R12. It reuses the November 11 CSP/LDA, Riemannian/LDA, transition analysis, utility metrics, and fixed gate settings.

## Current execution state (November 14)

- Implemented `eeg_decoder.bounded_study`: strict campaign checkpoint/provenance/hash/alignment checks; score and cue probability archives; six fixed gates per decoder; per-person seed averaging before the declared utility test; deterministic rule-based selection; causal harness-format traces; 20 Hz timing cross-check; exact-duration contiguous and 10-second shifted schedule creation; protocol, selection, and SHA256 manifests; command-tradeoff plot; fixed 270-job manifest when a gate passes.
- Added seven focused tests for causal timing/staleness, REST carryover, exact schedule matching, seed averaging, missing seeds, and fixed gate candidates. Full repository suite: `113 passed, 8 subtests passed` (two benign NumPy warnings from a deliberately tiny gate fixture).
- All S001-S015 R04/R08 local cached recordings exist. The old local `outputs/local_scratch_2026-11-11/transfer/S001..S015/model.pt` files are not the requested campaign checkpoints and must not be substituted.
- The 45 requested checkpoints remain in the private Drive campaign archive. Colab was connected to a GPU, but its Drive mount authorization had expired. Reauthorization returned Google `Error 400: invalid_request`; the artifact ZIP cell therefore did not complete. The current checkout does not contain those campaign checkpoints.
- Both saved Brev SSH endpoints timed out/refused. No compatible robomimic runtime is available in this Windows `.venv`; no robot rollout, representative video, or Drive backup of this new run has happened.

## Resume commands

First restore the original Colab Drive authorization or use the Drive desktop mount/API to copy `eegnet8_seed{42,123,2026}/transfer/lr_0.0001/S001..S015/epoch_300.{pt,json}`, each shard's `manifest.json` and `complete.json`, and each source `source/best.pt` whose hash is referenced by `complete.json`. Do not alter or regenerate those artifacts. Then run:

```powershell
python -m eeg_decoder.bounded_study `
  --campaign-dir C:\path\to\gpu_campaign_2026_11_12 `
  --cache-dir outputs\local_scratch_2026-11-11 `
  --output results\bounded_eeg_robot_2026-11-14
```

The command exits on any missing file, hash mismatch, provenance mismatch, or preprocessing/alignment mismatch. Review `selection.json` before starting the robot batch. If no candidate passes on R08, `robot_jobs.json` must remain empty and stop there.

For a selected candidate, bring up the pinned compatible native robomimic environment described in `docs/NEXT_STEPS_AUDIT_2026-11-11.md`. Reproduce one seed-0 always-enable and always-HOLD case and inspect summary invariants first. Execute only the recorded `robot_jobs.json` with `simulation/src/fri_bci_sim/robomimic_harness.py`, exact listed seed, trace, condition and schedule, checkpoint from the established harness runbook, and video only for participant 1 / robot seed 0 / selected condition. Do not use a replacement simulator or policy. Record one JSON result per job and aggregate paired acquisition, terminal retention, REST motion, requested policy steps, executed calls and latch duration. Plot those outcomes against the matched controls before writing conclusions.

The next action is artifact recovery. It is not a new training run. Do not claim task completion until the checkpoint checks, selection, robot controls, all 270 rollouts (only if a gate passes), result plots, report, and artifact recovery manifest are complete.
