# Bounded EEG-to-robot finish run handoff

The November 13 R08 campaign summary is in `results/gpu_campaign_2026-11-13/`. This run uses the selected `EEGNet8 / transfer / lr 0.0001 / epoch 300` recipe, all three target-training seeds, R04 target fitting, R08-only development, and S001-S015. It does not open R12. It reuses the November 11 CSP/LDA, Riemannian/LDA, transition analysis, utility metrics, and fixed gate settings.

## Current execution state (November 14, updated)

- Implemented `eeg_decoder.bounded_study`: strict campaign checkpoint/provenance/hash/alignment checks; score and cue probability archives; six fixed gates per decoder; per-person seed averaging before the declared utility test; deterministic rule-based selection; causal harness-format traces; 20 Hz timing cross-check; exact-duration contiguous and 10-second shifted schedule creation; protocol, selection, and SHA256 manifests; command-tradeoff plot; fixed 270-job manifest when a gate passes.
- Added seven focused tests for causal timing/staleness, REST carryover, exact schedule matching, seed averaging, missing seeds, and fixed gate candidates. Full repository suite: `113 passed, 8 subtests passed` (two benign NumPy warnings from a deliberately tiny gate fixture).
- All S001-S015 R04/R08 local cached recordings exist. The old local `outputs/local_scratch_2026-11-11/transfer/S001..S015/model.pt` files are not the requested campaign checkpoints and must not be substituted.
- All 45 selected checkpoints, source models and manifests were recovered from private Drive via authenticated downloads. Checkpoint digests, campaign identity, provenance, epoch, source linkage and alignment were verified. Compact outputs and all JSONL traces are in `results/bounded_eeg_robot_2026-11-14/`; see its `summary.md`.
- The selected R08 candidate is EEGNet / moderate hysteresis (threshold .55, margin .02, persistence .25 s): 7/15 participant utility passes; mean REST activation .279; imagery coverage .365. R08 was used for both selection and scoring, so this is exploratory/optimistic.
- Both saved Brev SSH endpoints still time out/refuse. The pinned native runtime and robot checkpoint are absent locally. No robot rollout, representative video, or Drive backup of this new run has happened. Exact 270-job manifest and schedules are prepared pending native-runtime recovery.

## Resume commands

The required campaign artifacts are already recovered under `outputs/campaign_recovered/`. No Drive access is needed to reproduce the decoder comparison. Run:

```powershell
python -m eeg_decoder.bounded_study `
  --campaign-dir C:\path\to\gpu_campaign_2026_11_12 `
  --cache-dir outputs\local_scratch_2026-11-11 `
  --output results\bounded_eeg_robot_2026-11-14
```

The command exits on any missing file, hash mismatch, provenance mismatch, or preprocessing/alignment mismatch. The R08 selection is recorded in `selection.json`; 270 robot jobs are frozen in `robot_jobs.json`.

Next recover access to the saved Brev host/runtime and its native checkpoint; do not replace the robot policy or simulator. Bring up the pinned compatible native robomimic environment described in `docs/NEXT_STEPS_AUDIT_2026-11-11.md`. Reproduce one seed-0 always-enable and always-HOLD case and inspect summary invariants first. Execute only the recorded `robot_jobs.json` with `simulation/src/fri_bci_sim/robomimic_harness.py`, exact listed seed, trace, condition and schedule. Record one JSON result per job and aggregate paired acquisition, terminal retention, REST motion, requested policy steps, executed calls and latch duration. Plot those outcomes against matched controls before writing conclusions.

The next action is native robot-runtime recovery. It is not a new training run. Do not claim task completion until the robot controls, all 270 rollouts, robot plot, report, and Drive backup manifest are complete.
