# Angel branch review and reproduction - 2026-11-11

Reviewed AL-DEV06/FRI-BCI-ML-RL-Decoder branch codex/integrate-eeg-experiments at 46cff2a23348c98d5c8191f87988210157f0b9d5 (November 10). Review used a detached local worktree, without merging or modifying Angel's repository or the current EEG pipeline.

## Verdict

The new work runs and its reported numerical results reproduce. It is useful as an offline diagnostic/comparison suite, not a demonstrated improvement to our personalized decoder or a drop-in causal preprocessing path. Selective integration is warranted; replacing current project documentation or simulation code with this older combined snapshot is not.

## Verified execution

- Full branch test suite, including optional cleaning and CPU RL tests: 47 passed, 8 subtests passed.
- Verified all 15 bundled EDF checksums (38,581,920 bytes).
- Ran the complete fixed-150-microvolt cleaning experiment on S001-S005: all five subjects and all four preprocessing arms completed.
- Reproduced the baseline and individual comparison/amplitude-audit script.
- Trained CPU PPO for 20,000 requested steps, seed 0, and evaluated its four controller conditions on 150 baseline prediction trials. This is a single-seed reproduction, not a robustness study.
- Inspected persisted fitting IDs: ICA and both repair stages used only R04/R08. R12 remains previously inspected exploratory evidence.
- Gym environment checker passed. The configured 1/120-second physics timestep persisted through reset.
- Optional cleaning dependencies were installed into the review worktree's outputs/review_dependencies, not into the shared Python environment. Python 3.14.3 and the existing project environment were used; cleaner versions are copied alongside this report.

## Reproduced cleaning results

| Measure | Causal CSP baseline | Full offline cleaner + CSP |
|---|---:|---:|
| Pooled balanced accuracy, accepted windows | 49.33% | 52.33% |
| Mean individual balanced accuracy, accepted windows | 70.67% | 69.95% |
| Mean individual coverage-adjusted correct fraction | 70.67% | 69.33% |
| Test windows retained | 150/150 | 149/150 |

The small pooled gain is not a consistent individualized improvement. Full cleaning combines filtering, referencing, ICA, interpolation/rejection and a covariance-estimator change, so it does not isolate the effect of artifact removal. Six retained test windows still have unrepaired flags, as the generated audit openly reports. Exclusions are provisional frontal-proxy decisions, not verified ocular-artifact labels.

Do not compare these five-subject results directly against our 15-subject EEGNet average: cohorts, training partitions (R04/R08 here versus R04 adaptation and R08 calibration there), and preprocessing differ.

## Reproduced CPU policy results

| Condition | REST compliance | Grasp success | Balanced success |
|---|---:|---:|---:|
| PPO with decoded commands | 57.33% | 41.33% | 49.33% |
| PPO with correct commands | 100% | 100% | 100% |
| Always HOLD | 100% | 0% | 50% |
| Always grasp | 0% | 100% | 50% |

The toy controller works, but simply reflects decoder errors in this setup. The environment disables jaw-cube collisions, creates a fixed grasp constraint, holds one command fixed per episode, and terminates on lift success. It does not measure realistic contact acquisition, sustained retention after success, or continuous EEG gate HOLD/resume. Those limitations are mostly documented honestly. This is earlier Mario prototype code integrated by Angel, not a new learned Franka capability.

## Useful pieces to integrate selectively

1. scripts/compare_individual_baselines.py and scripts/inspect_eeg_windows.py: amplitude and waveform audits, trial identity checks, and pooled/individual comparisons. These can help investigate signal quality before developing a new gate, without claiming amplitude flags identify artifact type.
2. eeg_decoder/offline_cleaning.py, its configs, optional requirements and tests: retain as a clearly separate offline experimental arm. The coverage accounting and rejection/repair logs are particularly useful.
3. eeg_decoder/baseline.py build_model extraction and eeg_decoder/validation.py run-level selection: reusable structure and tests that keep R12 out of fitting/selection. The historical cohort text needs correction first.
4. pytest.ini's explicit local simulation/src import path: useful to avoid the shared environment importing another checkout. Combine it with our current test-discovery exclusions rather than replacing our configuration.
5. rl_policy: optional CPU teaching/interface fixture only; low priority for the current learned-robot/gate problem.

## Issues and integration prerequisites

- **Stale confirmation claim:** eeg_decoder/validation.py:96 and :117 still generate text describing S006-S010 as planned/reserved confirmation, although the combined handoff correctly says these subjects were already inspected. Correct generated text and historical plan labels before reuse. Otherwise a new run produces misleading evaluation claims.
- **Output preservation:** offline_cleaning.py:188, :232, :295 and :348 reuse directories and overwrite artifacts; the validation runner also uses a fixed output path. Add fresh-run directory requirements or config/hash checks before adopting these in our immutable experiment workflow. This review used an empty isolated worktree.
- **Private dependency API:** FixedVoltageAutoReject calls autoreject.autoreject._run_local_reject_cv. Its explicit version guard and repair/serialization test are good safeguards, and it works with the pinned version. Keep that pin and rerun compatibility tests on upgrades.
- **Causal incompatibility:** sosfiltfilt and full-recording zero-phase filters use future samples. Cleaner artifacts also differ from current replay exports. Reusing them online requires a new causal design and matched model retraining; merely plugging in the saved classifier is invalid.
- **Do not replace current status/robot documentation:** Angel's combined branch incorporates Mario's November 1 snapshot, whereas our branch contains subsequent robot results, EEGNet/CRC work and November 11 evidence.

No blocker was found for reproducing the documented offline experiment. Passing tests and reproducing scores do not establish clean physiological signals, clinical usefulness, independent generalization, or robot safety. No improvement was selected using R12 during this review, and no current model or threshold changed.

## Evidence and reproduction

Review checkout: C:/Users/mario/.gemini/antigravity/scratch/FRI-BCI-Angel-review

Commands executed from that checkout using the existing project Python, with PYTHONPATH pointing to outputs/review_dependencies for cleaning:

```text
python -m pytest -q
python -m scripts.verify_shared_data
python -m eeg_decoder.offline_cleaning --config configs/offline_cleaning_150uv.json
python -m eeg_decoder run
python -m scripts.compare_individual_baselines
python -m rl_policy.train --steps 20000 --seed 0 --output outputs/review_rl/ppo_clean.zip
python -m rl_policy.evaluate --model outputs/review_rl/ppo_clean.zip --predictions outputs/baseline/test_predictions.json --output outputs/review_rl/evaluation.json
```

Compact numerical evidence and the generated cleaning report are copied here. Models, plots, logs and optional dependencies remain under the isolated review checkout's outputs directory. No merge, commit or push was performed.
