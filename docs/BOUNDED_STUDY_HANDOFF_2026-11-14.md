# Bounded EEG-to-robot finish run handoff

## Completed execution (November 14)

The run used the selected `EEGNet8 / transfer / lr 0.0001 / epoch 300` recipe, all three target-training seeds, R04 target fitting, R08-only development, and S001-S015. It did not open R12. The campaign artifacts and R04/R08 caches were hash- and provenance-verified. The study reuses the November 11 CSP/LDA, Riemannian/LDA, transition analysis, utility metrics, and fixed gate settings.

- `eeg_decoder.bounded_study` checks campaign checkpoint/provenance/hash/alignment, creates score and cue probability archives, evaluates six fixed gates per decoder, averages EEGNet seeds within person, selects the gate deterministically, creates causal harness traces and matched schedules, and writes protocol, selection, SHA256 manifests, plots, and the frozen 270-job manifest.
- The selected R08 candidate is EEGNet / moderate hysteresis (threshold .55, margin .02, persistence .25 s): 7/15 participant utility passes, mean REST activation .279, imagery coverage .365. R08 was used for both selection and scoring, so this is exploratory/optimistic.
- The saved Brev hostname was stale. The running L4 host was reached at `global.prd.ga.run.brev.nvidia.com:28952`. The pinned native runtime is Python 3.8, torch 1.13.1+cpu, robomimic 0.1, robosuite 1.2, mujoco_py 2.1.2.14, MuJoCo 2.1. The checkpoint SHA-256 is `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`; harness SHA-256 is `739cc2ca761799006c6c1e45b1cebdb20a830097391481a41b78464c7fd5aa56`.
- The seed-0 always-enable and always-HOLD smoke runs passed task/latch invariants. All 270 fixed jobs (15 participants × 3 robot seeds × 6 conditions) then completed with no job failures. A representative video was saved for S001/seed 0/selected using fixed participant/seed order.
- Aggregated robot results and paired differences are in `results/bounded_eeg_robot_2026-11-14/robot/robot_results/aggregate.json`; per-job telemetry and summaries are in `robot_results_bundle.tar.gz`. The report and plots are in that study directory. SHA-256 checksums are in `SHA256SUMS.json`.
- The selected gate had 42/45 terminal-retention outcomes versus 45/45 for raw EEGNet and 0/45 for always-HOLD. Selected-gate REST end-effector path averaged 0.175 m versus 0.209 m for raw; actual policy calls averaged 2.22 s and requested REST activation was 38.9%. Across matched participant/seed starts, selected minus raw retention was -6.7 percentage points and REST path was -0.034 m. The policy-call change from raw was small because the completion latch ended control after lift acquisition. The selected gate trades some task performance for modestly less REST motion; the evidence does not establish reliable control.
- Seven focused tests cover causal timing/staleness, REST carryover, exact schedule matching, seed averaging, missing seeds, and fixed gate candidates. The final full repository suite passed: `113 passed, 8 subtests passed` (two benign NumPy warnings from a deliberately tiny test fixture).
- The robot bundle, command traces, report, protocol, manifests, plots, and representative video are archived in the [Drive study folder](https://drive.google.com/drive/folders/1ip1eEn_tlL7f5uR8xpkhD7qZxbhWcmG9). The recovered campaign inputs remain under ignored `outputs/campaign_recovered/`, with provenance in `artifact_verification.json`.

## Reproduction

From the repository root, reproduce the robot aggregates from the committed telemetry, traces, schedules, and annotations. The aggregation runs with Python, NumPy, and Matplotlib. Each invocation writes to a fresh directory under `outputs`.

```powershell
$published = 'results\bounded_eeg_robot_2026-11-14'
$replay = Join-Path 'outputs' ('robot_reaggregation_' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $replay -Force | Out-Null
Copy-Item "$published\robot_jobs.json" $replay
Copy-Item "$published\scores","$published\schedules" $replay -Recurse
tar -xzf "$published\command_traces.tar.gz" -C $replay
if ($LASTEXITCODE) { throw 'Could not unpack the command traces' }
tar -xzf "$published\robot_results_bundle.tar.gz" -C $replay
if ($LASTEXITCODE) { throw 'Could not unpack the robot telemetry' }

python simulation\scripts\aggregate_frozen_robot_jobs.py `
  --study $replay --robot-results "$replay\robot_results"
```

To regenerate the EEG comparison from the recovered campaign checkpoints and R04/R08 cache, use the EEG project environment and a new output path. The native robot rollouts and the EEG comparison are separate commands.

```powershell
$eegOutput = Join-Path 'outputs' ('bounded_eeg_replay_' + [guid]::NewGuid())
python -m eeg_decoder.bounded_study `
  --campaign-dir outputs\campaign_recovered `
  --cache-dir outputs\local_scratch_2026-11-11 `
  --output $eegOutput
```

For a robot rerun, provision the exact native environment and checkpoint listed above. Execute the immutable `robot_jobs.json` with `simulation/scripts/run_frozen_robot_jobs.py` and `simulation/src/fri_bci_sim/robomimic_harness.py`. The runner checks that the frozen manifest still contains the same 270 unique jobs and refuses a changed manifest. The current Windows environment cannot run this native simulator build; the tested runtime was Brev. Do not run target R12 or change the gate, robot policy, or simulator in this finish pass.
