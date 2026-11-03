# FRI-BCI-ML-RL-Decoder

Public EEG motor-imagery decoding research for simulated robotic-hand control using machine learning and reinforcement learning.

For the earlier project timeline, read [project history](docs/PROJECT_HISTORY.md).
For the latest cross-workstream status, read the [project check-in](docs/PROJECT_CHECK_IN_2026-11-03.md).
For cross-account repository access and reproduction, read [sharing and reproduction](docs/SHARING_AND_REPRODUCTION.md).

## Simulation scaffold

The [Isaac Lab project scaffold](simulation/README.md) connects causal recorded
EEG replay, three command gates, a HOLD-preserving executor, a scripted Franka
lift skill, measured cube/contact telemetry, and a separate PPO evaluation.
The historical first matched five-person EEG pilot had an invalid HOLD control,
so its physical lift counts do not support a comparison. The corrected matched
batch remains incomplete. Robot-only learning has progressed from an in-hand
lift to a near-table policy with 2/5 sampled physical acquisitions in a fixed
scene, but one acquisition dropped and the policy had early terminations and
high finger forces. EEG is not connected to that learned policy. The recorded
command scores are separate from physical motion.
Read the
[measured cohort results](docs/multi_person_replay_2026-11-02.md),
[PPO playback results](docs/ppo_lift_evaluation_2026-11-02.md), and
[Brev/TACC runbook](simulation/cloud/README.md).

## EEG intent decoder

First milestone: load real public EEG, train a rest-versus-hand-imagery decoder,
and explain its errors on a recording it has not seen. Everything runs on CPU.

## Start on Windows

Run these commands in this project folder. The explicit executable path avoids
PowerShell activation-policy problems.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m eeg_decoder run
```

The run command downloads 15 EDF recordings: subjects 1-5, runs 4, 8, 12,
with up to three downloads at a time.
Data stays under `data/physionet/`. Re-running reuses cached recordings.
Dependencies and data downloads require internet access. Model training does not.
The tested environment is Python 3.14.3 on Windows; `requirements-lock.txt`
records its exact package versions. Install that file instead of `requirements.txt`
to reproduce this environment. Point VS Code's Python interpreter at
`.venv/Scripts/python.exe` for editing and debugging.

You can also separate the stages:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoder prepare
.\.venv\Scripts\python.exe -m eeg_decoder train
.\.venv\Scripts\python.exe -m pytest -q
```

## What the model sees

For these runs, the experiment instructs a person to rest, imagine opening and
closing their left fist, or imagine opening and closing their right fist.

| EDF annotation | Participant instruction | Our class |
|---|---|---|
| T0 | Rest within the task recording | REST (0) |
| T1 | Imagine opening/closing the left fist | HAND_IMAGERY (1) |
| T2 | Imagine opening/closing the right fist | HAND_IMAGERY (1) |

The label is the experiment instruction, not proof of a person's exact thought.
We preserve original left/right annotations for later analysis. We do not label
these trials as wrist flexion, twist, or an isolated grasp: this dataset does not
provide those labels. Runs 6, 10, 14 include foot imagery and are excluded here.
Rest comes from the same task recordings, not separate eyes-closed recordings,
which could introduce an easy but misleading difference between conditions.

Each training example is `[9 channels, 320 samples]`: 2 seconds at 160 Hz.
One window starts 1 second after each annotation onset. A window is skipped if
it would cross a state boundary or run end. This reduces cue-onset contamination,
but task cues can still influence results; it is a cue-based first benchmark.

## Processing and training

1. MNE reads EDF samples in volts and standardizes electrode names.
2. Select the fixed central channels in `configs/baseline.json`.
3. Subtract the mean of those selected channels at each sample.
4. Apply a causal, fourth-order Butterworth 8-30 Hz bandpass to each full run.
5. Extract labeled windows without crossing annotations.
6. Fit four CSP spatial features and shrinkage LDA on runs 4 and 8 only.
7. Evaluate once on held-out run 12, then compare with a majority-class model.

CSP learns combinations of electrodes whose signal variance helps distinguish
the two classes. LDA uses those features to predict a class and probabilities.
There is no normalization fitted to the complete dataset, no random split of
overlapping windows, and no model fitting on the held-out run.

This first experiment uses a pooled model and the SAME five people across train
and test, but different recordings. It does not test unseen-person performance.
For an additional subject holdout, copy the config to a new experiment config,
set `test_subjects` to `[5]`, choose a different `output_dir`, and run with
`--config path/to/your/config.json`. All runs from person 5 will be held out.
Keep each test partition fixed and tune only with validation inside training data.

## Read your results

Generated artifacts in `outputs/baseline/`:

- `results.md`: summary and per-person scores.
- `diagnostics.png`: confusion matrix and C3 spectra by class.
- `metrics.json`: complete metrics, actual package versions, and split trial IDs.
- `data_audit.json`: class counts, source files, and skipped windows.
- `metadata.json`: trace each example back to its subject/run/annotation.
- `test_predictions.json`: model output for each held-out window.
- `model.joblib`: trained CSP/LDA and its preprocessing configuration.
- `windows.npz`: prepared arrays `X` and `y`.

Balanced accuracy averages recall of rest and imagery; 0.50 is the binary
majority baseline when both classes are present. Macro-F1 considers missed
and incorrect predictions for both classes. In the confusion matrix, rows are
true classes and columns are predicted classes. `REST -> HAND_IMAGERY` is a
false activation: it matters when a robot could move during instructed rest.
This window fraction is not yet false commands per minute in continuous replay.

The spectral plot describes class-average EEG; a visible difference is not proof
that the model detects intended movement or generalizes. There is no full artifact
rejection yet. Inspect data quality before scaling to more people.

## Interface with your partner

The initial vocabulary is REST and HAND_IMAGERY. Agree on a simulator behavior
mapping with your partner separately. For example, hand imagery might *select*
a grasp primitive; that would be a mapped command, not decoded grasp imagery.

For later replay, a prediction message will contain the source subject/run,
window start/end times, intent, and probability per class. The current
`test_predictions.json` provides this structure for held-out windows, but it is
an offline export, not a real-time stream or accepted robot command. Probabilities
are uncalibrated; choosing a confidence cutoff needs validation.

Replay must use the exact channel order, selected-channel reference, and filter.
Carry SOS filter state across chunks; reset at each new recording, not each window.
The timestamp when a prediction becomes available is at least its window end,
plus computation delay. Never give the model annotation labels as input.

## Your next milestone

Run the fixed development/confirmation protocol with the optional covariance
baseline installed:

```powershell
python -m pip install -r requirements-riemannian.txt
python -m eeg_decoder.validate_decoder --output outputs/decoder_validation_001
```

Use a new output directory for each independent run. Model family is selected
on subjects 1–5 (fit run 4, validate run 8), then frozen before subjects 6–10
are evaluated on run 12 after individual calibration on runs 4/8. Once inspected,
that confirmation cohort cannot be reused as fresh evidence after tuning.
The current result is recorded in [decoder findings](docs/decoder_comparison_2026-11-01.md).

Compare pooled CSP/LDA with a separately calibrated model for each person using
the same held-out run and unchanged preprocessing:

```powershell
python -m eeg_decoder prepare --config configs/baseline.json
python -m eeg_decoder.compare_subjects --config configs/baseline.json --output outputs/subject_comparison
```

The output directory must be new. This exports per-person models compatible
with the replay adapter and metrics for both approaches on identical windows.
Use each model only with its calibration subject. This is an exploratory
diagnostic on the previously inspected test run, not independent confirmation
or proof of improvement. It does not select hyperparameters using test scores.

Read `results.md` and identify which errors dominate. Inspect a few original
windows and annotations using their metadata. Then compare the same fixed
protocol with a Riemannian baseline, followed by a separate subject-holdout
experiment. Only after those checks add continuous replay and command filtering.
Synthetic tests verify software behavior; they do not establish EEG performance.

## References

The [literature reference summary](docs/literature_reference_summary.md) covers
all 18 sources cited in the literature review, with citation mappings, methods,
findings, limitations, verified links, and project relevance.

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [MNE CSP motor-imagery example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [MNE EEGBCI loader](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)
- [SciPy causal SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)
