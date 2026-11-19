# FRI-BCI-ML-RL-Decoder

**Research question:** *When should noisy, non-stationary motor-imagery EEG command a simulated robotic pickup?*

## What this repository investigates & attribution

Across two completed simulation studies, we evaluate how causal EEG decoding and command-timing rules govern false starts during rest versus task execution during motor imagery:

- **Project-built contributions:** Causal `8–30 Hz` bandpass EEG preprocessing across 9 sensorimotor channels, three decoder families (`CSP/LDA`, `Riemannian Tangent-Space + LDA`, `EA-EEGNet` transfer), continuous hysteresis/confidence/temporal gates and a leaky evidence accumulator, the `robomimic-latch6-v2.0.0` `HOLD` + `+0.015 m` completion-latch adapter ([`simulation/src/fri_bci_sim/robomimic_harness.py`](simulation/src/fri_bci_sim/robomimic_harness.py)), timing/capacity audits, and replay evaluation harnesses.
- **External assets (credit & scope):** Neural recordings come from public datasets — [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/) (`S001–S015`) and [BCI Competition IV Dataset 1](https://www.bbci.de/competition/iv/desc_1.html) (`ds1a, ds1b, ds1f, ds1g`). The physical lifting skill is the **externally pretrained** `robomimic` Lift PH low-dimensional `BC-RNN` policy checkpoint (`artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth`, SHA-256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`) running in `robosuite 1.2` / `MuJoCo 2.1`; this project did not train that `robomimic` checkpoint.

## Completed studies & headline evidence

| Study & Denominator | Core Question | Measured Headline Result | Evidence & Summary |
| :--- | :--- | :--- | :--- |
| **Study 1: Continuous Gating**<br>PhysioNet EEGBCI (`15` participants `S001–S015`; `270` native rollouts = `15` × `3` shared seeds × `6` conditions) | What happens when EEG continuously enables or holds the lift policy over `20 s`? | Selected EEGNet moderate hysteresis gate reduced R08 REST activation `40.1% -> 27.9%` and imagery coverage `52.1% -> 36.5%` (`7/15` passed utility screen). In `270` rollouts: **`42/45`** retained lifts vs. **`45/45`** raw (all 3 misses from `S006`, given `1.0 s` total enable) and mean REST EEF path **`0.175 m`** vs. **`0.209 m`**. R04 and R08 share `100%` identical cue schedules (`7,278/7,278`), motivating irregular timing. | [Study 1 summary](results/bounded_eeg_robot_2026-11-14/summary.md) · [Tradeoff figure](results/bounded_eeg_robot_2026-11-14/command_tradeoff.png) · [Aggregate JSON](results/bounded_eeg_robot_2026-11-14/robot/robot_results/aggregate.json) · [Nov 18 review](results/project_review_2026-11-18/review.md) |
| **Study 2: Irregular-Timing Requests**<br>BCI Competition IV Dataset 1 (`4` real humans `ds1a, ds1b, ds1f, ds1g`; `720` held-out imagery episodes in Runs 2–4; `333` complete `20 s` slots; `240` native rollouts = `80` start offsets × `3` seeds) | Can EEG time a single start-lift request better than a clock when cues arrive unpredictably? | Under variable timing (clock baselines `50.63%` / `50.01%`), the frozen EEG accumulator emitted `255` first detections (`469` total requests) and achieved **`161`** accepted imagery starts / **`62`** REST / **`59`** transition → **`154.0 / 59.0 / 57.0`** continuously retained lifts (3-seed mean, `0` post-acquisition drops), vs. EEG threshold `139 / 60 / 56` (`129.0 / 57.7 / 52.0`) and periodic timer `130 / 133 / 70` (`123.0 / 130.0 / 68.0`). Wins `4/4` humans, while missing **`465/720`** imagery episodes (`64.6%`) plus `94` busy/tail-blocked detections. | [Study 2 summary](results/bciciv_ds1_request_pilot_2026-11-19/summary.md) · [Study 2 overview figure](report/final/figures/bciciv_ds1_pilot_overview.png) · [Pilot metrics JSON](results/bciciv_ds1_request_pilot_2026-11-19/pilot_metrics.json) · [Clock audit JSON](report/final/bciciv_ds1_clock_audit.json) · [240 rollout summaries](results/bciciv_ds1_request_pilot_2026-11-19/robomimic_240_summaries.json) |

## Key entry points, runners & representative media

- **Final release deliverables:** [Final Report (PDF)](report/final/final_report.pdf) · [Final Report Source (`TeX`)](report/final/final_report.tex) · [CPU Reproduction Guide](docs/REPRODUCTION_GUIDE.md)
- **Roadmap & current status:** [docs/ANTIGRAVITY_ROADMAP.md](docs/ANTIGRAVITY_ROADMAP.md) · [STATUS.md](STATUS.md) · [START_HERE.md](START_HERE.md) · [results/README.md](results/README.md) · [Midterm report context](report/midterm/)
- **Main experiment runners & artifact builders:**
  - Study 1 EEG + continuous gate sweep: [eeg_decoder/bounded_study.py](eeg_decoder/bounded_study.py)
  - Timing & capacity audits: [analysis/audit_cue_clock.py](analysis/audit_cue_clock.py) · [analysis/request_capacity.py](analysis/request_capacity.py) · [analysis/audit_bciciv_ds1_clock.py](analysis/audit_bciciv_ds1_clock.py)
  - Study 2 frozen EEG request pilot (BCICIV-1): [analysis/bciciv_ds1_request_pilot.py](analysis/bciciv_ds1_request_pilot.py)
  - Study 2 240-rollout native `robomimic-latch6-v2.0.0` grid: [simulation/scripts/run_one_request_window_grid.py](simulation/scripts/run_one_request_window_grid.py)
  - Final report figure & manifest builder: [analysis/build_final_report_artifacts.py](analysis/build_final_report_artifacts.py)
- **Representative robot video:** [Study 1 PhysioNet `R08`, participant `S001`, robot `seed 0`, `selected` EEGNet moderate gate (`complete_retained`, acquisition at step `137` / `6.85 s`)](results/bounded_eeg_robot_2026-11-14/robot/robot_results/S001_seed0_selected/video.mp4).

## Scope & honest limitations

- **Recorded cued imagery, not self-paced or live closed-loop BCI:** Both datasets present timed visual or spoken task cues; labels reflect experimental instructions *at request time*, not native grasp-intent verification or online feedback adaptation.
- **Simulation only:** Robot rollouts share `3` initial states (`seeds 0, 1, 2`) on CPU MuJoCo under an engineered completion latch (`+0.015 m` $z$-offset after acquisition); no hardware robot or clinical safety claim is made.
- **Study 1 selection & timing caveats:** Gate selection and evaluation both used PhysioNet `R08` (optimistic exploratory selection), and `R04`/`R08` shared identical predictable cue timings; the November 14 dense-window run (`62.52%` BA) used `178` vs. `42` source updates/epoch (not budget-matched).
- **Study 2 cohort size & missed cues:** Evaluates `N = 4` real humans (`ds1c–e` excluded as artificial); the frozen accumulator still missed `465/720` imagery episodes (`10/115` short cues `<= 2.5 s` detected).

---

## Historical notes and earlier setup

> **November 19 closeout:** The irregular-timing request pilot and 240 versioned-harness (`robomimic-latch6-v2.0.0`) robot replays on BCI Competition IV Dataset 1 (`ds1a, ds1b, ds1f, ds1g`) are packaged in [results/bciciv_ds1_request_pilot_2026-11-19/summary.md](results/bciciv_ds1_request_pilot_2026-11-19/summary.md) with runner scripts [analysis/bciciv_ds1_request_pilot.py](analysis/bciciv_ds1_request_pilot.py) and [simulation/scripts/run_one_request_window_grid.py](simulation/scripts/run_one_request_window_grid.py).

> [November 18 engineering review](results/project_review_2026-11-18/review.md) documents a correction to the historical dense training budget, a timing-only EEG control, and the feasibility limit for the proposed request scheduler. Read it before running the supervisor experiment.

> **November 18, 2026 — agent handoff:** Start with [the Antigravity handoff](docs/ANTIGRAVITY_HANDOFF_2026-11-18.md) for verified robomimic policy attribution, completed November 14 results, artifact recovery, and the proposed learned EEG supervisory policy. The supervisor is a proposal, not an implemented result. Older entries below are historical.

> **November 14 update:** The bounded EEG command comparison and all 270 robot replays are complete. Read the [completed results](results/bounded_eeg_robot_2026-11-14/summary.md) and [reproduction handoff](docs/BOUNDED_STUDY_HANDOFF_2026-11-14.md). The [research follow-up handoff](docs/NEXT_RESEARCH_HANDOFF_2026-11-14.md) specifies the proposed diagnostic, repeated-request experiment, conditional training ablation, and independent confirmation requirements. That follow-up is planned, not yet implemented.

> **Next research diagnostic:** The saved R08 stream audit and focused prior-art scan are complete in [results/next_research_diagnostic_2026-11-14/](results/next_research_diagnostic_2026-11-14/). It finds both elevated transition scores and weak state-interior separation. Do not present a replay of cued labels as self-paced intent control.

> **Navigation & Reproducibility Update (2026-11-11):** For current project status, verified November 11 artifacts, and the forward execution plan, start at [STATUS.md](STATUS.md) and [docs/NEXT_STEPS_AUDIT_2026-11-11.md](docs/NEXT_STEPS_AUDIT_2026-11-11.md). See [results/eegnet_training_2026-11-11/evaluation_summary.md](results/eegnet_training_2026-11-11/evaluation_summary.md) for the 89-source / 15-target `EA-EEGNet` + `CRC-Hysteresis` evaluation (`S001–S015` `R12` are now inspected) and [results/project_audit_2026-11-11/horizon_aligned_telemetry_audit.md](results/project_audit_2026-11-11/horizon_aligned_telemetry_audit.md) for the 20-second horizon-aligned robot telemetry audit and `robomimic-latch6-v2.0.0` control harness. Sections below are preserved for historical context.

Public EEG motor-imagery decoding research for simulated robotic-hand control using machine learning and reinforcement learning.

For the earlier project timeline, read [project history](docs/PROJECT_HISTORY.md).
For the current execution plan and handoff, read [the autonomous project plan](docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md) and [START_HERE](START_HERE.md).
Earlier cross-workstream findings are in the [project check-in](docs/PROJECT_CHECK_IN_2026-11-03.md).
For cross-account repository access and reproduction, read [sharing and reproduction](docs/SHARING_AND_REPRODUCTION.md).

## Project files and downloads

- [Repository layout and recovery guide](docs/REPOSITORY_LAYOUT.md)
- [Current execution status](STATUS.md)
- [Midterm report and source](report/midterm/)
- [Raw experiment archives and data](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal/releases/tag/local-consolidation-2026-11-08)
- [Historical local files](archive/)

## Simulation scaffold

The [Isaac Lab project scaffold](simulation/README.md) connects causal recorded
EEG replay, three command gates, a HOLD-preserving executor, a scripted Franka
lift skill, measured cube/contact telemetry, and a separate PPO evaluation.
The historical first matched five-person EEG pilot had an invalid HOLD control,
so its physical lift counts do not support a comparison. The corrected matched
120-second batch is complete: raw 5/5 lifts, confidence 3/5, temporal 2/5, using
the frozen scripted skill. See the [corrected results](results/corrected_eeg_2026-11-03-complete/summary.md)
for unintended motion, missed imagery and delay as well as completion.
Robot-only learning reached 3/5 acquired and retained lifts in a fixed near-table
diagnostic screen. This does not establish reliable farther-ready-pose pickup.
EEG is not connected to that learned policy, and learned HOLD/resume is untested.
See the [latest evidence handoff](docs/HANDOFF_2026-11-03_AFTER_RETENTION.md).
The next route is native pretrained-policy reproduction, early HOLD/resume
validation, and recorded-EEG integration, as specified in the execution handoff.
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

## Reproduce the completed decoder experiments

These commands document the existing decoder workflow; they are not the current
project's next phase. Use the execution handoff above for ongoing work.

Reproduce the fixed development/confirmation protocol with the optional covariance
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

When reproducing these experiments, use `results.md` and window metadata to
inspect the errors. Continuous replay and command filtering are already
implemented; learned-controller integration is the current gap. Further decoder
comparisons need an explicit development protocol and unused confirmation data.
Synthetic tests verify software behavior; they do not establish EEG performance.

## References

The [literature reference summary](docs/literature_reference_summary.md) covers
all 18 sources cited in the literature review, with citation mappings, methods,
findings, limitations, verified links, and project relevance.

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [MNE CSP motor-imagery example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [MNE EEGBCI loader](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)
- [SciPy causal SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)
