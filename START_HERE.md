# Start here: EEG-to-Franka project

**Research question:** *When should noisy motor-imagery EEG command a simulated robotic pickup?*

For the finished release deliverables, read the [Final Report (PDF)](report/final/final_report.pdf) ([TeX source](report/final/final_report.tex)), the [CPU Reproduction Guide](docs/REPRODUCTION_GUIDE.md), and [STATUS.md](STATUS.md); see [README.md](README.md) for the full project entry point and [docs/ANTIGRAVITY_ROADMAP.md](docs/ANTIGRAVITY_ROADMAP.md) for release scope.

## Ownership & Completed Evidence

- **Contributions:** The project team built the causal EEG decoders (`CSP/LDA`, `Riemannian+LDA`, `EA-EEGNet` transfer), command gating/accumulator controllers, timing/capacity audits, and `robomimic-latch6-v2.0.0` `HOLD` + completion-latch simulation wrapper ([`simulation/src/fri_bci_sim/robomimic_harness.py`](simulation/src/fri_bci_sim/robomimic_harness.py)). The lift skill itself is the **externally pretrained** `robomimic` Lift PH low-dimensional `BC-RNN` policy (`artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth`, SHA-256 `3ee222ca...`), applied to public EEG recordings (`PhysioNet EEGBCI` and `BCI Competition IV Dataset 1`).
- **Study 1 — Continuous Enable/HOLD Gating (`PhysioNet EEGBCI`, `15` participants, `270` rollouts):** Selected EEGNet moderate hysteresis gate reduced R08 REST activation `40.1% -> 27.9%` and imagery coverage `52.1% -> 36.5%` (`7/15` passed utility screen; exploratory `R08` selection), retaining `42/45` lifts vs. `45/45` raw (`0.175 m` vs. `0.209 m` REST EEF path). [Study 1 summary](results/bounded_eeg_robot_2026-11-14/summary.md) · [Nov 18 review (`R04 == R08` clock & slot ceiling)](results/project_review_2026-11-18/review.md).
- **Study 2 — Irregular-Timing Single-Request Pilot (`BCI Competition IV Dataset 1`, `4` real humans, `720` imagery episodes, `240` rollouts):** Frozen EEG accumulator yielded `161` accepted imagery starts / `62` REST / `59` transition (`154.0 / 59.0 / 57.0` continuously retained lifts, 3-seed mean, `0` drops) vs. EEG threshold `139 / 60 / 56` (`129.0 / 57.7 / 52.0`) and periodic timer `130 / 133 / 70` (`123.0 / 130.0 / 68.0`), while missing `465/720` episodes (plus `94` busy/tail-blocked). [Study 2 summary](results/bciciv_ds1_request_pilot_2026-11-19/summary.md) · [Study 2 overview figure](report/final/figures/bciciv_ds1_pilot_overview.png) · [Clock audit](report/final/bciciv_ds1_clock_audit.json).
- **Main runners & index:** [Results index](results/README.md) · [Study 1 runner](eeg_decoder/bounded_study.py) · [PhysioNet clock audit](analysis/audit_cue_clock.py) · [Slot capacity audit](analysis/request_capacity.py) · [BCICIV-1 clock audit](analysis/audit_bciciv_ds1_clock.py) · [Study 2 EEG pilot](analysis/bciciv_ds1_request_pilot.py) · [Study 2 native 240-rollout grid](simulation/scripts/run_one_request_window_grid.py) · [Report artifact builder](analysis/build_final_report_artifacts.py).

## Limitations to keep in mind

Recorded, acoustically/visually cued imagery evaluated offline in MuJoCo simulation (`3` shared robot seeds); robot outcomes reflect instructed states *at request time*; PhysioNet R08 suffers from selection bias and identical `R04 == R08` cue timing; BCICIV-1 evaluates `N = 4` real humans (`ds1a, ds1b, ds1f, ds1g`) with `64.6%` missed episodes (`10/115` short cues `<= 2.5 s`).

---

## Historical execution notes (November 3–11, 2026)

> **Navigation updated November 11, 2026:** Use [STATUS.md](STATUS.md) and [docs/NEXT_STEPS_AUDIT_2026-11-11.md](docs/NEXT_STEPS_AUDIT_2026-11-11.md) for current execution status, the [November 11 EA-EEGNet + CRC-Hysteresis evaluation](results/eegnet_training_2026-11-11/evaluation_summary.md) (note: `S011–S015` `R12` were opened on 2026-11-11 and are no longer an unopened confirmation cohort), and the [20-second horizon-aligned robot telemetry audit](results/project_audit_2026-11-11/horizon_aligned_telemetry_audit.md). Earlier navigation from November 8, 2026 ([repository layout](docs/REPOSITORY_LAYOUT.md)) and the plans/results below are retained as dated historical context.

The [November 3 execution handoff](docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md) and its [evidence handoff](docs/HANDOFF_2026-11-03_AFTER_RETENTION.md) describe the runs available at that stage.

## Objective and working repository (historical November 3 plan)

Build a reproducible simulation demonstration in which recorded EEG enables, pauses and resumes a learned Franka table-pickup skill. PhysioNet labels instructed rest and fist imagery, not grasp-specific intent. The required first interface is continuous enable/HOLD/resume. Pretrained policies, imitation learning and an isolated simulator change are authorized; preserve existing Isaac work and the frozen scripted reference.

Source of record: [Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal), branch `codex/robot-learning-design`. Inspect actual remotes/history before modifying a checkout. Angel's repository is reference-only; do not modify it.

## Last verified results

- Individually calibrated CSP/LDA achieved 59.33% balanced accuracy on S006-S010 confirmation recordings. These recordings are now inspected; subsequent analysis is exploratory.
- The corrected five-person, three-gate 120-second EEG-to-scripted-robot batch is **complete**: raw 5/5 lifts, confidence 3/5, temporal 2/5. Read unintended motion, missed imagery and delay alongside completion. [Summary and plots](results/corrected_eeg_2026-11-03-complete/summary.md)
- The frozen scripted always-HOLD, uninterrupted-lift and pause/resume controls passed.
- Original PPO lifted in 5/5 in-hand starts; the farther Approach table evaluation was 0/5. Different starts prevent treating those as a paired comparison.
- The strongest later learned checkpoint acquired and retained the cube in 3/5 fixed near-table diagnostic episodes. This is not a qualified farther-ready-pose pickup controller. The half-arm inference ablation failed 0/5 and was rejected.
- The previously tested NVIDIA native candidate produced 1/5 sustained lifts in fresh-process evaluations. The newly proposed robomimic candidate has **not** been reproduced here.
- EEG has not yet controlled a learned robot policy; learned HOLD/resume remains untested.

## Next actions

Follow the [execution handoff](docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md):

1. Preserve source/artifacts, verify recoverable off-host copies, and verify continuing model-account and GitHub access.
2. Reproduce the official robomimic low-dimensional Lift policy in its matching isolated native environment.
3. Test physical HOLD/resume early, then connect the frozen recorded-EEG pipeline.
4. Qualify the complete demonstration and package commands, video, plots and reproducible artifacts.

Do not reset/clean/force-push the dirty Brev checkout or overwrite old outputs. Latest off-host raw-artifact recovery and independent CLI publication are unverified. Keep one current STATUS.md as execution proceeds.

## Evidence and historical records

- [Results index](results/README.md)
- [Latest detailed evidence and artifact paths](docs/HANDOFF_2026-11-03_AFTER_RETENTION.md)
- [Research protocol](docs/research_protocol.md)
- [Sharing and reproduction](docs/SHARING_AND_REPRODUCTION.md)
- [Older artifact release](docs/ARTIFACT_HANDOFF.md) — does not establish backup of later runs
- [Project history](docs/PROJECT_HISTORY.md)

Older check-ins and startup instructions may describe then-incomplete work. Preserve them as history; use the current handoffs for status and execution order.
