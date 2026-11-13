# Post-Submission Amendment to Midterm Report (2026-11-11)

The six submitted midterm report files in this directory (`midterm_report.pdf`, `midterm_report.tex`, `references.bib`, `fri_rl_2026.sty`, `corlabbrvnat.bst`, and `FRI_II_Midterm_Report_Overleaf.zip`) are preserved byte-for-byte as submitted on November 8, 2026.

## Dated Updates and Corrections Since Midterm Submission

1. **Consumption of `S011–S015` `R12` Held-Out Recordings (2026-11-11)**:
   - Section 6 ("Planned Final-Phase Work") of the midterm report designated PhysioNet participants `S011–S015` (`R12`) as an unopened prospective confirmation cohort.
   - On November 11, 2026, `S011–S015` `R12` were opened and evaluated in the 89-source / 15-target `EA-EEGNet` + `CRC-Hysteresis` experiment ([results/eegnet_training_2026-11-11/evaluation_summary.md](../../results/eegnet_training_2026-11-11/evaluation_summary.md)).
   - Consequently, **all `S001–S015` `R12` recordings have now been inspected**, and `S011–S015` can no longer be described as an unopened confirmation cohort in subsequent experiments. Any future unopened-cohort confirmation claim requires a newly pre-registered, disjoint set of subject IDs outside `S001–S015` (and outside the 89 source pretraining subjects if target-disjoint confirmation is claimed).

2. **Retraction of the `(59, 120]` Full-Recording Command-Count Threshold (2026-11-11)**:
   - The earlier `(59, 120]` command-count threshold cited from `results/robomimic_eeg_s002s003_2026-11-06/summary.md` compared full-recording (`123.0–125.0 s`) EEG decision counts against `20.0 s` (`400`-step @ `20 Hz`) robot episodes.
   - On the actual `20.0 s` robot horizon, `S002 confidence` (`59` full-recording events) issued only **7** enable events (`36` steps = `1.80 s`, `0/3` retained), while `S002 raw` (`150` full-recording events) issued **22** enable events (`110` steps = `5.50 s`, `3/3` retained), `S003 temporal` issued **26** (`130` steps = `6.50 s`, `3/3` retained), and `S006 raw` issued **44** (`218` steps = `10.90 s`, `3/3` retained). See [results/project_audit_2026-11-11/horizon_aligned_telemetry_audit.md](../../results/project_audit_2026-11-11/horizon_aligned_telemetry_audit.md) and [docs/NEXT_STEPS_AUDIT_2026-11-11.md](../../docs/NEXT_STEPS_AUDIT_2026-11-11.md).
