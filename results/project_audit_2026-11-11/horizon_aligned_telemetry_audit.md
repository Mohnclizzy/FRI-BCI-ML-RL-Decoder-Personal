# 20-Second Horizon-Aligned Exposure, Annotation & Telemetry Audit (2026-11-11)

- **Harness Version**: `robomimic-latch6-v2.0.0` (`simulation/src/fri_bci_sim/robomimic_harness.py`)
- **Robot Episode Clock**: `400` steps (`1..400`) at `20.0 Hz` (`[0.0, 20.0)` s)
- **Saved Rollouts Audited**: `42` (`14` subject/condition groups $\times$ `3` seeds `0..2`)
- **Key Distinction**:
  1. **Full-recording enable events** count 4 Hz EEG windows over the entire `123.0–125.0 s` EDF file.
  2. **20s-horizon requested enable steps** count steps in `1..400` where `CausalDecisionReplay` requests `ENABLE` (`issued_at_s < 20.0` s).
  3. **Executed policy steps** count steps where the robot policy actually ran (`mode == "policy"`) before the post-acquisition completion latch (`mode == "latch"`) took over.

## 1. Horizon-Aligned Summary Across All 42 Saved Robomimic Rollouts

| Subject | Condition / Gate | Retained | Full-Trace Events (123–125s) | `<20s` Events | `<20s` Requested Steps (`s`) | `<20s` Bouts | Executed Policy Steps (seeds `0,1,2`) | Latch Step (seeds `0,1,2`) | Post-Latch Overridden Enable / Hold Steps (mean) | `<20s` `REST` Requested Rate | `<20s` Pre-Latch `REST` EEF Path (m, mean) | `<20s` `HAND_IMAGERY` Missed Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| `S001` | `always-enable` | `3/3` | - | - | 400 (20.00s) | 1 | `[58, 54, 47]` | `[58, 54, 47]` | 347.0 / 0.0 | 1.000 | 0.360 | 0.000 |
| `S001` | `always-hold` | `0/3` | - | - | 0 (0.00s) | 0 | `[0, 0, 0]` | `[-, -, -]` | 0.0 / 0.0 | 0.000 | 0.029 | 1.000 |
| `S001` | `eeg / raw` | `3/3` | 227 | 66 | 330 (16.50s) | 1 | `[50, 42, 44]` | `[90, 82, 84]` | 284.7 / 30.0 | 0.703 | 0.353 | 0.000 |
| `S001` | `eeg / confidence` | `3/3` | 170 | 54 | 272 (13.60s) | 4 | `[50, 42, 44]` | `[90, 82, 84]` | 226.7 / 88.0 | 0.585 | 0.353 | 0.183 |
| `S001` | `eeg / temporal` | `3/3` | 128 | 47 | 236 (11.80s) | 3 | `[50, 42, 44]` | `[100, 92, 94]` | 190.7 / 114.0 | 0.483 | 0.279 | 0.256 |
| `S002` | `eeg / raw` | `3/3` | 150 | 22 | 110 (5.50s) | 5 | `[52, 44, 48]` | `[298, 201, 294]` | 62.0 / 73.7 | 0.254 | 0.184 | 0.695 |
| `S002` | `eeg / confidence` | `0/3` | 59 | 7 | 36 (1.80s) | 3 | `[36, 36, 36]` | `[-, -, -]` | 0.0 / 0.0 | 0.110 | 0.200 | 0.939 |
| `S002` | `eeg / temporal` | `0/3` | 35 | 3 | 16 (0.80s) | 1 | `[16, 16, 16]` | `[-, -, -]` | 0.0 / 0.0 | 0.068 | 0.183 | 1.000 |
| `S003` | `eeg / raw` | `3/3` | 483 | 72 | 360 (18.00s) | 1 | `[50, 42, 44]` | `[90, 82, 84]` | 314.7 / 0.0 | 0.831 | 0.353 | 0.000 |
| `S003` | `eeg / confidence` | `3/3` | 365 | 43 | 212 (10.60s) | 11 | `[51, 47, 48]` | `[155, 151, 152]` | 163.3 / 84.0 | 0.559 | 0.218 | 0.512 |
| `S003` | `eeg / temporal` | `3/3` | 308 | 26 | 130 (6.50s) | 5 | `[49, 44, 45]` | `[191, 186, 187]` | 84.0 / 128.0 | 0.458 | 0.287 | 0.866 |
| `S006` | `eeg / raw` | `3/3` | 120 | 44 | 218 (10.90s) | 4 | `[50, 42, 44]` | `[136, 128, 130]` | 172.7 / 96.0 | 0.364 | 0.029 | 0.195 |
| `S006` | `eeg / confidence` | `0/3` | 6 | 5 | 24 (1.20s) | 1 | `[24, 24, 24]` | `[-, -, -]` | 0.0 / 0.0 | 0.000 | 0.029 | 0.854 |
| `S006` | `eeg / temporal` | `0/3` | 3 | 3 | 14 (0.70s) | 1 | `[14, 14, 14]` | `[-, -, -]` | 0.0 / 0.0 | 0.000 | 0.029 | 0.915 |

## 2. Why the Earlier `(59, 120]` Command-Threshold Claim Was Retracted

- The earlier report (`results/robomimic_eeg_s002s003_2026-11-06/summary.md`) observed that `S002 confidence` (`59` full-recording commands over `123.0 s`) failed (`0/3` retained) while `S006 raw` (`120` full-recording commands over `123.0 s`) succeeded (`3/3` retained), and inferred a critical command-count threshold in `(59, 120]`.
- Aligning the decision traces to the actual `20.0 s` (`400`-step) robot episode horizon shows that the robot never sees commands issued at `t >= 20.0 s`:
  - `S002 confidence` (`59` full-recording events) issued only **7** enable events (`36` requested steps = `1.80 s`, `3` bouts) inside `[0, 20.0)` s, executing `[36, 36, 36]` policy calls — fewer than the `42–58` policy calls required for grasp acquisition.
  - `S002 raw` (`150` full-recording events) issued only **22** enable events (`110` requested steps = `5.50 s`, `5` bouts) inside `[0, 20.0)` s (`128` of its `150` events occurred at `t >= 20.0 s`), yet succeeded (`3/3` retained) with `[52, 44, 48]` executed policy calls (`latch_step` at `[298, 201, 294]`).
  - `S003 temporal` (`308` full-recording events) issued **26** enable events (`130` requested steps = `6.50 s`, `5` bouts) inside `[0, 20.0)` s and succeeded (`3/3` retained) with `[49, 44, 45]` executed policy calls (`latch_step` at `[191, 186, 187]`).
  - `S006 raw` (`120` full-recording events) issued **44** enable events (`218` requested steps = `10.90 s`, `4` bouts) inside `[0, 20.0)` s and succeeded (`3/3` retained) with `[50, 42, 44]` executed policy calls (`latch_step` at `[136, 128, 130]`).
- Thus, on the actual `20.0 s` robot horizon, the transition occurs between **7** `<20s` enable events (`36` steps = `1.80 s`, `S002 confidence`, `0/3`) and **22** `<20s` enable events (`110` steps = `5.50 s`, `S002 raw`, `3/3`), with every successful condition requiring `42–58` executed policy calls (`2.10–2.90 s`) before the completion latch takes over.

## 3. Pre-Generated Matched Control Schedules (`matched_control_schedules_20s.json`)

| Subject | Gate | `<20s` Requested Steps (`s`) | `<20s` Bouts | Bout Lengths (steps) | Contiguous Duration-Matched `enable-window` (`start, steps`) |
| --- | --- | ---: | ---: | --- | --- |
| `S001` | `raw` | 330 (16.50s) | 1 | `[330]` | `(41, 330)` |
| `S001` | `confidence` | 272 (13.60s) | 4 | `[190, 50, 6, 26]` | `(41, 272)` |
| `S001` | `temporal` | 236 (11.80s) | 3 | `[180, 40, 16]` | `(51, 236)` |
| `S002` | `raw` | 110 (5.50s) | 5 | `[20, 20, 4, 30, 36]` | `(137, 110)` |
| `S002` | `confidence` | 36 (1.80s) | 3 | `[4, 6, 26]` | `(147, 36)` |
| `S002` | `temporal` | 16 (0.80s) | 1 | `[16]` | `(361, 16)` |
| `S003` | `raw` | 360 (18.00s) | 1 | `[360]` | `(41, 360)` |
| `S003` | `confidence` | 212 (10.60s) | 11 | `[20, 4, 14, 4, 70, 10, 4, 4, 16, 6, 60]` | `(41, 212)` |
| `S003` | `temporal` | 130 (6.50s) | 5 | `[10, 4, 60, 6, 50]` | `(51, 130)` |
| `S006` | `raw` | 218 (10.90s) | 4 | `[110, 64, 4, 40]` | `(87, 218)` |
| `S006` | `confidence` | 24 (1.20s) | 1 | `[24]` | `(257, 24)` |
| `S006` | `temporal` | 14 (0.70s) | 1 | `[14]` | `(267, 14)` |

## 4. November 11 `EA-EEGNet` `R12` 20-Second Horizon Exposure (`S001–S015`)

Note: `S011–S015` `R12` were evaluated on 2026-11-11 and are no longer an unopened confirmation cohort.

| Subject | Partition | `raw` `<20s` Steps (`REST` / `IMAGERY` Rate) | `confidence` `<20s` Steps (`REST` / `IMAGERY` Rate) | `temporal` `<20s` Steps (`REST` / `IMAGERY` Rate) | `crc_hysteresis` `<20s` Steps (`REST` / `IMAGERY` Rate) |
| --- | --- | --- | --- | --- | --- |
| `S001` | Dev (`S001–S010`) | `5` (`0.02` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S002` | Dev (`S001–S010`) | `230` (`0.52` / `0.66`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S003` | Dev (`S001–S010`) | `135` (`0.25` / `0.46`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S004` | Dev (`S001–S010`) | `360` (`0.83` / `1.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S005` | Dev (`S001–S010`) | `20` (`0.04` / `0.06`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S006` | Dev (`S001–S010`) | `335` (`0.79` / `0.91`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S007` | Dev (`S001–S010`) | `255` (`0.61` / `0.68`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S008` | Dev (`S001–S010`) | `5` (`0.02` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S009` | Dev (`S001–S010`) | `255` (`0.56` / `0.74`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S010` | Dev (`S001–S010`) | `190` (`0.40` / `0.59`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S011` | Evaluated 2026-11-11 (`S011–S015`) | `340` (`0.79` / `0.94`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S012` | Evaluated 2026-11-11 (`S011–S015`) | `305` (`0.70` / `0.85`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S013` | Evaluated 2026-11-11 (`S011–S015`) | `105` (`0.09` / `0.51`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S014` | Evaluated 2026-11-11 (`S011–S015`) | `115` (`0.20` / `0.41`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
| `S015` | Evaluated 2026-11-11 (`S011–S015`) | `355` (`0.81` / `1.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) | `0` (`0.00` / `0.00`) |
