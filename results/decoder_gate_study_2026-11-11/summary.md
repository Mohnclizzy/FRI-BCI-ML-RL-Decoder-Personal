# Matched Development-Only Decoder, Transition & Gate Utility Study (`S001–S010`, `R04 -> R08`)

- **Date**: `2026-11-11`
- **Development Cohort**: `S001–S010` (`10` participants)
- **Target Fitting Allowance**: `R04` only (`30` cue-aligned windows per participant)
- **Development Scoring & Calibration Partition**: `R08` only (`30` cue-aligned windows + `4 Hz` continuous sliding-window replay)
- **Data-Split Integrity**: **Zero `R12` recordings or `S011–S015` recordings were loaded or inspected.**

## 1. Matched `R04 -> R08` Decoder Comparison (`S001–S010`)

| Decoder Family | Target Fit Budget | Additional Source Pretraining | Mean `R08` Balanced Acc ($\pm$ SD) | Mean `R08` Macro-F1 | Mean `R08` ROC-AUC | Mean `R08` `HAND` Recall | Mean `R08` `REST` FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `csp_lda_r04` | `R04 only (30 cue-aligned windows/subject)` | None (target R04 only) | **61.00%** ($\pm$ 9.89%) | `0.575` | `0.677` | `0.593` | `0.373` |
| `riemannian_lda_r04` | `R04 only (30 cue-aligned windows/subject)` | None (target R04 only) | **62.67%** ($\pm$ 11.81%) | `0.595` | `0.687` | `0.607` | `0.353` |
| `scratch_ea_eegnet_r04` | `R04 only (30 cue-aligned windows/subject)` | None (target R04 only) | **56.00%** ($\pm$ 8.00%) | `0.490` | `0.620` | `0.573` | `0.453` |
| `transfer_ea_eegnet_r04` | `R04 only (30 cue-aligned windows/subject)` | 89 source subjects (S016-S109 excl. 5 128Hz/anomalous IDs), R04+R08 train, R12 val | **66.33%** ($\pm$ 12.24%) | `0.644` | `0.680` | `0.660` | `0.333` |

### Per-Participant `R08` Cue-Window Balanced Accuracy (`R04` Fit Only)

| Subject | `CSP + LDA (R04)` | `Riemannian + LDA (R04)` | `Scratch EA-EEGNet (R04)` | `Transfer EA-EEGNet (89 src + R04)` |
| --- | ---: | ---: | ---: | ---: |
| `S001` | `70.0%` | `76.7%` | `50.0%` | `76.7%` |
| `S002` | `60.0%` | `56.7%` | `73.3%` | `76.7%` |
| `S003` | `50.0%` | `53.3%` | `60.0%` | `43.3%` |
| `S004` | `86.7%` | `86.7%` | `53.3%` | `86.7%` |
| `S005` | `60.0%` | `53.3%` | `46.7%` | `60.0%` |
| `S006` | `56.7%` | `53.3%` | `53.3%` | `50.0%` |
| `S007` | `60.0%` | `76.7%` | `66.7%` | `70.0%` |
| `S008` | `53.3%` | `53.3%` | `50.0%` | `70.0%` |
| `S009` | `56.7%` | `56.7%` | `50.0%` | `66.7%` |
| `S010` | `56.7%` | `60.0%` | `56.7%` | `63.3%` |

## 2. `Scratch EA-EEGNet` Training-Adequacy Ablation on `R04 -> R08` (`S001–S010`)

Each target participant has `30` `R04` windows, so `batch_size=64, epochs=25` performs `25` gradient updates (`1` batch/epoch). Testing `100` updates (`batch_size=15, epochs=50`) and `200` updates (`batch_size=15, epochs=100`) on `R04 -> R08` shows:

| Regime | Optimizer Updates | Mean `R04` Final Train Loss | Mean `R04` Train Balanced Acc | Mean `R08` Dev Balanced Acc | Mean `R08` Dev Macro-F1 | Mean `R08` Dev ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scratch_25_updates_bs64_ep25` | `25` | `0.6563` | `76.00%` | **56.00%** | `0.490` | `0.620` |
| `scratch_100_updates_bs15_ep50` | `100` | `0.5538` | `99.00%` | **58.00%** | `0.555` | `0.621` |
| `scratch_200_updates_bs15_ep100` | `200` | `0.4027` | `98.00%` | **60.33%** | `0.588` | `0.659` |

## 3. `R08` Continuous Replay Score Distributions: Class-Interior vs Transition Windows

| Decoder Family | Mean Pure `REST` Interior Score | Mean `IMAGERY -> REST` Transition Score (`<2.0 s`) | Mean Pure `IMAGERY` Interior Score | Mean `REST` Episode Max (All) | Mean `REST` Episode Max (Pure Interior Only) | Post-Imagery `REST` Maxima Occurring in `<2.0 s` Transition Zone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `csp_lda_r04` | `0.369` | `0.457` | `0.539` | `0.724` | `0.533` | **59.3%** |
| `riemannian_lda_r04` | `0.374` | `0.471` | `0.585` | `0.771` | `0.586` | **64.3%** |
| `scratch_ea_eegnet_r04` | `0.496` | `0.498` | `0.499` | `0.512` | `0.508` | **56.4%** |
| `transfer_ea_eegnet_r04` | `0.416` | `0.470` | `0.505` | `0.614` | `0.531` | **60.7%** |

## 4. Predeclared `R08` Gate Utility Tradeoff Table (`S001–S010`, Zero `R12` Tuning)

**Predeclared Utility Criteria (rejecting `always-HOLD`)**:
- `enabled_seconds_20s >= 2.50 s` (`>= 50` steps on the 20 Hz robot clock, matching the `42–58` policy-call acquisition requirement),
- `imagery_command_coverage >= 0.30` (`missed_imagery_fraction <= 0.70`), and
- `rest_command_fraction <= 0.35`.

| Decoder | Gate Candidate | Mean Threshold | Active Subj (`/10`) | Subj Passing Utility (`/10`) | Mean `R08` `REST` Cmd Frac ($\pm$ SD) | Mean `R08` Imagery Coverage ($\pm$ SD) | Mean Imagery Episode Det. Rate | Mean Onset Delay (`s`) | Mean `<20s` Enabled (`s`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transfer_ea_eegnet_r04` | `raw_0.50` | `0.500` | `10/10` | **2/10** | `0.428` ($\pm$ 0.159) | `0.525` ($\pm$ 0.194) | `0.880` | `0.74s` | `11.38s` (`10/10` $\ge 2.5$s) |
| `transfer_ea_eegnet_r04` | `confidence_0.70` | `0.700` | `8/10` | **0/10** | `0.061` ($\pm$ 0.072) | `0.062` ($\pm$ 0.066) | `0.333` | `2.26s` | `1.82s` (`2/10` $\ge 2.5$s) |
| `transfer_ea_eegnet_r04` | `temporal_0.70_p0.50s` | `0.700` | `6/10` | **0/10** | `0.032` ($\pm$ 0.049) | `0.027` ($\pm$ 0.039) | `0.147` | `2.05s` | `1.00s` (`1/10` $\ge 2.5$s) |
| `transfer_ea_eegnet_r04` | `frozen_crc_hysteresis_alpha0.10` | `0.754` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `transfer_ea_eegnet_r04` | `interior_only_crc_hysteresis_alpha0.10` | `0.738` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `transfer_ea_eegnet_r04` | `moderate_fixed_hysteresis_0.55` | `0.550` | `10/10` | **2/10** | `0.232` ($\pm$ 0.120) | `0.277` ($\pm$ 0.144) | `0.713` | `1.58s` | `7.53s` (`7/10` $\ge 2.5$s) |
| `transfer_ea_eegnet_r04` | `r08_pure_rest_q80_hysteresis` | `0.575` | `10/10` | **2/10** | `0.171` ($\pm$ 0.054) | `0.203` ($\pm$ 0.100) | `0.600` | `1.84s` | `6.25s` (`9/10` $\ge 2.5$s) |
| `csp_lda_r04` | `raw_0.50` | `0.500` | `10/10` | **2/10** | `0.422` ($\pm$ 0.253) | `0.497` ($\pm$ 0.277) | `0.827` | `0.94s` | `10.15s` (`10/10` $\ge 2.5$s) |
| `csp_lda_r04` | `confidence_0.70` | `0.700` | `10/10` | **2/10** | `0.311` ($\pm$ 0.224) | `0.379` ($\pm$ 0.260) | `0.720` | `1.34s` | `8.00s` (`7/10` $\ge 2.5$s) |
| `csp_lda_r04` | `temporal_0.70_p0.50s` | `0.700` | `10/10` | **2/10** | `0.244` ($\pm$ 0.204) | `0.299` ($\pm$ 0.249) | `0.647` | `1.75s` | `6.70s` (`6/10` $\ge 2.5$s) |
| `csp_lda_r04` | `frozen_crc_hysteresis_alpha0.10` | `0.984` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `csp_lda_r04` | `interior_only_crc_hysteresis_alpha0.10` | `0.923` | `2/10` | **0/10** | `0.003` ($\pm$ 0.007) | `0.009` ($\pm$ 0.019) | `0.033` | `2.35s` | `0.17s` (`0/10` $\ge 2.5$s) |
| `csp_lda_r04` | `moderate_fixed_hysteresis_0.55` | `0.550` | `10/10` | **2/10** | `0.356` ($\pm$ 0.241) | `0.423` ($\pm$ 0.275) | `0.740` | `1.14s` | `8.70s` (`8/10` $\ge 2.5$s) |
| `csp_lda_r04` | `r08_pure_rest_q80_hysteresis` | `0.699` | `10/10` | **2/10** | `0.192` ($\pm$ 0.091) | `0.253` ($\pm$ 0.164) | `0.613` | `1.45s` | `5.95s` (`8/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `raw_0.50` | `0.500` | `10/10` | **3/10** | `0.411` ($\pm$ 0.242) | `0.499` ($\pm$ 0.288) | `0.827` | `1.01s` | `11.05s` (`10/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `confidence_0.70` | `0.700` | `10/10` | **3/10** | `0.357` ($\pm$ 0.238) | `0.459` ($\pm$ 0.298) | `0.780` | `1.19s` | `9.97s` (`10/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `temporal_0.70_p0.50s` | `0.700` | `10/10` | **2/10** | `0.293` ($\pm$ 0.230) | `0.375` ($\pm$ 0.306) | `0.687` | `1.44s` | `8.28s` (`7/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `frozen_crc_hysteresis_alpha0.10` | `1.000` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `interior_only_crc_hysteresis_alpha0.10` | `0.991` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `moderate_fixed_hysteresis_0.55` | `0.550` | `10/10` | **2/10** | `0.352` ($\pm$ 0.236) | `0.442` ($\pm$ 0.296) | `0.793` | `1.25s` | `9.88s` (`10/10` $\ge 2.5$s) |
| `riemannian_lda_r04` | `r08_pure_rest_q80_hysteresis` | `0.747` | `10/10` | **3/10** | `0.271` ($\pm$ 0.190) | `0.361` ($\pm$ 0.265) | `0.707` | `1.42s` | `8.47s` (`9/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `raw_0.50` | `0.500` | `10/10` | **0/10** | `0.425` ($\pm$ 0.300) | `0.464` ($\pm$ 0.332) | `0.827` | `0.69s` | `8.22s` (`7/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `confidence_0.70` | `0.700` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `temporal_0.70_p0.50s` | `0.700` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `frozen_crc_hysteresis_alpha0.10` | `0.525` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `interior_only_crc_hysteresis_alpha0.10` | `0.522` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `moderate_fixed_hysteresis_0.55` | `0.550` | `0/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.000` ($\pm$ 0.000) | `0.000` | `-` | `0.00s` (`0/10` $\ge 2.5$s) |
| `scratch_ea_eegnet_r04` | `r08_pure_rest_q80_hysteresis` | `0.508` | `1/10` | **0/10** | `0.000` ($\pm$ 0.000) | `0.002` ($\pm$ 0.005) | `0.007` | `2.05s` | `0.10s` (`0/10` $\ge 2.5$s) |

## 5. Key Findings & Scientific Conclusions (`R04 -> R08` Development Only)

1. **Fair Matched-Budget Decoder Ranking (`S001–S010`, `R04` Fit Only)**:
   - Under an identical single-run target fitting budget (`30` `R04` windows per participant), **`Transfer EA-EEGNet`** achieves the highest `R08` balanced accuracy (**`66.33%`**, Macro-F1 `0.644`), followed by **`Riemannian Tangent-Space + LDA`** (**`62.67%`**, Macro-F1 `0.595`, ROC-AUC `0.687`) and **`CSP + LDA`** (**`61.00%`**, Macro-F1 `0.575`, ROC-AUC `0.677`), while the frozen `25`-update **`Scratch EA-EEGNet`** reaches **`56.00%`**.
   - Notably, **`Riemannian + LDA`** and **`CSP + LDA`** require **zero source-subject pretraining** and fit in `< 1 s` per participant on CPU, whereas `Transfer EA-EEGNet`'s `+3.66` to `+5.33` percentage-point advantage on `R08` relies on pretraining across `89` additional source participants (`5,296` windows).
2. **Training-Adequacy Check on `Scratch EA-EEGNet`**:
   - The frozen `Scratch EA-EEGNet` protocol (`batch_size=64, epochs=25` on `30` `R04` windows = `25` gradient updates) reached only `76.00%` training accuracy on `R04` (`loss = 0.6563`), leaving output probabilities compressed near `0.50` (`0.496–0.499`).
   - Increasing to `100` updates (`batch_size=15, epochs=50`) and `200` updates (`batch_size=15, epochs=100`) fits `R04` to `98.00–99.00%` training accuracy and raises `R08` development balanced accuracy from **`56.00%`** to **`58.00%`** (`100` updates) and **`60.33%`** (`200` updates) — matching single-run `CSP + LDA` (`61.00%`) while still trailing `Transfer EA-EEGNet` (`66.33%`) by `6.00` points.
3. **Why Continuous Replay Scores Differ from Cue-Aligned Windows (Transition vs Interior)**:
   - Across all `140` post-imagery `REST` episodes on `S001–S010` `R08`, **`59.3%` to `64.3%`** of `REST` episode maximum scores occur within the first **`2.0 s` after `REST` onset**, where the causal `2.0 s` sliding window still overlaps the preceding `HAND_IMAGERY` interval or carries over its final command.
   - However, transition overlap is **not** the sole cause of `CRC-Hysteresis` suppression: even when `CRC` (`alpha = 0.10`, `15` `REST` episodes) is calibrated strictly on **pure `REST` interior maxima** (`interior_only_crc_hysteresis_alpha0.10`), the finite-sample penalty `(errors + 1) / (15 + 1) <= 0.10` still requires `0/15` episode exceedances (`1/16 = 0.0625`), setting the threshold above the single highest interior `REST` spike (`mean threshold = 0.738` on `Transfer EA-EEGNet`, `0.991` on `Riemannian + LDA`) and yielding **`0/10` active subjects**.
4. **Predeclared Gate Utility Frontier (No Unbounded Sweep)**:
   - Fixed `0.70` thresholds (`confidence_0.70`, `temporal_0.70_p0.50s`) fail on `Transfer EA-EEGNet` (`0/10` passing utility; mean `<20s` enabled duration `1.00–1.82 s`) because `EEGNet`'s max-norm weight constraints (`1.0` spatial, `0.25` linear) compress output probabilities relative to shrinkage `LDA`.
   - Scale-aware gates (`moderate_fixed_hysteresis_0.55` and `r08_pure_rest_q80_hysteresis`) restore command delivery across `10/10` development participants (`7/10` to `9/10` delivering `>= 2.50 s` of `ENABLE` inside the `20.0 s` robot episode horizon and reducing mean `R08` `REST` command fraction from `0.428` to `0.171–0.232`), but only **`2/10`** (`Transfer EA-EEGNet` / `CSP+LDA`) to **`3/10`** (`Riemannian+LDA`) participants simultaneously satisfy all three predeclared utility bounds (`>= 2.50 s` in `20 s`, `>= 30%` imagery coverage, `<= 35%` `REST` command fraction).
