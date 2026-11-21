# BCI Competition IV Dataset 1 — Irregular-Timing Request Pilot & Robot Replay (2026-11-19)

## Provenance and Verification

- **Dataset:** BCI Competition IV Dataset 1 (human participants ds1a, ds1b, ds1f, ds1g)
- **100 Hz Signal Archive (`BCICIV_1_mat.zip`):** SHA-256 `f7acc9420158e6d58629589040ced64b49ffd8e946c4024f85cc77896acdf686`
- **1000 Hz Official Evaluation Labels (`true_labels.zip`):** SHA-256 `b34de8373d6476ff534f3a7496397ed86726c04dd12f610d84b8baaa71e28c72`
- **Robot Harness:** `robomimic-latch6-v2.0.0` (SHA-256 `739cc2ca761799006c6c1e45b1cebdb20a830097391481a41b78464c7fd5aa56`)
- **Robot Policy Checkpoint (`lift_ph_low_dim_epoch_1000_succ_100.pth`):** SHA-256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`
- **240 Start-Offset/Seed Rollout Bundle (`robomimic_240_rollouts_bundle.tar.gz`):** SHA-256 `39a8035775c2e66e4f68162d8a9263bda818f97a6d44f3b3a73c295aae24fa5e`
- **Harness Invariants Across All 240 Rollouts (`80` 4 Hz start steps `1..396` × `3` seeds `0..2`):** `all_hold_invariants_pass = True`, `all_latch_invariants_pass = True`, `all_contact_proxy_validated = True`, `any_post_acquisition_loss = False` (`0` post-acquisition drops across all 240 rollouts).

## 1. Full-Stream Request Detection (Reserved Evaluation Runs 2–4, Pre-Scheduler)

Each participant's classical Riemannian Tangent-Space + Shrinkage LDA decoder (`9` sensorimotor channels `C3, Cz, C4, C1, C2, FC1, FC2, CP1, CP2`, causal `8–30 Hz` Butterworth `sosfilt`, trailing `2.0 s` window at `4 Hz`) was fitted on **Evaluation Run 1**, and controller parameters were frozen on Run 1 (using 5-fold chronological out-of-fold probabilities with a `2.0 s` boundary guard) under a shared `<= 3.0` false starts / REST minute budget. Controllers ran continuously across **Evaluation Runs 2–4** (`180` imagery episodes per person, `720` total); the 1-second post-cue `NaN` transition masks were applied only when scoring.

| Participant | Method (Frozen from Run 1) | Emitted Total (`Img 1st+Dup / REST / Trans`) | Imagery Episodes Detected (`Missed`) | Detection Rate | False Starts / REST Min (`Count / Min`) | Mean Onset Delay: Detected (s) | Mean Onset Delay: With Misses (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`ds1a`** | Periodic timer (`T=15.0s`) | `115` (`40+0 / 46 / 29`) | `40 / 180` (`140`) | 22.2% | `3.99` (`46 / 11.5m`) | 2.24 s | 3.13 s |
| **`ds1a`** | EEG threshold (`p>=0.91`) | `95` (`48+0 / 26 / 21`) | `48 / 180` (`132`) | 26.7% | `2.25` (`26 / 11.5m`) | 2.15 s | 2.98 s |
| **`ds1a`** | EEG accumulator (`A>=1.05`) | `132` (`68+2 / 38 / 24`) | `68 / 180` (`112`) | 37.8% | `3.29` (`38 / 11.5m`) | 1.89 s | 2.62 s |
| **`ds1b`** | Periodic timer (`T=16.5s`) | `106` (`34+0 / 53 / 19`) | `34 / 180` (`146`) | 18.9% | `4.79` (`53 / 11.1m`) | 2.49 s | 3.52 s |
| **`ds1b`** | EEG threshold (`p>=0.91`) | `87` (`51+1 / 16 / 19`) | `51 / 180` (`129`) | 28.3% | `1.45` (`16 / 11.1m`) | 2.29 s | 3.25 s |
| **`ds1b`** | EEG accumulator (`A>=0.97`) | `128` (`69+3 / 21 / 35`) | `69 / 180` (`111`) | 38.3% | `1.90` (`21 / 11.1m`) | 2.12 s | 2.96 s |
| **`ds1f`** | Periodic timer (`T=19.0s`) | `90` (`35+0 / 36 / 19`) | `35 / 180` (`145`) | 19.4% | `3.16` (`36 / 11.4m`) | 2.07 s | 3.05 s |
| **`ds1f`** | EEG threshold (`p>=0.79`) | `84` (`49+1 / 15 / 19`) | `49 / 180` (`131`) | 27.2% | `1.32` (`15 / 11.4m`) | 1.98 s | 2.88 s |
| **`ds1f`** | EEG accumulator (`A>=0.5`) | `108` (`63+2 / 26 / 17`) | `63 / 180` (`117`) | 35.0% | `2.29` (`26 / 11.4m`) | 1.92 s | 2.66 s |
| **`ds1g`** | Periodic timer (`T=18.75s`) | `90` (`40+0 / 33 / 17`) | `40 / 180` (`140`) | 22.2% | `3.18` (`33 / 10.4m`) | 2.34 s | 3.38 s |
| **`ds1g`** | EEG threshold (`p>=0.82`) | `118` (`63+2 / 32 / 21`) | `63 / 180` (`117`) | 35.0% | `3.08` (`32 / 10.4m`) | 1.70 s | 2.92 s |
| **`ds1g`** | EEG accumulator (`A>=0.97`) | `101` (`55+0 / 31 / 15`) | `55 / 180` (`125`) | 30.6% | `2.98` (`31 / 10.4m`) | 2.12 s | 3.15 s |

## 2. Robot Readiness Schedule & Verified Continuous-Retention Lift Outcomes (`333` Slots)

Every exported timestamped request was passed through the consecutive nonoverlapping 20-second slot schedule (`83 + 85 + 82 + 83 = 333` complete 20-second slots across Runs 2–4). Under the one-request-starts-one-lift contract, the first request in a 20-second slot starts the policy at 20 Hz step `s_start = round(t_rel * 20) + 1`, and subsequent requests in that slot are rejected as busy. Continuous post-acquisition retention (`first_acquisition_step is not None and not post_acquisition_loss and terminal_contact and terminal_rise_m >= 0.08`) is verified from the versioned harness rollouts averaged across robot seeds `0, 1, 2`.

| Participant (`20 s` Slots) | Method | Accepted Starts (`Img / REST / Trans`) | Rejected: Busy (`Img / REST / Trans`) `[+Tail]` | Continuously Retained Lifts (`Intended Img / Unwanted REST / Unscored Trans`) | Post-Acquisition Drops |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **`ds1a`** (`83`) | Periodic timer | **`83`** (`31 / 31 / 21`) | `28` (`8 / 13 / 7`) `[+4]` | **`83.0`** (`31.0 / 31.0 / 21.0`) | `0` |
| **`ds1a`** (`83`) | EEG threshold | **`64`** (`32 / 17 / 15`) | `27` (`15 / 8 / 4`) `[+4]` | **`60.0`** (`28.0 / 17.0 / 15.0`) | `0` |
| **`ds1a`** (`83`) | EEG accumulator | **`76`** (`39 / 20 / 17`) | `50` (`30 / 15 / 5`) `[+6]` | **`71.0`** (`36.0 / 19.0 / 16.0`) | `0` |
| **`ds1b`** (`85`) | Periodic timer | **`85`** (`28 / 42 / 15`) | `19` (`5 / 11 / 3`) `[+2]` | **`85.0`** (`28.0 / 42.0 / 15.0`) | `0` |
| **`ds1b`** (`85`) | EEG threshold | **`62`** (`33 / 15 / 14`) | `25` (`19 / 1 / 5`) `[+0]` | **`56.0`** (`29.0 / 13.0 / 14.0`) | `0` |
| **`ds1b`** (`85`) | EEG accumulator | **`75`** (`38 / 13 / 24`) | `53` (`34 / 8 / 11`) `[+0]` | **`71.0`** (`37.0 / 11.0 / 23.0`) | `0` |
| **`ds1f`** (`82`) | Periodic timer | **`82`** (`33 / 32 / 17`) | `6` (`2 / 2 / 2`) `[+2]` | **`76.0`** (`30.0 / 30.0 / 16.0`) | `0` |
| **`ds1f`** (`82`) | EEG threshold | **`55`** (`33 / 10 / 12`) | `27` (`17 / 4 / 6`) `[+2]` | **`51.0`** (`32.0 / 10.0 / 9.0`) | `0` |
| **`ds1f`** (`82`) | EEG accumulator | **`65`** (`45 / 13 / 7`) | `40` (`20 / 12 / 8`) `[+3]` | **`63.0`** (`43.0 / 13.0 / 7.0`) | `0` |
| **`ds1g`** (`83`) | Periodic timer | **`83`** (`38 / 28 / 17`) | `6` (`2 / 4 / 0`) `[+1]` | **`77.0`** (`34.0 / 27.0 / 16.0`) | `0` |
| **`ds1g`** (`83`) | EEG threshold | **`74`** (`41 / 18 / 15`) | `44` (`24 / 14 / 6`) `[+0]` | **`71.7`** (`40.0 / 17.7 / 14.0`) | `0` |
| **`ds1g`** (`83`) | EEG accumulator | **`66`** (`39 / 16 / 11`) | `35` (`16 / 15 / 4`) `[+0]` | **`65.0`** (`38.0 / 16.0 / 11.0`) | `0` |

## 3. Pooled Summary Across All Four Human Participants (`333` Slots, `720` Imagery Episodes)

| Metric | Periodic timer | EEG threshold | EEG accumulator |
| :--- | ---: | ---: | ---: |
| **Emitted requests total** | `401` | `384` | `469` |
| — First detections during imagery (`+ duplicates`) | `149` (`+0`) | `211` (`+4`) | `255` (`+7`) |
| — Emitted during scored REST | `168` | `89` | `116` |
| — Emitted during unscored `1 s` transitions | `84` | `80` | `91` |
| **Accepted robot starts (`out of 333 slots`)** | **`333`** | **`255`** | **`282`** |
| — **During instructed imagery** | **`130`** | **`139`** | **`161`** |
| — **During scored REST** | **`133`** | **`60`** | **`62`** |
| — **During unscored transitions (`start-cue / stop-cue`)** | `70` (`39 / 31`) | `56` (`28 / 28`) | `59` (`27 / 32`) |
| **Rejected because robot busy (`+ incomplete tail`)** | `59` (`+9`) | `123` (`+6`) | `178` (`+9`) |
| **Continuously retained lifts: total (seed mean)** | **`321.0`** | **`238.7`** | **`270.0`** |
| — **Continuously retained lifts started during imagery** | **`123.0`** | **`129.0`** | **`154.0`** |
| — **Continuously retained lifts started during scored REST** | **`130.0`** | **`57.7`** | **`59.0`** |
| — **Continuously retained lifts started during transitions** | `68.0` | `52.0` | `57.0` |
| **Post-acquisition losses (drops across all seeds/starts)** | `0` | `0` | `0` |

## 4. Key Diagnostics and Failure Accounting

1. **Continuous retention verified:** Running all `240` `(seed, window_start)` combinations through `robomimic_harness.py` (`robomimic-latch6-v2.0.0`, line 877) confirmed **`0` post-acquisition drops** (`post_acquisition_loss = False` in `240/240` rollouts). Once acquired, the completion latch maintained bilateral contact and `>= 0.08 m` cube rise on every subsequent frame through step `400`. Consequently, the continuous-retention counts exactly equal the terminal-retention counts (`154.0` vs. `129.0` vs. `123.0` lifts started during imagery; `59.0` vs. `57.7` vs. `130.0` started during scored REST).
2. **Missed imagery is the dominant bottleneck (`465` undetected + `94` unaccepted out of `720`):** For the frozen EEG accumulator, **`465 / 720` (`64.6%`)** of instructed imagery episodes had no detected request at all, and another **`94 / 720` (`13.1%`)** had a valid detection that never became an accepted start because the robot was already busy in that 20-second slot (or in the incomplete tail), leaving **`161 / 720` (`22.4%`)** episodes with an accepted start.
3. **Short vs. long imagery cues:** Across the `720` held-out imagery episodes, `115` had full cue durations `<= 2.5 s` (unmasked duration `<= 1.5 s`) and `605` lasted `> 2.5 s`. The accumulator detected only **`10 / 115` (`8.7%`)** of short episodes versus **`245 / 605` (`40.5%`)** of longer episodes.
4. **Post-stop transition audit correction:** Of the accumulator's `59` accepted transition requests, `27` occurred in the `1.0 s` window after a start cue and `32` occurred in the `1.0 s` window after a stop cue. Only **`2 / 32`** accepted post-stop requests crossed a 20-second slot boundary between the stop cue and the request; the remaining `30` occurred in slots that were still idle when the imagery episode ended without having triggered during the imagery interior.
5. **Slot-rotation timing control (`282` starts):** Keeping all `282` accepted accumulator starts and their exact slot-relative offsets while circularly rotating slots within each recording yielded **`114.3`** imagery starts, **`108.3`** REST starts, and **`59.4`** transition starts on average, compared with the observed **`161`** imagery and **`62`** REST starts.
