# Connected EEG→robot bottleneck — Stage 1 FROZEN protocol + manifest

Status: **FROZEN (prospective).** Written 2026-11-05 by the sole experiment writer,
per `docs/CONNECTED_BOTTLENECK_STUDY_2026-11-05.md` and `docs/research_protocol.md`.
This document fixes the operating point, controls, metrics, endpoints, independence
units, and budget **before** any S002/S003 run-12 decode is run. Nothing in this
document may be re-tuned against run-12 outcomes.

---

## 1. What this freeze is (and the discipline)

- **Prospective:** the operating point, split, gates, thresholds, seeds, budget, and
  success criteria below are fixed now, before S002/S003 run-12 is decoded or replayed.
  S001 is already opened (preserved anchor). S002/S003 are fresh only while uninspected.
- **No cherry-picking:** if a held-out subject fails, the failure is reported. We do not
  re-tune the decoder/gates/threshold/latch to make a held-out subject pass, and we do not
  inspect run-12 to choose which decoder or gate to test.
- **Reproducible:** all inputs are pinned by sha256 (Section 6). A new machine can rerun
  every command verbatim.
- **Honest:** independence unit, the per-person calibration caveat, and the dev-vs-held-out
  distinction are stated explicitly (Section 4). This is **not** a live BCI, not a
  hardware/clinical-safety result, and not an uncalibrated-new-person generalization claim.

## 2. Stage 1 goal (bounded)

Finish the **existing-system** evaluation and localize the EEG→robot bottleneck:
quantify (a) the EEG-side command signal (activation false positives, duty cycle, timing)
and (b) the robot-side outcome (latch completion, retention, post-acquisition loss), and
**decompose causally** how much of the outcome difference is due to the EEG decoder/timing
versus the replay regime/policy — using the harness's own deterministic control conditions.

Stages 2–6 (CSP/Riemannian/EEGNet comparison, demonstration→policy action mapping,
BC→PPO vs PPO-continuation, Ofner labels/mapping, full connect) are **out of scope** for
this freeze and are only entered after Stage 1 evidence is published.

## 3. Asset inventory (verified 2026-11-05, host `ubuntu`)

| Subject | Raw EEG (EDF) | Decision trace (run-12) | Status |
|---|---|---|---|
| S001 | **absent** | `~/S001R12-decisions.jsonl` (493 rows) | **OPENED** anchor (held-out) |
| S002 | **absent** | **absent** | FRESH — no local asset; needs PhysioNet download + frozen per-person decode |
| S003 | **absent** | **absent** | FRESH — no local asset; needs PhysioNet download + frozen per-person decode |
| S006 | MNE cache | `outputs/isaac/S006R12-decisions.jsonl` | opened dev (training subject) |
| S007 | MNE cache | `outputs/isaac/S007R12-decisions.jsonl` | opened dev (training subject) |
| S008 | MNE cache | `outputs/isaac/S008R12-decisions.jsonl` | opened dev (training subject) |
| S009 | MNE cache | `outputs/isaac/S009R12-decisions.jsonl` | opened dev (training subject) |
| S010 | MNE cache | `outputs/isaac/S010R12-decisions.jsonl` | opened dev (training subject) |

Raw EDFs live in the MNE PhysioNet cache (run 12 only; calibration runs 4/8 are not cached):
`data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S0xx/S0xxR12.edf`.

**S002/S003 blocker (honest):** no local raw EEG, not in the MNE cache, and **no PhysioNet
credential** is present on this host. Acquiring S002/S003 requires (a) a PhysioNet cred +
network download of `S002R12.edf`/`S003R12.edf` (+ calibration runs 4/8 for per-person
calibration), then (b) a **frozen** per-person CSP+LDA decode to produce their run-12
decision traces. Until (a)+(b) are done, the full 3-subject held-out set is **partial
(S001 of 3)** — this is preserved, not hidden.

## 4. Independence units (honest)

- **Decoder split unit = the recording (subject × run).** `eeg_decoder/baseline.py::
  split_indices` enforces `train_runs & test_runs == ∅` (no recording in both partitions).
  Held-out mode: `held_out_subjects` (S002/S003 ∉ S006–S010) AND held-out run 12
  (vs calibration runs 4/8).
- **Per-person calibration caveat (preserved from S001):** each held-out subject's decoder
  is calibrated on that **same subject's** runs 4/8. This is **not** uncalibrated-new-person
  generalization. "Held-out" means held-out subject (∉ S006–S010) and held-out run (12 vs 4/8).
  No cross-subject calibration-transfer claim is made.
- **Robot-replay seeds (0,1,2) are nuisance replicates** of the same start distribution, not
  independent subjects. **The honest inference unit for robot outcome is the subject**
  (S001/S002/S003); seed is pooled to estimate start-placement noise. Effects are therefore
  reported per subject, and per subject × gate, not as 27/18/9 independent trials.
- **Dev vs held-out:** S006–S010 numbers are **development/tuning** numbers (the decoder was
  built and compared on them). Only S001 (opened) and, once decoded, S002/S003 are held-out.
  A headline "pass rate" that mixes dev + held-out is **forbidden**.

## 5. Frozen operating point (no post-freeze tuning)

| Item | Frozen value | Pin / source |
|---|---|---|
| Decoder | MNE `CSP` + `LinearDiscriminantAnalysis(solver=lsqr, shrinkage=auto)` | `eeg_decoder/baseline.py` |
| Channels | C3, Cz, C4, C1, C2, FC1, FC2, CP1, CP2 (average-referenced) | `configs/baseline.json` |
| Preprocess | 4th-order Butterworth bandpass **8–30 Hz**, 160 Hz, filter state reset per run | `baseline.py::preprocess_run` |
| Window | 2.0 s, 1.0 s onset offset, one window/annotation, inside-state only | `baseline.py::extract_windows` |
| CSP | `n_components=4`, `reg=0.1`, rank = channels−1 | `configs/baseline.json` |
| **Calibration split** | per-person: runs **4/8 train**, run **12 held out** | `IMAGERY_RUNS={4,8,12}` |
| Labels | T0→REST, T1/T2→HAND_IMAGERY (2-class) | `baseline.py::LABELS` |
| **Gates** | `raw`=argmax; `confidence`=p_HAND≥0.7; `temporal`=p_HAND≥0.7 + 0.5 s persistence | existing gate logic |
| Decision latency | 0.05 s (50 ms), `available_at_s = window_end_s + 0.05` | existing traces |
| **Policy** | Phase 2 BC-RNN `lift_ph_low_dim_epoch_1000_succ_100.pth` | `…/robomimic-native-phase2-20261104/artifacts/…` |
| Policy sha256 | `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605` | verified 2026-11-05 |
| Regime | always-ENABLE / HOLD, latch z-offset **+0.015 m** (frozen once at acquisition) | `--latch-z-offset 0.015` |
| Horizon | **400 steps @ 20 Hz (20 s)**, native robosuite/MuJoCo, **CPU only** | `latch6_rollout.py` |
| Seeds | **0, 1, 2** | matched to S001/S006 |
| **Harness** | `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py` | unchanged |
| **No** | no live EEG, no new robot, no Franka, no retraining, no new latch params | study scope |

## 6. Frozen input manifest (sha256)

- Policy checkpoint: `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`
- S001 decision trace (opened anchor): `84bab7c63594bf91dbbeb2eb0b20d9beed6d80664457878fae21383569871566`
- Dev raw EDFs (opened, run 12):
  - S006 `296b5cb14549098d…`  S007 `70cd3287d6b48216…`  S008 `ddced4ba4dc80131…`
  - S009 `0c3f9700d6bfd8a8…`  S010 `0983609a12e6fd9b…`
  (full 64-hex recorded in `results/connected_bottleneck_stage1_2026-11-05/manifest.json`)
- S002/S003: **not yet present** — hashes to be recorded at acquisition time, before decode.

## 7. Matched activation/timing controls (the causal design)

The harness exposes four **deterministic** conditions that are the matched controls:

- `--condition always-hold` → 0 % command duty (HOLD baseline).
- `--condition always-enable` → 100 % command duty (always-on baseline).
- `--condition pause --pause-start S --pause-steps L` → a **deterministic timed trigger**:
  policy ENABLED for a contiguous block of `L` steps, disabled elsewhere. This is the
  **matched activation/timing control**: we set the enabled-block onset and length to match a
  chosen EEG trace's first-activation step and duty cycle, so the EEG arm and the control arm
  see the **same regime, same policy, same seeds, same latch, same horizon** and differ only
  in *how the command is issued* (causal EEG signal vs deterministic schedule).
- `--condition eeg --gate {raw,confidence,temporal} --decisions <trace>` → the causal EEG arm.

**Causal read (per subject × gate):**
1. If the `pause`-matched control (same onset + duty) **completes** but the `eeg` arm does not
   → the bottleneck is the **EEG decoder/timing** (duty cycle, false activations, or latency),
   not the regime/policy.
2. If the `eeg` arm and the matched control **agree** → the regime/policy is the dominant
   condition (EEG is carrying a usable command at that duty level).
3. `always-hold` (0 %) vs `always-enable` (100 %) bound the duty-cycle axis; the EEG arm's
   position within that band, per subject, is the subject-specific activation bottleneck.

## 8. Causal metrics

**EEG-side (from the decision trace; no robot):**
- `rest_false_activation_fraction` — REST windows decoded as HAND_IMAGERY (needs annotation
  labels; computed at decode time on the held-out run).
- **command duty cycle** per gate (fraction of windows issuing `grasp_lift`) — per subject × gate.
- **first-activation step** (`issued_at_s` of first `grasp_lift`) and **temporal flips** (gate
  output changes) — timing/stability.
- mean/min/max `p_hand`.

**Robot-side (causal, from `latch6_rollout.py` summary):**
- `native_success_any`, `completion_state` (`complete_retained`/`complete_unretained`),
  `terminal_rise_m`, `terminal_contact`, `terminal_retained`, `post_acquisition_loss`,
  `contact_proxy_validated`, latch invariants, HOLD invariants, first-acquisition step,
  longest bilateral-contact steps.

**Causal decomposition (the study's key number):**
- `Δ_outcome(subject×gate) = outcome(eeg_arm) − outcome(pause_matched_control)` — isolates the
  EEG decoder/timing effect from the regime effect.
- **Bottleneck localization rule:** `|Δ|` large & `pause`-control passes ⇒ EEG-side bottleneck;
  `|Δ|` small ⇒ regime/policy is the dominant condition; both fail ⇒ regime/policy bottleneck
  (EEG not the limiter); both pass ⇒ regime sufficient offline, remaining gap is live latency /
  not captured offline.

## 9. Endpoints (frozen)

- **Primary:** `complete_retained` rate per subject (EEG arm vs matched control) and the causal
  `Δ` per subject × gate — this localizes the bottleneck (EEG vs regime/policy).
- **Secondary:** `rest_false_activation_fraction` (held-out), command duty cycle, first-activation
  latency, temporal-flip rate, `terminal_rise_m`, `post_acquisition_loss`.
- **Safety/validity gates (must pass to report a "complete"):** latch invariants, HOLD
  invariants, no post-acquisition loss, `contact_proxy_validated`.

## 10. Budget and stop conditions

- **Budget:** ≤ **2 CPU·h** robot time, CPU-only, no new robot. Current committed runs: S001
  EEG arm (9, already run, preserved) + S001 matched controls (≤9) + S001/S002/S003 EEG arms
  (9 each, when decoded) + matched controls (≤9 each). Staged, one subject/batch at a time;
  stop and report if the running total approaches 2 CPU·h.
- **Stop / escalate (no silent retry):** PhysioNet cred missing (S002/S003 acquisition blocked);
  harness import/MuJoCo failure; any held-out failure (report, do not re-tune); budget near cap.
- **Not started (requires owner authority):** any live EEG, new robot, Franka, retraining, new
  latch parameter, or an unbudgeted multi-hour sweep.

## 11. Execution plan (one bounded phase at a time)

- **Phase A (this phase):** inventory + this freeze + EEG-side duty-cycle/timing table on the
  existing (opened) traces + execution-path dry-run. → *done this phase.*
- **Phase B:** S001 matched-control replay (Phase 7 causal read): `pause`-matched control at
  S001's raw-gate onset/duty + `always-enable` + `always-hold`, seeds 0–2. Report `Δ`.
- **Phase C (blocked pending cred):** acquire S002/S003 `R12` (+ runs 4/8), record hashes, then
  **frozen** per-person decode → run-12 traces → EEG arm + matched control, seeds 0–2 each.
- **Phase D:** consolidate held-out causal table (S001 + S002 + S003), publish compact evidence
  via isolated Git integration, update STATUS.md.
- **Stages 2–6:** only after D is published (decoder comparison, action mapping, BC→PPO, Ofner,
  full connect) — each its own bounded phase + freeze.

## 12. Honest labels (preserved)

- This is **offline replay** of **pre-decoded** recorded-EEG decision streams through the Phase 2
  learned policy under the Phase 6 latch regime, native robosuite/MuJoCo on CPU, 20 s horizon.
- **Not** a live BCI, **not** a hardware/clinical/physical-safety result, **not** an
  uncalibrated-new-person generalization claim, and (until S002/S003 are decoded) **not** the full
  3-subject held-out set.
- Rest motion, drop rate, missed-command rate, and participant-level uncertainty are **not** measured here.

## 13. Verification against project documents (self-check)

- `research_protocol.md` §2 (reproducible, sha-pinned, CPU-only, ≤2 CPU·h, no live EEG, stop
  conditions): **conforms** (Sections 5, 6, 10).
- `research_protocol.md` §3 (no hidden success criteria / no cherry-picking): **conforms**
  (Section 1, 9).
- `CONNECTED_BOTTLENECK_STUDY_2026-11-05.md` Stage-1 requirements (frozen operating point, matched
  activation/timing controls, new robot starts, causal metrics, honest independence units,
  honest labels, endpoints + budget ≤2 CPU·h, no new robot): **all covered** (Sections 5, 7, 8, 4,
  12, 9, 10). "New robot starts" = fresh `robosuite` reset per run (400 steps each); no new *hardware*
  robot is started (consistent with "no new robot").
- `literature_reference_summary.md`: the per-person-calibration + cue-based small-dataset caveat and
  the false-activation-rate emphasis match the Ofner / Schalk / CSP→Riemannian/EEGNet framing this
  study is scoped against (Section 4, 8).

## 14. Publish plan (isolated Git integration)

- Full protocol + manifest: this file (canonical checkout, untracked — canonical dirty checkout
  preserved, **not** committed here).
- Compact evidence (this doc + Stage-1 metrics + manifest.json): copied to the isolated clean
  clone `/home/ubuntu/robomimic-pub-s006r12-20261104` and committed there (branch
  `codex/robot-learning-design`), then push if credentials allow.
- `STATUS.md` updated after this phase.
