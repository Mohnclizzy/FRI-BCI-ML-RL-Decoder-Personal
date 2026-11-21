# Connected EEG→robot bottleneck — Stage 1 ADDENDUM (2026-11-06)

**Date:** 2026-11-06 (UTC) · **Author:** sole experiment writer
**Supersedes / corrects:** parts of `results/connected_bottleneck_stage1_2026-11-05/summary.md` (lines 13, 25) and the "Key result" framing.
**Preserved as evidence (untouched):** `docs/CONNECTED_BOTTLENECK_STAGE1_FREEZE_2026-11-05.md`, `results/connected_bottleneck_stage1_2026-11-05/`, all prior decision traces, the frozen policy checkpoint, and the prior `docs/SESSION_PAUSE_2026-11-05.md`.

This addendum is written **before** any new S002/S003 run-12 outcome is inspected, per the prior handoff. It corrects two things that were previously stated more strongly than the evidence supports, and records the exact unblock that arrived on 2026-11-06.

---

## 1. The `pause`-matched control is inverted in the harness (spec vs. implementation)

The Stage-1 causal design (freeze doc §7, "Matched activation/timing controls (the causal design)") defines the `pause` control as the **matched activation/timing control** and states, verbatim:

> `--condition pause --pause-start S --pause-steps L` → a **deterministic timed trigger**:
> policy **ENABLED for a contiguous block of `L` steps, disabled elsewhere**. This is the
> **matched activation/timing control**: we set the enabled-block onset and length to match [the EEG trace].

The freeze doc's bottleneck-localization rule then depends on this control behaving as specified:

> `Δ_outcome(subject×gate) = outcome(eeg_arm) − outcome(pause_matched_control)` — isolates the [EEG side];
> **Bottleneck localization rule:** `|Δ|` large & `pause`-control passes ⇒ EEG-side bottleneck.

**But the harness implements the opposite.** In the frozen rollout harness
(`results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py`, **lines 85–86**):

```python
elif a.condition=='pause':
    requested='hold' if a.pause_start<=step<a.pause_start+a.pause_steps else 'policy'
    reason='scripted_'+requested
```

This sets the policy to **`hold` (disabled) inside** the block `[pause_start, pause_start+pause_steps)`
and to **`policy` (enabled) outside** it. That is the exact inverse of the freeze spec
(enabled inside the block, disabled elsewhere).

Consequences, stated plainly:

1. **Any `--condition pause` run is a "disable-during-this-window" trigger, not an "enable-during-this-window" trigger.** Running it to "match the EEG onset/duty" would actually *remove* policy exactly when the EEG is active and *add* it elsewhere — the wrong causal control.
2. **No `pause`-control run was ever executed** for S001 (only `always-enable` seed 0, per the Stage-1 summary). So **no published number is invalidated** by this; the defect is latent.
3. The `always-hold` (line 83) and `always-enable` (line 84) branches are implemented exactly as specified (0 % and 100 % duty) and are **not** affected. The `eeg` arm (lines 87–88) correctly follows the decision trace's temporal command pattern.

**Disposition:** the frozen spec is authoritative (it is the causal design). The correct fix is to make the harness match the spec (policy enabled inside the contiguous block, disabled elsewhere) **or** to re-label the control honestly. Either way, **no Stage-1 or later number should be interpreted as a `pause`-matched result until the harness semantics are reconciled with the spec.** Until then, causal reads must rest on the unambiguous controls (`always-hold`, `always-enable`) and the EEG arm itself.

---

## 2. The S001 "robot is not the bottleneck" claim is an overclaim

The Stage-1 summary states (line 13):

> ## Key result — the bottleneck is EEG-side activation, not the robot

and (line 25):

> → Whenever the EEG arm is active, the policy acquires **and retains**; the robot side holds the cube … **For the anchor, the robot is not the bottleneck.**

This is **stronger than the evidence supports**, for two independent reasons.

**(a) It rests on one anchor subject and one control seed.** The positive evidence is:
- S001 EEG arm (raw/confidence/temporal × seeds 0–2): 9/9 `complete_retained` — a genuine positive result (the pipeline works for S001).
- S001 `always-enable` **seed 0 only**: 1/1 `complete_retained`.

That is one held-out subject and one always-on seed. It shows the robot regime *can* complete and retain when commanded for S001. It does **not** establish that the robot is *never* a bottleneck, and it is not a causal localization of the bottleneck to the EEG side.

**(b) The causal localization depends on the `pause`-matched control, which (i) was never run and (ii) is inverted in the harness (§1).** The freeze doc's own localization rule (`|Δ|` large & `pause`-control passes ⇒ EEG-side) therefore **cannot be applied** from the current evidence.

The prior handoff already said this correctly (`docs/SESSION_PAUSE_2026-11-05.md`, line 7):

> The S001-only evidence does not establish that the robot is never a bottleneck; the existing contiguous pause block does not exactly match the EEG temporal command pattern.

**Corrected statement (to replace lines 13/25 wherever reused):**

> **For the anchor subject S001 the connected pipeline works end-to-end (EEG arm 9/9 `complete_retained`; `always-enable` seed 0 1/1).** This shows the robot regime is *capable* of completing and retaining when commanded for S001. It does **not** establish that the robot is never a bottleneck, nor a causal localization of the bottleneck to the EEG side: the `pause`-matched control that the localization rule requires was never executed and is implemented inverted relative to the frozen spec (see §1). The S006R12 result (raw 3/3 retained; confidence and temporal 0/3) is consistent with an EEG-side *activation-dutycycle* effect but is within-subject/exploratory, not a held-out causal read.

The **activation-duty-cycle** observation (confidence-gate collapse on S006/S010, §5 of the Stage-1 summary) remains valid as a descriptive EEG-side effect from existing traces. What is withdrawn is the **causal localization** claim that it therefore *is the bottleneck and the robot is not*.

---

## 3. What changed on 2026-11-06 (the unblock)

The two blockers that held Stage 1 at a partial state are now resolved:

| Blocker (2026-11-05) | Status (2026-11-06) |
|---|---|
| S002/S003 raw EEG not local / no PhysioNet credentials | **S002 and S003 present locally** at `data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/{S002,S003}/` (all of runs 4/8/12). |
| `mne` not installed in the Phase-2 venv | **`mne 1.13.2` installed** in `/home/ubuntu/.venvs/eeg-decoder-stage1/` (with sklearn 1.9.1, scipy 1.18.1, numpy 2.5.4, joblib). |
| Robot-replay path (MuJoCo native lib) | Verified live: `~/.mujoco/mujoco210` + `LD_LIBRARY_PATH`; `robosuite 1.2.0` imports; frozen checkpoint `robomimic-native-phase2-20261104/artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth` present. |

**Therefore the "definitive causal test" that Stage 1 listed as blocked — the S002/S003 held-out read — is now runnable** with the frozen per-person split (runs 4/8 → calibration, run 12 → held-out), exactly as the freeze doc specifies.

---

## 4. What this means for the plan (forward, bounded)

Ordered per the frozen protocol, each a bounded step:

1. **(this addendum) Correct the record** — done here; Stage-1 `summary.md` framing is now superseded for the two claims above.
2. **S001 Phase-B matched controls (unambiguous):** `always-hold` × seeds 0–2 and `always-enable` × seeds 1–2 (seed 0 already exists), latch z-offset +0.015. This gives the duty-cycle bounds and a multi-seed always-on read for the anchor. The `pause`-matched control is **held back** until the §1 inversion is reconciled, or is run **only** after the harness is corrected to the spec and re-verified on a known case.
3. **S002/S003 decode (Phase C, now unblocked):** frozen per-person CSP/LDA, runs 4/8 calibration, run 12 held-out, 9-channel (C3, Cz, C4, C1, C2, FC1, FC2, CP1, CP2), 8–30 Hz, 2 s window / 1 s offset. Produce `S002R12-decisions.jsonl` and `S003R12-decisions.jsonl` (raw/confidence/temporal), balanced accuracy + rest-window false-activation, identical gates/thresholds to S001.
4. **S002/S003 EEG arm + matched controls (Phase B per subject):** EEG arm × gates × seeds 0–2; `always-hold` × seeds 0–2; `always-enable` × seeds 0–2. Then the localization read **subject×gate**, reported as `Δ = outcome(eeg_arm) − outcome(matched_control)` with the control chosen per the corrected §1 semantics.
5. **Integrate + STATUS + publish** (isolated Git integration, verified before push).

**Do-not list (preserved):** do not present S001 as a fresh unopened confirmation; do not inspect S002/S003 run-12 outcomes before the decode step is set up; do not interpret any number as a `pause`-matched result until §1 is reconciled.

---

## 5. Files

- This addendum: `docs/CONNECTED_BOTTLENECK_STAGE1_ADDENDUM_2026-11-06.md`
- Superseded framing: `results/connected_bottleneck_stage1_2026-11-05/summary.md` (lines 13, 25)
- Frozen spec (authoritative): `docs/CONNECTED_BOTTLENECK_STAGE1_FREEZE_2026-11-05.md` §7
- Inverted harness (evidence): `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py` lines 83–88
- Prior handoff caveat (corroborating): `docs/SESSION_PAUSE_2026-11-05.md` line 7
- Unblocked data: `data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/{S002,S003}/`
- Decode env: `/home/ubuntu/.venvs/eeg-decoder-stage1/` (mne 1.13.2)
