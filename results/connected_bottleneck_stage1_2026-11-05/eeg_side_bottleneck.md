# Stage 1 — EEG-side bottleneck table (opened traces only; run 12)
Generated 2026-11-05. Metrics: command duty cycle (fraction of windows issuing `grasp_lift`),
first-activation time, gate-to-gate temporal flips, and p_HAND stats. Same frozen decoder + gates;
only the subject/trace differs. Dev rows are TUNING numbers; S001 is the opened held-out anchor.

| subject | n_win | duty raw | duty conf | duty temporal | first grasp_lift (issued_at_s) | p_HAND mean | flips raw→conf | flips conf→temporal |
|---|---|---|---|---|---|---|---|---|
| S001 (held-out anchor) | 493 | 0.460 | 0.345 | 0.260 | 2.0 | 0.443 | 34 | 44 |
| S006 (dev) | 485 | 0.247 | 0.012 | 0.006 | 4.3 | 0.413 | 51 | 4 |
| S007 (dev) | 493 | 0.379 | 0.217 | 0.128 | 2.0 | 0.411 | 66 | 48 |
| S008 (dev) | 485 | 0.666 | 0.565 | 0.472 | 2.0 | 0.643 | 42 | 48 |
| S009 (dev) | 485 | 0.596 | 0.447 | 0.363 | 2.0 | 0.573 | 45 | 44 |
| S010 (dev) | 485 | 0.254 | 0.101 | 0.041 | 2.0 | 0.311 | 43 | 31 |

## Bottleneck localization (EEG-side, opened data)

- **Command duty cycle is highly subject-dependent under one frozen decoder+gates.** raw-gate duty spans
  0.25 (S006) – 0.67 (S008); confidence-gate duty spans 0.012 (S006) – 0.565 (S008).
- **The confidence gate collapses to near-zero for S006 (0.012) and S010 (0.101):** under a fixed 0.7
  confidence threshold, these subjects issue almost no `grasp_lift` commands → the robot receives ~no
  command → replay cannot complete on the confidence/temporal gates (S006 completed 3/3 raw-only, 0/3
  confidence/temporal). This is the **activation/duty-cycle bottleneck**: a single frozen threshold leaves
  some subjects "silent."
- **S008/S009 over-fire** (high duty, many flips) → the opposite risk: false-activation / instability.
- **First activation is early** (≈2.0 s, i.e. the first window) for 5/6 traces → the decoder marks the
  earliest windows as HAND_IMAGERY, a false-activation tendency on likely-rest onsets (rest label not yet
  available for these traces — to be measured once annotation labels are joined at decode time).

## What this means for the robot outcome (hypothesis to test in Phase B, causal)

The `pause`-matched control (harness `--condition pause`, onset+duty matched to the EEG trace) isolates the
regime/policy from the EEG signal: if the matched control completes but the `eeg` arm does not (or differs),
the bottleneck is the **EEG decoder/timing** (subject duty cycle / false activation / latency); if they agree,
the bottleneck is the **regime/policy**. Per-subject Δ is reported in Phase B.

