# Candidate recorder and reset guards — 2026-11-02

This candidate-only follow-up is based on Personal branch commit
`45d0bed6083a92f2bd524e3bb0abd99bc0d23d20`. No simulator, GPU playback,
training, or EEG input was launched for this change.

## Changes

- Reject in-process five-episode playback. The supported candidate invocation
  now permits one episode per process until the reset-state loss is resolved.
- Stop on the first recorded physics sample that disagrees with its reset pose,
  and on missing/non-finite contact measurement. Preserve the offending sample
  and record incomplete/error status. Persist the initial row before checking
  table/open-gripper geometry so invalid starts also leave evidence.
- Distinguish raw sampled lift observations from valid episode outcomes.
  Incomplete, invalid-reset, missing-contact, or timestamp-gap measurements
  cannot return a valid physical pass.
- Record individual native position/velocity clamp events with before/after
  values, joint and cube velocities, actual applied joint position targets,
  episode correction counts, and imported candidate source paths/hashes.
- Flag peak finger force >=100 N and native state corrections for review.
  The 100 N value is a diagnostic marker for the already observed concern,
  not a safety limit. It does not clip forces or change the lift predicate.

Native task actions, gains, gate, reset distribution, contact physics, horizon,
weights, existing project code and pinned Isaac Lab were not changed. Prior
raw outputs and published result files were preserved. The new machine-readable
record is `results/nvidia_candidate_2026-11-02/guard_review.json`.

## Validation and limits

Sixteen focused CPU tests pass, including immediate recorder failure on stale
reset state/missing contact, preservation of error evidence, high-force review,
and rejection of incomplete/non-finite measurements as valid successes.
Offline rescoring of the five saved fresh-process traces (seeds 9–13) matches
all original outcomes: one valid sampled lift out of five. No new physical
performance evidence was collected.

The underlying Newton reset-state loss is still unresolved. Restricting each
process to one episode prevents reuse of that broken reset path; it does not
repair the backend. The causes of large forces and frequent native clamps also
remain unresolved. Do not qualify the controller, alter its gains, or expand
training based on this recorder change. A source/trace fidelity investigation
and separately reviewed physical validation are still required.
