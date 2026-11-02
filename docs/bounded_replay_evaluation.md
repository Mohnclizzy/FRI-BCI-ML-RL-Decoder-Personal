# Bounded evaluation of EEG command traces

`fri_bci_sim.replay_evaluation` scores the JSONL output written by
`compare_trace` against evaluator-owned, timestamped instruction intervals. It
is an offline post-processing step: the decoder and command gate receive only
EEG estimates and their timestamps. The evaluator's annotation objects are
passed separately after the saved decision trace has been produced.

> **Interpretation limit:** The evaluator treats each decision as a state that
> persists across gaps until another saved decision arrives. It assumes HOLD
> before the first decision, including during decoder warm-up. It does not
> simulate the executor watchdog, command expiry, or robot behavior. These
> assumptions are bookkeeping rules for offline scoring; the resulting
> durations and switches must not be interpreted as live physical behavior.

## Interface

Construct `AnnotatedTrial(source, episode, horizon_s, intervals)` with an
ordered tuple of `LabelInterval(start_s, end_s, label)` values. Intervals are
half-open and may be separated by unlabelled gaps, but may not overlap. Labels
are `rest` and `imagery`; source and episode must match the saved trace. The
declared horizon is an explicit upper bound in recording seconds. Then call
`evaluate_decision_trace(path, trials)`. The result has per-condition aggregate
metrics and a per-trial breakdown.

The evaluator rejects traces with non-finite times, estimates available before
their window ends, decisions issued before estimate availability, duplicate or
out-of-order estimate timestamps, unsupported command states, or decisions
that do not match an annotated source/episode. It preserves source and episode
boundaries and never reads label fields from trace rows.

## Time and metric definitions

- A decision's state applies from its `issued_at_s` through the next decision,
  including across missing-event gaps. The state starts at HOLD, including
  decoder warm-up before the first saved decision. Events after the declared
  horizon are ignored. No executor watchdog or command expiry is modeled.
- Observability ends at the last saved decision timestamp, capped by the
  declared horizon. The final state is not extrapolated beyond that event.
  Trial truncation is reported as `declared_horizon_s - observed_horizon_s`;
  a recording with no decision has zero observable duration.
- Rest false activation duration is the overlap between a `grasp_lift` state
  and observed annotated rest intervals. Its fraction divides this duration by
  observed annotated rest duration.
- An imagery bout is detected if an active state overlaps the bout. Delay is
  measured from bout onset to the first active overlap; if active at onset,
  delay is zero. A fully observed bout without active overlap is a miss. A bout
  extending beyond the observed horizon without detection is censored, not a
  miss. Detection coverage divides detected bouts by detected plus missed
  bouts, so censored bouts are excluded.
- Active coverage is active-state time divided by observable recording time.
  Observation coverage is observable declared-horizon time divided by total
  declared horizon. State switches count transitions between HOLD and
  GRASP_LIFT within each trial; initial HOLD is not a switch.

Intervals and trace times use the same recording-time clock. Causal estimates
may be issued later than their window end; delay begins at the annotated
instruction onset and therefore includes acquisition/window and decision delay
only insofar as those delays are represented in trace timestamps. Metrics
depend on the declared labels, horizon, and the state-hold convention above.

This evaluator does not measure robot motion, task success, physical safety,
human adaptation, live acquisition, wall-clock latency, or real-time operation.
It is not a substitute for the planned simulator-state recorder or matched
physical-task evaluation. Do not describe its outputs as physical or real-time
results.
