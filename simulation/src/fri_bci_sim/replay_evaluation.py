"""Bounded, label-separated evaluation of saved command decision traces.

This module consumes command traces only after decoding has finished. It does
not expose annotations to the decoder or command gate and does not run a robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LabelInterval:
    """Half-open evaluator interval; label is ``rest`` or ``imagery``."""

    start_s: float
    end_s: float
    label: str


@dataclass(frozen=True)
class AnnotatedTrial:
    """One source/episode annotation schedule and its declared time horizon."""

    source: str
    episode: str
    horizon_s: float
    intervals: tuple[LabelInterval, ...]


def evaluate_decision_trace(
    trace_path: Path,
    trials: Iterable[AnnotatedTrial],
    *,
    modes: Iterable[str] | None = None,
) -> dict:
    """Evaluate saved comparison JSONL against independently supplied labels.

    For each trial, only the interval from time zero through the last decision
    timestamp (capped by ``horizon_s``) is observable. The trace is never
    extrapolated beyond its last event. An imagery bout is counted as a miss
    only when its full annotated interval lies inside that observed window;
    partial bouts are reported as censored. A decision state applies from its
    ``issued_at_s`` until the next decision, with HOLD before the first event.
    """
    trial_map = _validate_trials(trials)
    events: dict[tuple[str, str], list[dict]] = {key: [] for key in trial_map}
    discovered_modes: set[str] = set()
    row_modes: set[str] | None = None
    last_by_key: dict[tuple[str, str], tuple[float, float]] = {}

    with Path(trace_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on trace line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Trace line {line_number} must contain a JSON object")
            estimate = row.get("estimate")
            decisions = row.get("decisions")
            if not isinstance(estimate, dict) or not isinstance(decisions, dict) or not decisions:
                raise ValueError(f"Trace line {line_number} needs estimate and decisions")
            key = (estimate.get("source"), estimate.get("episode"))
            if any(not isinstance(part, str) or not part for part in key):
                raise ValueError(f"Trace line {line_number} needs nonempty source and episode strings")
            if key not in trial_map:
                raise ValueError(f"Trace line {line_number} has no matching annotated trial: {key}")
            current_modes = set(decisions)
            if row_modes is None:
                row_modes = current_modes
            elif current_modes != row_modes:
                raise ValueError(f"Trace line {line_number} changes comparison conditions mid-trace")
            window_end = _finite_time(estimate.get("window_end_s"), "window_end_s", line_number)
            available = _finite_time(estimate.get("available_at_s"), "available_at_s", line_number)
            if window_end < 0 or available < window_end:
                raise ValueError(f"Trace line {line_number} violates prediction timestamp causality")
            previous = last_by_key.get(key)
            if previous and (available <= previous[0] or window_end <= previous[1]):
                raise ValueError(f"Trace line {line_number} is duplicate or out of event order for {key}")
            last_by_key[key] = (available, window_end)
            for mode, decision in decisions.items():
                discovered_modes.add(mode)
                if not isinstance(decision, dict):
                    raise ValueError(f"Trace line {line_number} decision {mode!r} is malformed")
                skill = decision.get("skill")
                issued = _finite_time(decision.get("issued_at_s"), "issued_at_s", line_number)
                if skill not in ("hold", "grasp_lift"):
                    raise ValueError(f"Trace line {line_number} has unsupported skill {skill!r}")
                if issued < available:
                    raise ValueError(f"Trace line {line_number} decision precedes estimate availability")
                events[key].append({"mode": mode, "time": issued, "skill": skill, "line": line_number})

    selected = set(modes) if modes is not None else discovered_modes
    if not selected or not selected <= discovered_modes:
        raise ValueError("Requested modes must be present in the trace")
    for key, mode_events in events.items():
        mode_events.sort(key=lambda e: (e["mode"], e["time"], e["line"]))
        for mode in selected:
            sequence = [e for e in mode_events if e["mode"] == mode]
            # Decision timestamps are chronological within each mode/trial.
            if any(a["time"] >= b["time"] for a, b in zip(sequence, sequence[1:])):
                raise ValueError(f"Decision event order is not strictly increasing for {key}/{mode}")

    results = {}
    for mode in sorted(selected):
        aggregate = _empty_metrics()
        per_trial = []
        for key, trial in trial_map.items():
            trial_events = [e for e in events[key] if e["mode"] == mode and e["time"] <= trial.horizon_s]
            last_event = max((e["time"] for e in trial_events), default=0.0)
            observed_end = min(trial.horizon_s, last_event)
            result = _evaluate_trial(trial, trial_events, observed_end)
            per_trial.append(result)
            for field in ("rest_duration_s", "false_activation_duration_s", "imagery_duration_s",
                          "observed_duration_s", "active_duration_s", "imagery_bouts",
                          "detected_bouts", "missed_bouts", "censored_bouts", "state_switches"):
                aggregate[field] += result[field]
            aggregate["detection_delays_s"].extend(result["detection_delays_s"])
            aggregate["truncated_trials"] += int(result["truncated_s"] > 1e-12)
            aggregate["truncated_duration_s"] += result["truncated_s"]
        aggregate["false_activation_fraction_of_rest"] = _ratio(
            aggregate["false_activation_duration_s"], aggregate["rest_duration_s"])
        aggregate["active_coverage_fraction"] = _ratio(aggregate["active_duration_s"], aggregate["observed_duration_s"])
        aggregate["observation_coverage_fraction"] = _ratio(
            aggregate["observed_duration_s"], sum(t.horizon_s for t in trial_map.values()))
        aggregate["detection_coverage_fraction"] = _ratio(
            aggregate["detected_bouts"], aggregate["detected_bouts"] + aggregate["missed_bouts"])
        delays = aggregate.pop("detection_delays_s")
        aggregate["mean_detection_delay_s"] = sum(delays) / len(delays) if delays else None
        aggregate["per_trial"] = per_trial
        results[mode] = aggregate
    return {
        "scope": "offline evaluator of saved decisions; no decoder labels, robot outcomes, or real-time claims",
        "time_model": "decision state held until next decision; HOLD before first; no extrapolation after last event",
        "interval_convention": "half-open [start_s, end_s); imagery bouts partially outside observed trace are censored",
        "trials": len(trial_map),
        "conditions": results,
    }


def _evaluate_trial(trial: AnnotatedTrial, events: list[dict], observed_end: float) -> dict:
    metrics = _empty_metrics()
    metrics.update({"source": trial.source, "episode": trial.episode, "declared_horizon_s": trial.horizon_s,
                    "observed_horizon_s": observed_end, "truncated_s": trial.horizon_s - observed_end})
    events = sorted(events, key=lambda e: e["time"])
    state = "hold"
    cursor = 0.0
    idx = 0
    switch_count = 0
    # Integrate state intervals over the bounded observable time window.
    state_spans = []
    while idx < len(events) and events[idx]["time"] <= observed_end:
        time_s = events[idx]["time"]
        if time_s > cursor:
            state_spans.append((cursor, time_s, state))
        next_state = events[idx]["skill"]
        if next_state != state:
            switch_count += 1
        state = next_state
        cursor = time_s
        idx += 1
    if cursor < observed_end:
        state_spans.append((cursor, observed_end, state))
    metrics["state_switches"] = switch_count
    metrics["observed_duration_s"] = observed_end
    for interval in trial.intervals:
        end = min(interval.end_s, observed_end)
        if end <= interval.start_s:
            continue
        observed_start = interval.start_s
        duration = end - observed_start
        if interval.label == "rest":
            metrics["rest_duration_s"] += duration
        else:
            metrics["imagery_duration_s"] += duration
    for start, end, state in state_spans:
        if state != "grasp_lift":
            continue
        metrics["active_duration_s"] += end - start
        for interval in trial.intervals:
            overlap = max(0.0, min(end, interval.end_s, observed_end) - max(start, interval.start_s))
            if interval.label == "rest":
                metrics["false_activation_duration_s"] += overlap
    for interval in trial.intervals:
        if interval.label != "imagery":
            continue
        metrics["imagery_bouts"] += 1
        full = interval.end_s <= observed_end + 1e-12
        first = min((max(start, interval.start_s) for start, end, state in state_spans
                     if state == "grasp_lift" and end > interval.start_s and start < min(interval.end_s, observed_end)),
                    default=None)
        if first is not None:
            metrics["detected_bouts"] += 1
            metrics["detection_delays_s"].append(max(0.0, first - interval.start_s))
        elif full:
            metrics["missed_bouts"] += 1
        else:
            metrics["censored_bouts"] += 1
    metrics["observation_coverage_fraction"] = _ratio(observed_end, trial.horizon_s)
    metrics["false_activation_fraction_of_rest"] = _ratio(
        metrics["false_activation_duration_s"], metrics["rest_duration_s"])
    metrics["active_coverage_fraction"] = _ratio(metrics["active_duration_s"], observed_end)
    metrics["detection_coverage_fraction"] = _ratio(
        metrics["detected_bouts"], metrics["detected_bouts"] + metrics["missed_bouts"])
    delays = metrics.pop("detection_delays_s")
    metrics["mean_detection_delay_s"] = sum(delays) / len(delays) if delays else None
    metrics["detection_delays_s"] = delays
    return metrics


def _validate_trials(trials: Iterable[AnnotatedTrial]) -> dict[tuple[str, str], AnnotatedTrial]:
    result = {}
    for trial in trials:
        if not isinstance(trial.source, str) or not trial.source or not isinstance(trial.episode, str) or not trial.episode:
            raise ValueError("Annotated trials need source and episode identifiers")
        horizon = _finite_time(trial.horizon_s, "trial horizon")
        if horizon <= 0:
            raise ValueError("Trial horizon must be positive")
        key = (trial.source, trial.episode)
        if key in result:
            raise ValueError(f"Duplicate annotated trial {key}")
        intervals = tuple(trial.intervals)
        if not intervals:
            raise ValueError(f"Trial {key} has no annotation intervals")
        cursor = 0.0
        for interval in intervals:
            start = _finite_time(interval.start_s, "interval start")
            end = _finite_time(interval.end_s, "interval end")
            if interval.label not in ("rest", "imagery") or start < 0 or end <= start or end > horizon:
                raise ValueError(f"Invalid annotation interval in trial {key}: {interval}")
            if start < cursor - 1e-9:
                raise ValueError(f"Annotation intervals must be ordered and nonoverlapping in {key}")
            cursor = end
        result[key] = AnnotatedTrial(trial.source, trial.episode, horizon, intervals)
    if not result:
        raise ValueError("At least one annotated trial is required")
    return result


def _finite_time(value, name: str, line_number: int | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite" + (f" on trace line {line_number}" if line_number else "")) from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite" + (f" on trace line {line_number}" if line_number else ""))
    return number


def _empty_metrics() -> dict:
    return {"rest_duration_s": 0.0, "false_activation_duration_s": 0.0,
            "imagery_duration_s": 0.0, "observed_duration_s": 0.0,
            "active_duration_s": 0.0, "imagery_bouts": 0, "detected_bouts": 0,
            "missed_bouts": 0, "censored_bouts": 0, "state_switches": 0,
            "truncated_trials": 0, "truncated_duration_s": 0.0, "detection_delays_s": []}


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None
