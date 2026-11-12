"""Versioned native BC-RNN completion-latch harness and 20s horizon exposure joiner.

Separates the two historical meanings of 'pause' into explicit, unambiguous
modes:
  - ``enable-window``: policy enabled inside ``[window_start, window_start + window_steps)``, HOLD outside.
  - ``hold-window``:   HOLD inside ``[window_start, window_start + window_steps)``, policy enabled outside.
  - ``enable-schedule``: policy enabled on an explicit step schedule (e.g. exact
    fragmented or bout-shuffled matched control).

Also provides deterministic matched-control schedule builders and a 20-second
horizon exposure/telemetry/annotation joiner that aligns EEG decision traces,
EDF REST/imagery intervals, and robot telemetry on the same ``[0, 20.0)`` s
robot control clock, distinguishing requested enable time, actual policy-call
time, and post-acquisition completion-latch override time.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

HARNESS_VERSION = "robomimic-latch6-v2.0.0"
VALID_CONDITIONS = (
    "always-hold",
    "always-enable",
    "enable-window",
    "hold-window",
    "enable-schedule",
    "eeg",
)
VALID_GATES = ("raw", "confidence", "temporal", "crc_hysteresis")
DEFAULT_CKPT = Path("/home/ubuntu/robomimic-native-phase2-20261104/artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth")
TOLERANCES = dict(
    rise_m=0.08,
    contact_steps=10,
    goal_pos_m=1e-9,
    goal_ori_matrix=1e-9,
    gripper_command=1e-9,
    eef_drift_m=0.035,
    cube_drift_m=0.025,
)


def harness_source_sha256() -> str:
    """Return the SHA-256 digest of this harness module."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


class CausalDecisionReplay:
    """Causal decision trace reader supporting raw, confidence, temporal, and crc_hysteresis."""

    def __init__(self, path: Path, gate: str, max_age_s: float = 1.0):
        if gate not in VALID_GATES:
            raise ValueError(f"Unsupported gate {gate!r}; expected one of {VALID_GATES}")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.path = Path(path)
        self.gate = gate
        self.max_age_s = float(max_age_s)
        self.events: list[tuple[float, float, str, str]] = []
        self.times: list[float] = []
        self.source_key: tuple[str, str] | None = None
        previous = None
        with self.path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                est = row["estimate"]
                if gate not in row["decisions"]:
                    raise ValueError(f"Gate {gate!r} not present in decision trace line {number}")
                decision = row["decisions"][gate]
                source = (str(est["source"]), str(est["episode"]))
                end = float(est["window_end_s"])
                available = float(est["available_at_s"])
                issued = float(decision["issued_at_s"])
                if not all(map(math.isfinite, (end, available, issued))) or end < 0 or available < end or issued < available:
                    raise ValueError(f"Causality violation at decision line {number}")
                if self.source_key is not None and source != self.source_key:
                    raise ValueError("Mixed EEG source/episode in trace")
                if previous and (end <= previous[0] or available < previous[1] or issued < previous[2]):
                    raise ValueError(f"Out-of-order decision at line {number}")
                skill = str(decision["skill"])
                if skill not in ("hold", "grasp_lift"):
                    raise ValueError(f"Unknown decision skill {skill!r} at line {number}")
                reason = str(decision.get("reason", "recorded_replay"))
                self.events.append((issued, end, skill, reason))
                self.times.append(issued)
                previous = (end, available, issued)
                self.source_key = source
        if not self.events:
            raise ValueError("Empty EEG decision trace")

    def at(self, time_s: float) -> tuple[str, str]:
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Query time_s must be finite and nonnegative")
        index = bisect.bisect_right(self.times, time_s) - 1
        if index < 0:
            return "hold", "awaiting_estimate"
        issued, end, skill, reason = self.events[index]
        if time_s - end > self.max_age_s:
            return "hold", "stale"
        return skill, reason


def validate_window_args(window_start: int | None, window_steps: int | None, horizon: int) -> tuple[int, int]:
    """Validate 1-based half-open step window [window_start, window_start + window_steps) within 1..horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if window_start is None or window_steps is None:
        raise ValueError("window_start and window_steps are required for windowed control modes")
    start = int(window_start)
    steps = int(window_steps)
    if start < 1 or steps < 0 or start > horizon + 1 or start + steps - 1 > horizon:
        raise ValueError(
            f"Invalid window [{start}, {start + steps}) for 1-based horizon 1..{horizon}"
        )
    return start, steps


def spans_to_active_steps(spans: Sequence[Sequence[int]], horizon: int) -> set[int]:
    """Convert ordered half-open 1-based step spans [[start, end), ...] into a validated set of active steps."""
    active: set[int] = set()
    prev_end = 1
    for span in spans:
        if len(span) != 2:
            raise ValueError(f"Each span must be [start_step, end_step), got {span!r}")
        start, end = int(span[0]), int(span[1])
        if start < 1 or end < start or end > horizon + 1:
            raise ValueError(f"Span [{start}, {end}) out of bounds for 1..{horizon}")
        if start < prev_end:
            raise ValueError(f"Overlapping or unordered spans: [{start}, {end}) after {prev_end}")
        prev_end = end
        active.update(range(start, end))
    return active


def active_steps_to_spans(active_steps: Iterable[int], horizon: int) -> list[list[int]]:
    """Collapse a set/sequence of 1-based active steps into disjoint half-open spans [[start, end), ...]."""
    sorted_steps = sorted({int(s) for s in active_steps})
    for s in sorted_steps:
        if s < 1 or s > horizon:
            raise ValueError(f"Active step {s} out of bounds for 1..{horizon}")
    spans: list[list[int]] = []
    for s in sorted_steps:
        if spans and spans[-1][1] == s:
            spans[-1][1] = s + 1
        else:
            spans.append([s, s + 1])
    return spans


def resolve_step_request(
    step: int,
    condition: str,
    *,
    horizon: int = 400,
    control_freq: float = 20.0,
    window_start: int | None = None,
    window_steps: int | None = None,
    active_steps: set[int] | None = None,
    replay: CausalDecisionReplay | None = None,
) -> tuple[str, str]:
    """Resolve the requested robot mode ('hold' or 'policy') and reason at 1-based step.

    Explicitly rejects the ambiguous historical 'pause' condition name.
    """
    if condition == "pause":
        raise ValueError(
            "Ambiguous condition 'pause' is not permitted in robomimic_harness v2. "
            "Use 'enable-window' (policy enabled inside [window_start, window_start+window_steps), hold outside), "
            "'hold-window' (hold inside [window_start, window_start+window_steps), policy outside), "
            "or 'enable-schedule' (explicit step schedule)."
        )
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}; expected one of {VALID_CONDITIONS}")
    if step < 1 or step > horizon:
        raise ValueError(f"Step {step} out of range 1..{horizon}")

    if condition == "always-hold":
        return "hold", "always_hold_control"
    if condition == "always-enable":
        return "policy", "always_enable_control"
    if condition == "enable-window":
        start, length = validate_window_args(window_start, window_steps, horizon)
        inside = start <= step < start + length
        return ("policy", "enable_window_inside") if inside else ("hold", "enable_window_outside")
    if condition == "hold-window":
        start, length = validate_window_args(window_start, window_steps, horizon)
        inside = start <= step < start + length
        return ("hold", "hold_window_inside") if inside else ("policy", "hold_window_outside")
    if condition == "enable-schedule":
        if active_steps is None:
            raise ValueError("active_steps set is required for condition 'enable-schedule'")
        return ("policy", "scheduled_enable") if step in active_steps else ("hold", "scheduled_hold")
    if replay is None:
        raise ValueError("CausalDecisionReplay instance is required for condition 'eeg'")
    query_time_s = (step - 1) / float(control_freq)
    skill, reason = replay.at(query_time_s)
    return ("hold" if skill == "hold" else "policy"), reason


def build_step_schedule(
    condition: str,
    *,
    horizon: int = 400,
    control_freq: float = 20.0,
    window_start: int | None = None,
    window_steps: int | None = None,
    active_steps: set[int] | None = None,
    replay: CausalDecisionReplay | None = None,
) -> list[dict]:
    """Build the full 1..horizon step schedule before latch override."""
    schedule = []
    for step in range(1, horizon + 1):
        query_time_s = (step - 1) / float(control_freq)
        end_time_s = step / float(control_freq)
        req_mode, reason = resolve_step_request(
            step,
            condition,
            horizon=horizon,
            control_freq=control_freq,
            window_start=window_start,
            window_steps=window_steps,
            active_steps=active_steps,
            replay=replay,
        )
        schedule.append(
            {
                "step": step,
                "query_time_s": query_time_s,
                "step_end_time_s": end_time_s,
                "requested_mode": req_mode,
                "requested_skill": "grasp_lift" if req_mode == "policy" else "hold",
                "reason": reason,
            }
        )
    return schedule


def derive_matched_control_schedules(
    decisions_path: Path,
    gate: str,
    *,
    horizon: int = 400,
    control_freq: float = 20.0,
    max_age_s: float = 1.0,
    shuffle_seeds: Sequence[int] = (0, 1, 2),
) -> dict:
    """Derive duration-matched contiguous and bout-shuffled schedules from an EEG trace on [0, horizon/control_freq).

    Preserves exact total enabled step count K and, for bout-shuffled controls,
    the exact multiset of active bout lengths and hold gap lengths within the
    observable robot episode horizon.
    """
    replay = CausalDecisionReplay(decisions_path, gate, max_age_s=max_age_s)
    eeg_schedule = build_step_schedule("eeg", horizon=horizon, control_freq=control_freq, replay=replay)
    active_steps = [row["step"] for row in eeg_schedule if row["requested_mode"] == "policy"]
    spans = active_steps_to_spans(active_steps, horizon)
    total_enabled = len(active_steps)
    bout_lengths = [end - start for start, end in spans]

    # Pre-estimate awaiting prefix: before first EEG estimate is issued, the robot is structurally in awaiting_estimate.
    first_issued_s = replay.times[0]
    first_possible_step = max(1, min(horizon + 1, int(math.ceil(first_issued_s * control_freq)) + 1))

    if total_enabled == 0:
        contiguous = {
            "condition": "enable-window",
            "window_start": 1,
            "window_steps": 0,
            "spans": [],
            "requested_enable_steps": 0,
            "requested_enable_duration_s": 0.0,
        }
    else:
        first_enable_step = active_steps[0]
        start_step = min(first_enable_step, horizon - total_enabled + 1)
        contiguous = {
            "condition": "enable-window",
            "window_start": start_step,
            "window_steps": total_enabled,
            "spans": [[start_step, start_step + total_enabled]],
            "requested_enable_steps": total_enabled,
            "requested_enable_duration_s": total_enabled / float(control_freq),
        }

    # Decompose post-awaiting region [anchor_step, horizon + 1) into B active bouts and B + 1 hold gaps.
    anchor_step = min([ first_possible_step, *( [active_steps[0]] if active_steps else [first_possible_step] ) ])
    post_anchor_len = (horizon + 1) - anchor_step
    total_hold_in_region = post_anchor_len - total_enabled

    gaps: list[int] = []
    cursor = anchor_step
    for start, end in spans:
        gaps.append(start - cursor)
        cursor = end
    gaps.append((horizon + 1) - cursor)
    assert sum(gaps) == total_hold_in_region
    assert sum(bout_lengths) == total_enabled

    shuffled_controls = []
    for seed in shuffle_seeds:
        if total_enabled == 0:
            shuffled_controls.append(
                {
                    "shuffle_seed": int(seed),
                    "condition": "enable-schedule",
                    "spans": [],
                    "bout_lengths_steps": [],
                    "requested_enable_steps": 0,
                    "requested_enable_duration_s": 0.0,
                }
            )
            continue
        rng = np.random.default_rng(int(seed) + 10007)
        perm_bouts = [int(x) for x in rng.permutation(bout_lengths)]
        perm_gaps = [int(x) for x in rng.permutation(gaps)]
        # Ensure consecutive bouts remain separated by >=1 hold step so bout count & lengths are conserved.
        if len(perm_bouts) > 1:
            interior_indices = list(range(1, len(perm_gaps) - 1))
            for idx in interior_indices:
                if perm_gaps[idx] == 0:
                    donor = next((j for j in range(len(perm_gaps)) if j not in interior_indices and perm_gaps[j] > 1), None)
                    if donor is None:
                        donor = next((j for j in range(len(perm_gaps)) if perm_gaps[j] > 1), None)
                    if donor is not None:
                        perm_gaps[donor] -= 1
                        perm_gaps[idx] += 1
        pos = anchor_step
        shuf_spans: list[list[int]] = []
        for b_len, g_len in zip(perm_bouts, perm_gaps[:-1]):
            pos += g_len
            shuf_spans.append([pos, pos + b_len])
            pos += b_len
        pos += perm_gaps[-1]
        assert pos == horizon + 1
        shuf_steps = spans_to_active_steps(shuf_spans, horizon)
        assert len(shuf_steps) == total_enabled
        shuffled_controls.append(
            {
                "shuffle_seed": int(seed),
                "condition": "enable-schedule",
                "spans": shuf_spans,
                "bout_lengths_steps": [end - start for start, end in shuf_spans],
                "requested_enable_steps": len(shuf_steps),
                "requested_enable_duration_s": len(shuf_steps) / float(control_freq),
            }
        )

    return {
        "source": replay.source_key[0] if replay.source_key else None,
        "episode": replay.source_key[1] if replay.source_key else None,
        "decisions_path": str(Path(decisions_path)),
        "decisions_sha256": hashlib.sha256(Path(decisions_path).read_bytes()).hexdigest(),
        "gate": gate,
        "horizon_steps": horizon,
        "control_freq_hz": float(control_freq),
        "horizon_duration_s": horizon / float(control_freq),
        "eeg_schedule": {
            "spans": spans,
            "bout_count": len(spans),
            "bout_lengths_steps": bout_lengths,
            "first_enable_step": active_steps[0] if active_steps else None,
            "requested_enable_steps": total_enabled,
            "requested_enable_duration_s": total_enabled / float(control_freq),
        },
        "contiguous_duration_matched": contiguous,
        "bout_shuffled_matched": shuffled_controls,
    }


def join_horizon_exposure_and_telemetry(
    *,
    condition: str,
    horizon: int = 400,
    control_freq: float = 20.0,
    decisions_path: Path | None = None,
    gate: str | None = None,
    window_start: int | None = None,
    window_steps: int | None = None,
    schedule_spans: Sequence[Sequence[int]] | None = None,
    telemetry_path: Path | None = None,
    summary_path: Path | None = None,
    intervals: Sequence[Sequence[float | int | str]] | None = None,
    hand_speed_threshold_m_s: float = 0.005,
) -> dict:
    """Align command requests, EDF annotations, and robot telemetry on the declared [0, horizon/control_freq) interval.

    Distinguishes:
      1. Full-recording EEG decision events vs in-horizon (<20s) decision events,
      2. Requested enable steps/duration on the 20 Hz robot clock (before latch override),
      3. Executed policy calls/duration (before latch acquisition),
      4. Executed HOLD steps/duration (before latch acquisition), and
      5. Post-acquisition latch steps/duration (and how many post-latch HOLD requests were overridden).
    """
    horizon_s = horizon / float(control_freq)
    replay = CausalDecisionReplay(decisions_path, gate) if condition == "eeg" else None
    active_set = spans_to_active_steps(schedule_spans, horizon) if condition == "enable-schedule" and schedule_spans is not None else None

    step_schedule = build_step_schedule(
        condition,
        horizon=horizon,
        control_freq=control_freq,
        window_start=window_start,
        window_steps=window_steps,
        active_steps=active_set,
        replay=replay,
    )

    req_enable_steps = sum(1 for r in step_schedule if r["requested_mode"] == "policy")
    req_hold_steps = horizon - req_enable_steps
    req_spans = active_steps_to_spans([r["step"] for r in step_schedule if r["requested_mode"] == "policy"], horizon)

    trace_info = None
    if replay is not None and decisions_path is not None:
        full_enable = sum(1 for _, _, sk, _ in replay.events if sk == "grasp_lift")
        before_horizon = sum(1 for iss, _, sk, _ in replay.events if iss < horizon_s - 1e-12 and sk == "grasp_lift")
        at_or_before_horizon = sum(1 for iss, _, sk, _ in replay.events if iss <= horizon_s + 1e-12 and sk == "grasp_lift")
        trace_info = {
            "decisions_sha256": hashlib.sha256(Path(decisions_path).read_bytes()).hexdigest(),
            "gate": gate,
            "trace_rows": len(replay.events),
            "last_issued_s": replay.events[-1][0],
            "full_trace_enable_events": full_enable,
            "events_issued_strictly_before_horizon": before_horizon,
            "events_issued_at_or_before_horizon": at_or_before_horizon,
        }

    # Normalize annotation intervals [[start_s, end_s, label], ...] clipped to [0, horizon_s]
    norm_intervals: list[tuple[float, float, str]] = []
    if intervals is not None:
        for item in intervals:
            s_t, e_t, lbl = float(item[0]), float(item[1]), item[2]
            if lbl in (0, "0", "rest", "REST", "T0"):
                label_str = "rest"
            elif lbl in (1, "1", "imagery", "HAND_IMAGERY", "T1", "T2"):
                label_str = "imagery"
            else:
                raise ValueError(f"Unsupported interval label {lbl!r}")
            c_start = max(0.0, min(horizon_s, s_t))
            c_end = max(0.0, min(horizon_s, e_t))
            if c_end > c_start:
                if norm_intervals and norm_intervals[-1][2] == label_str and abs(norm_intervals[-1][1] - c_start) <= 1e-9:
                    norm_intervals[-1] = (norm_intervals[-1][0], c_end, label_str)
                else:
                    norm_intervals.append((c_start, c_end, label_str))

    annotation_metrics = None
    if norm_intervals:
        dt = 1.0 / float(control_freq)
        rest_s = sum(e - s for s, e, l in norm_intervals if l == "rest")
        imagery_s = sum(e - s for s, e, l in norm_intervals if l == "imagery")
        req_rest_enable_s = 0.0
        req_imagery_enable_s = 0.0
        for row in step_schedule:
            if row["requested_mode"] != "policy":
                continue
            a, b = row["query_time_s"], row["step_end_time_s"]
            for s, e, l in norm_intervals:
                overlap = max(0.0, min(b, e) - max(a, s))
                if l == "rest":
                    req_rest_enable_s += overlap
                else:
                    req_imagery_enable_s += overlap
        imagery_bouts = [(s, e) for s, e, l in norm_intervals if l == "imagery"]
        detected = missed = 0
        delays: list[float] = []
        for b_start, b_end in imagery_bouts:
            first_hit = next(
                (
                    max(b_start, row["query_time_s"])
                    for row in step_schedule
                    if row["requested_mode"] == "policy"
                    and row["step_end_time_s"] > b_start + 1e-12
                    and row["query_time_s"] < b_end - 1e-12
                ),
                None,
            )
            if first_hit is not None:
                detected += 1
                delays.append(max(0.0, first_hit - b_start))
            else:
                missed += 1
        annotation_metrics = {
            "horizon_rest_duration_s": rest_s,
            "horizon_imagery_duration_s": imagery_s,
            "requested_rest_enable_duration_s": req_rest_enable_s,
            "requested_rest_enable_fraction": req_rest_enable_s / rest_s if rest_s > 0 else None,
            "requested_imagery_enable_duration_s": req_imagery_enable_s,
            "requested_imagery_missed_duration_s": max(0.0, imagery_s - req_imagery_enable_s),
            "requested_imagery_missed_fraction": max(0.0, imagery_s - req_imagery_enable_s) / imagery_s if imagery_s > 0 else None,
            "imagery_bouts_in_horizon": len(imagery_bouts),
            "detected_imagery_bouts_in_horizon": detected,
            "missed_imagery_bouts_in_horizon": missed,
            "mean_requested_onset_delay_s": float(np.mean(delays)) if delays else None,
            "requested_onset_delays_s": delays,
        }

    telemetry_metrics = None
    if telemetry_path is not None and Path(telemetry_path).exists():
        rows = [
            json.loads(line)
            for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary_data = json.loads(Path(summary_path).read_text(encoding="utf-8")) if summary_path and Path(summary_path).exists() else {}
        step_rows = [r for r in rows if int(r["step"]) >= 1 and int(r["step"]) <= horizon]
        if len(step_rows) != horizon:
            raise ValueError(f"Expected {horizon} step rows in {telemetry_path}, found {len(step_rows)}")

        exec_policy_steps = 0
        exec_hold_steps = 0
        exec_latch_steps = 0
        post_latch_req_enable = 0
        post_latch_req_hold = 0

        pre_latch_rest_policy_s = 0.0
        post_latch_rest_s = 0.0
        rest_eef_path_m = 0.0
        pre_latch_rest_eef_path_m = 0.0
        post_latch_rest_eef_path_m = 0.0
        rest_moving_s = 0.0
        pre_latch_rest_moving_s = 0.0

        by_step = {int(r["step"]): r for r in rows}
        for sched in step_schedule:
            step = sched["step"]
            r = by_step[step]
            prev_r = by_step[step - 1]
            mode = str(r["mode"])
            req_mode = sched["requested_mode"]
            if mode != "latch" and mode != req_mode:
                raise ValueError(
                    f"Step {step} telemetry mode {mode!r} does not match expected unlatched request {req_mode!r}"
                )
            if mode == "policy":
                exec_policy_steps += 1
            elif mode == "hold":
                exec_hold_steps += 1
            elif mode == "latch":
                exec_latch_steps += 1
                if req_mode == "policy":
                    post_latch_req_enable += 1
                else:
                    post_latch_req_hold += 1
            else:
                raise ValueError(f"Unexpected telemetry mode {mode!r} at step {step}")

            if norm_intervals:
                a, b = sched["query_time_s"], sched["step_end_time_s"]
                dt = b - a
                disp = float(np.linalg.norm(np.asarray(r["eef_pos"], dtype=float) - np.asarray(prev_r["eef_pos"], dtype=float)))
                speed = disp / dt if dt > 0 else 0.0
                moving = speed > hand_speed_threshold_m_s
                for s, e, l in norm_intervals:
                    if l != "rest":
                        continue
                    overlap = max(0.0, min(b, e) - max(a, s))
                    if overlap <= 0:
                        continue
                    frac = overlap / dt if dt > 0 else 0.0
                    rest_eef_path_m += disp * frac
                    if moving:
                        rest_moving_s += overlap
                    if mode == "latch":
                        post_latch_rest_s += overlap
                        post_latch_rest_eef_path_m += disp * frac
                    else:
                        if mode == "policy":
                            pre_latch_rest_policy_s += overlap
                        pre_latch_rest_eef_path_m += disp * frac
                        if moving:
                            pre_latch_rest_moving_s += overlap

        dt_step = 1.0 / float(control_freq)
        telemetry_metrics = {
            "telemetry_sha256": hashlib.sha256(Path(telemetry_path).read_bytes()).hexdigest(),
            "seed": summary_data.get("seed"),
            "first_acquisition_step": summary_data.get("first_acquisition_step"),
            "first_acquisition_time_s": (
                summary_data["first_acquisition_step"] * dt_step
                if summary_data.get("first_acquisition_step") is not None
                else None
            ),
            "latch_step": summary_data.get("latch_step"),
            "terminal_retained": bool(summary_data.get("terminal_retained", False)),
            "completion_state": summary_data.get("completion_state"),
            "terminal_rise_m": summary_data.get("terminal_rise_m"),
            "max_rise_m": summary_data.get("max_rise_m"),
            "executed_policy_steps": exec_policy_steps,
            "executed_policy_duration_s": exec_policy_steps * dt_step,
            "executed_hold_steps": exec_hold_steps,
            "executed_hold_duration_s": exec_hold_steps * dt_step,
            "post_acquisition_latch_steps": exec_latch_steps,
            "post_acquisition_latch_duration_s": exec_latch_steps * dt_step,
            "post_latch_overridden_enable_steps": post_latch_req_enable,
            "post_latch_overridden_hold_steps": post_latch_req_hold,
            "hold_invariants_pass": summary_data.get("hold_invariants_pass"),
            "latch_invariants_pass": summary_data.get("latch_invariants_pass"),
            "rest_motion_on_horizon": None
            if not norm_intervals
            else {
                "pre_latch_executed_policy_during_rest_s": pre_latch_rest_policy_s,
                "post_latch_override_during_rest_s": post_latch_rest_s,
                "total_rest_eef_path_length_m": rest_eef_path_m,
                "pre_latch_rest_eef_path_length_m": pre_latch_rest_eef_path_m,
                "post_latch_rest_eef_path_length_m": post_latch_rest_eef_path_m,
                "total_rest_moving_duration_s": rest_moving_s,
                "pre_latch_rest_moving_duration_s": pre_latch_rest_moving_s,
            },
        }

    return {
        "harness_version": HARNESS_VERSION,
        "condition": condition,
        "horizon_steps": horizon,
        "control_freq_hz": float(control_freq),
        "horizon_duration_s": horizon_s,
        "requested_on_horizon": {
            "requested_enable_steps": req_enable_steps,
            "requested_enable_duration_s": req_enable_steps / float(control_freq),
            "requested_hold_steps": req_hold_steps,
            "requested_hold_duration_s": req_hold_steps / float(control_freq),
            "requested_enable_duty_fraction": req_enable_steps / float(horizon),
            "requested_bout_count": len(req_spans),
            "requested_spans": req_spans,
        },
        "trace_accounting": trace_info,
        "annotation_alignment_20s": annotation_metrics,
        "robot_telemetry_alignment_20s": telemetry_metrics,
    }


def run_rollout(argv: Sequence[str] | None = None) -> dict:
    """Execute a native robomimic rollout with explicit v2 control semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", required=True, help=f"One of {VALID_CONDITIONS} ('pause' is rejected)")
    parser.add_argument("--window-start", type=int, help="1-based start step for enable-window or hold-window")
    parser.add_argument("--window-steps", type=int, help="Step count for enable-window or hold-window")
    parser.add_argument("--schedule-json", type=Path, help="JSON file with 'spans' [[start, end), ...] for enable-schedule")
    parser.add_argument("--decisions", type=Path, help="EEG decisions JSONL for condition=eeg")
    parser.add_argument("--gate", choices=VALID_GATES, default="raw")
    parser.add_argument("--resume-state", choices=("freeze", "reset"), default="freeze")
    parser.add_argument("--latch-z-offset", type=float, default=0.015)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)

    if args.condition == "pause":
        parser.error(
            "Ambiguous '--condition pause' is retired in v2. "
            "Use '--condition enable-window' or '--condition hold-window'."
        )
    if args.condition not in VALID_CONDITIONS:
        parser.error(f"Unsupported condition {args.condition!r}; choose from {VALID_CONDITIONS}")

    replay = CausalDecisionReplay(args.decisions, args.gate) if args.condition == "eeg" else None
    schedule_spans = None
    if args.condition == "enable-schedule":
        if args.schedule_json is None:
            parser.error("--schedule-json is required for --condition enable-schedule")
        sched_data = json.loads(args.schedule_json.read_text(encoding="utf-8"))
        schedule_spans = sched_data["spans"] if isinstance(sched_data, dict) else sched_data

    # Validate schedule on 400-step default before importing heavy simulator modules
    active_set = spans_to_active_steps(schedule_spans, 400) if schedule_spans is not None else None
    build_step_schedule(
        args.condition,
        horizon=400,
        control_freq=20.0,
        window_start=args.window_start,
        window_steps=args.window_steps,
        active_steps=active_set,
        replay=replay,
    )

    import imageio
    import torch
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    args.out.mkdir(parents=True, exist_ok=False)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.ckpt),
        device=TorchUtils.get_torch_device(try_to_use_cuda=False),
        verbose=False,
    )
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    horizon = int(config.experiment.rollout.horizon)
    active_set = spans_to_active_steps(schedule_spans, horizon) if schedule_spans is not None else None
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=args.video, verbose=False
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy.start_episode()
    obs = env.reset()
    obs = env.reset_to(env.get_state())
    base = env.env
    sim = base.sim
    robot = base.robots[0]
    ctrl = robot.controller
    original = ctrl.set_goal
    left = set(robot.gripper.important_geoms["left_fingerpad"])
    right = set(robot.gripper.important_geoms["right_fingerpad"])
    obj = set(base.cube.contact_geoms)

    rows = []
    target_pos = target_ori = target_grip = target_cube = None
    calls = entries = resumes = 0
    previous = None
    contact_run = 0
    latched = False
    latch_step = None
    latch_goal_pos = latch_goal_ori = latch_cube = None
    writer = imageio.get_writer(str(args.out / "video.mp4"), fps=4) if args.video else None

    def held_set_goal(action):
        original(action, set_pos=target_pos, set_ori=target_ori)

    def measure(step, requested_mode, mode, reason, action, reward, done):
        lh = rh = False
        pairs = []
        for i in range(sim.data.ncon):
            con = sim.data.contact[i]
            n1 = sim.model.geom_id2name(int(con.geom1))
            n2 = sim.model.geom_id2name(int(con.geom2))
            if (n1 in obj and n2 in left) or (n2 in obj and n1 in left):
                lh = True
            if (n1 in obj and n2 in right) or (n2 in obj and n1 in right):
                rh = True
            if n1 in obj or n2 in obj:
                pairs.append([n1, n2])
        rows.append(
            dict(
                step=step,
                interval_start_s=0.0 if step == 0 else (step - 1) / float(base.control_freq),
                time_s=step / float(base.control_freq),
                requested_mode=requested_mode,
                requested_skill="grasp_lift" if requested_mode == "policy" else "hold",
                mode=mode,
                reason=reason,
                action=None if action is None else np.asarray(action).tolist(),
                cube_pos=sim.data.body_xpos[base.cube_body_id].tolist(),
                eef_pos=np.asarray(obs["robot0_eef_pos"]).tolist(),
                goal_pos=np.asarray(ctrl.goal_pos).tolist(),
                goal_ori=np.asarray(ctrl.goal_ori).tolist(),
                gripper_command=np.asarray(robot.gripper.current_action).tolist(),
                target_pos=None if target_pos is None else target_pos.tolist(),
                target_ori=None if target_ori is None else target_ori.tolist(),
                target_grip=None if target_grip is None else target_grip.tolist(),
                target_cube=None if target_cube is None else target_cube.tolist(),
                bilateral_contact=lh and rh,
                builtin_grasp=bool(base._check_grasp(gripper=robot.gripper, object_geoms=base.cube)),
                contact_pairs=pairs,
                native_success=bool(env.is_success()["task"]),
                reward=float(reward),
                done=bool(done),
                policy_calls=calls,
                rnn_counter=int(policy.policy._rnn_counter),
                latched=latched,
                latch_step=latch_step,
                contact_run=contact_run,
                latch_goal_pos=None if latch_goal_pos is None else latch_goal_pos.tolist(),
                latch_goal_ori=None if latch_goal_ori is None else latch_goal_ori.tolist(),
            )
        )

    measure(0, "hold", "initial", "reset", None, 0.0, False)
    try:
        for step in range(1, horizon + 1):
            requested, reason = resolve_step_request(
                step,
                args.condition,
                horizon=horizon,
                control_freq=float(base.control_freq),
                window_start=args.window_start,
                window_steps=args.window_steps,
                active_steps=active_set,
                replay=replay,
            )
            mode = "latch" if latched else requested
            if mode != previous:
                if mode == "hold":
                    entries += 1
                    ctrl.update()
                    target_pos = np.array(ctrl.ee_pos, copy=True)
                    target_ori = np.array(ctrl.ee_ori_mat, copy=True)
                    target_grip = np.array(robot.gripper.current_action, copy=True)
                    target_cube = np.array(sim.data.body_xpos[base.cube_body_id], copy=True)
                    ctrl.set_goal = held_set_goal
                elif mode == "latch":
                    target_pos = np.array(latch_goal_pos, copy=True)
                    target_ori = np.array(latch_goal_ori, copy=True)
                    target_grip = np.array(robot.gripper.current_action, copy=True)
                    target_cube = np.array(latch_cube, copy=True)
                    ctrl.set_goal = held_set_goal
                else:
                    if previous == "hold":
                        resumes += 1
                        if args.resume_state == "reset":
                            policy.policy.reset()
                    ctrl.set_goal = original
            if mode == "hold":
                action = np.zeros(7)
            elif mode == "latch":
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                action = policy(ob=obs)
                calls += 1
            obs, reward, done, _ = env.step(action)
            measure(step, requested, mode, reason, action, reward, done)
            contact_run = contact_run + 1 if rows[-1]["bilateral_contact"] else 0
            if (
                not latched
                and contact_run >= TOLERANCES["contact_steps"]
                and rows[-1]["cube_pos"][2] - rows[0]["cube_pos"][2] >= TOLERANCES["rise_m"]
            ):
                latched = True
                latch_step = step
                ctrl.update()
                latch_goal_pos = np.array(ctrl.ee_pos, copy=True) + np.array([0.0, 0.0, float(args.latch_z_offset)])
                latch_goal_ori = np.array(ctrl.ee_ori_mat, copy=True)
                latch_cube = np.array(sim.data.body_xpos[base.cube_body_id], copy=True)
            if writer and (step - 1) % 5 == 0:
                writer.append_data(env.render(mode="rgb_array", height=256, width=256, camera_name="agentview"))
            previous = mode
            if done:
                break
    finally:
        ctrl.set_goal = original
        if writer:
            writer.close()
        base.close()

    with (args.out / "telemetry.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    base_z = rows[0]["cube_pos"][2]
    first = latch_step
    drop = first is not None and any(
        r["cube_pos"][2] - base_z < TOLERANCES["rise_m"] or not r["bilateral_contact"] for r in rows[first:]
    )
    held = [r for r in rows if r["mode"] == "hold"]
    locked = [r for r in rows if r["mode"] == "latch"]
    req_policy_steps = sum(1 for r in rows[1:] if r["requested_mode"] == "policy")

    def maximum(xs):
        return max(xs, default=0.0)

    gp = maximum(float(np.linalg.norm(np.array(r["goal_pos"]) - r["target_pos"])) for r in held)
    go = maximum(float(np.max(np.abs(np.array(r["goal_ori"]) - r["target_ori"]))) for r in held)
    gr = maximum(float(np.max(np.abs(np.array(r["gripper_command"]) - r["target_grip"]))) for r in held)
    ed = maximum(float(np.linalg.norm(np.array(r["eef_pos"]) - r["target_pos"])) for r in held)
    cd = maximum(float(np.linalg.norm(np.array(r["cube_pos"]) - r["target_cube"])) for r in held)
    call_violations = sum(r["mode"] == "hold" and r["policy_calls"] != rows[i - 1]["policy_calls"] for i, r in enumerate(rows) if i)
    rnn_violations = sum(r["mode"] == "hold" and r["rnn_counter"] != rows[i - 1]["rnn_counter"] for i, r in enumerate(rows) if i)
    latch_call_violations = sum(r["mode"] == "latch" and r["policy_calls"] != rows[i - 1]["policy_calls"] for i, r in enumerate(rows) if i)
    latch_rnn_violations = sum(r["mode"] == "latch" and r["rnn_counter"] != rows[i - 1]["rnn_counter"] for i, r in enumerate(rows) if i)
    latch_pos_error = maximum(float(np.linalg.norm(np.array(r["goal_pos"]) - latch_goal_pos)) for r in locked)
    latch_ori_error = maximum(float(np.max(np.abs(np.array(r["goal_ori"]) - latch_goal_ori))) for r in locked)
    latch_close_error = maximum(float(np.max(np.abs(np.array(r["gripper_command"]) - np.array([-1.0, 1.0])))) for r in locked)
    run = best = 0
    for r in rows:
        run = run + 1 if r["bilateral_contact"] else 0
        best = max(best, run)
    retained = bool(
        first is not None
        and not drop
        and rows[-1]["bilateral_contact"]
        and rows[-1]["cube_pos"][2] - base_z >= TOLERANCES["rise_m"]
    )
    summary = dict(
        harness_version=HARNESS_VERSION,
        harness_sha256=harness_source_sha256(),
        seed=args.seed,
        condition=args.condition,
        gate=args.gate if replay else None,
        window_start=args.window_start,
        window_steps=args.window_steps,
        schedule_spans=schedule_spans,
        resume_state=args.resume_state,
        latch_z_offset_m=float(args.latch_z_offset),
        steps=len(rows) - 1,
        horizon=horizon,
        control_hz=float(base.control_freq),
        physics_dt_s=float(sim.model.opt.timestep),
        tolerances=TOLERANCES,
        requested_enable_steps=req_policy_steps,
        requested_enable_duration_s=req_policy_steps / float(base.control_freq),
        executed_policy_steps=calls,
        executed_policy_duration_s=calls / float(base.control_freq),
        executed_hold_steps=len(held),
        executed_hold_duration_s=len(held) / float(base.control_freq),
        post_acquisition_latch_steps=len(locked),
        post_acquisition_latch_duration_s=len(locked) / float(base.control_freq),
        initial_cube_pos=rows[0]["cube_pos"],
        terminal_cube_pos=rows[-1]["cube_pos"],
        terminal_rise_m=rows[-1]["cube_pos"][2] - base_z,
        max_rise_m=maximum(r["cube_pos"][2] - base_z for r in rows),
        first_acquisition_step=first,
        longest_bilateral_contact_steps=best,
        contact_proxy_validated=all(r["bilateral_contact"] == r["builtin_grasp"] for r in rows),
        post_acquisition_loss=bool(drop),
        terminal_contact=rows[-1]["bilateral_contact"],
        terminal_retained=retained,
        native_success_any=any(r["native_success"] for r in rows),
        latch_step=latch_step,
        latch_steps=len(locked),
        latch_goal_pos=None if latch_goal_pos is None else latch_goal_pos.tolist(),
        latch_goal_ori=None if latch_goal_ori is None else latch_goal_ori.tolist(),
        latch_policy_call_violations=latch_call_violations,
        latch_rnn_counter_violations=latch_rnn_violations,
        max_latch_goal_pos_error_m=latch_pos_error,
        max_latch_goal_ori_error=latch_ori_error,
        max_latch_gripper_close_error=latch_close_error,
        latch_invariants_pass=bool(
            latch_pos_error <= TOLERANCES["goal_pos_m"]
            and latch_ori_error <= TOLERANCES["goal_ori_matrix"]
            and latch_close_error <= TOLERANCES["gripper_command"]
            and latch_call_violations == 0
            and latch_rnn_violations == 0
            and (latch_step is None or len(locked) == horizon - latch_step)
        ),
        hold_steps=len(held),
        hold_entries=entries,
        resumes=resumes,
        policy_calls=calls,
        hold_policy_call_violations=call_violations,
        hold_rnn_counter_violations=rnn_violations,
        max_hold_goal_pos_error_m=gp,
        max_hold_goal_ori_error=go,
        max_hold_gripper_command_error=gr,
        max_hold_eef_drift_m=ed,
        max_hold_cube_drift_m=cd,
        hold_invariants_pass=bool(
            gp <= TOLERANCES["goal_pos_m"]
            and go <= TOLERANCES["goal_ori_matrix"]
            and gr <= TOLERANCES["gripper_command"]
            and call_violations == 0
            and rnn_violations == 0
        ),
        completion_state="complete_retained" if retained else "complete_unretained",
        checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        decisions_sha256=hashlib.sha256(args.decisions.read_bytes()).hexdigest() if args.decisions else None,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_rollout()
