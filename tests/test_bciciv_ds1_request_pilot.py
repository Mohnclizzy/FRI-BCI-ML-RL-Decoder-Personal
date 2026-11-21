import numpy as np
import pytest

from analysis.bciciv_ds1_request_pilot import (
    classify_request_state,
    run_dwell_controller,
    run_leaky_accumulator,
    run_periodic_clock,
    score_requests_multi_run,
    slot_rotation_control,
)


def test_transition_classification_distinguishes_start_and_stop_cues():
    # [0..1s] = REST (0), [1..2s] = NaN transition to imagery, [2..5s] = IMAGERY (1),
    # [5..6s] = NaN transition to REST, [6..9s] = REST (0)
    y_1000 = np.zeros(9000, dtype=float)
    y_1000[1000:2000] = np.nan
    y_1000[2000:5000] = 1.0
    y_1000[5000:6000] = np.nan
    intervals = [(0.0, 1.0, 0), (2.0, 5.0, 1), (6.0, 9.0, 0)]

    st1, idx1, sub1, cue1 = classify_request_state(1.5, intervals, y_1000)
    assert (st1, idx1, sub1, cue1) == ("transition", None, "start_cue_1s_transition_to_imagery", 1.0)

    st2, idx2, sub2, cue2 = classify_request_state(5.5, intervals, y_1000)
    assert (st2, idx2, sub2, cue2) == ("transition", None, "stop_cue_1s_transition_to_rest", 5.0)


def test_short_vs_long_episode_accounting_and_slot_rotation():
    # Unmasked [5.0, 6.2) -> full cue duration 2.2s <= 2.5s (short); [13.0, 17.5) -> full cue 5.5s > 2.5s (long)
    intervals = [[(0.0, 4.0, 0), (5.0, 6.2, 1), (8.0, 12.0, 0), (13.0, 17.5, 1)]]
    reqs = [[5.5, 14.0, 15.5]]
    m = score_requests_multi_run(reqs, intervals)
    assert m["n_imagery_episodes"] == 2
    assert m["detected_episodes"] == 2
    assert m["duplicate_requests"] == 1
    assert m["short_imagery_episodes_le_2_5s"] == 1
    assert m["short_detected_episodes"] == 1
    assert m["long_imagery_episodes_gt_2_5s"] == 1
    assert m["long_detected_episodes"] == 1

    accepted = [{"eval_run": 2, "slot_index": 0, "slot_relative_time_s": 5.5}]
    rot = slot_rotation_control(accepted, {2: intervals[0]}, {2: 1})
    assert rot["rotated_mean_imagery_starts"] == pytest.approx(1.0)
