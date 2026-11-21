import pytest

from analysis.request_capacity import request_capacity


def test_half_open_slots_and_unobserved_tail():
    # The last episode begins exactly at the end of the complete slots.
    intervals = [(1, 4, 1), (5, 20, 1), (20, 22, 1), (40, 42, 1)]
    report = request_capacity(intervals, 45)
    assert report["complete_slots"] == 2
    assert report["incomplete_tail_s"] == 5
    assert report["imagery_overlapping_robot_horizon"] == 3
    assert report["max_distinct_matches"] == 2
    assert report["full_recording_ceiling"] == pytest.approx(2 / 4)
    assert report["robot_horizon_ceiling"] == pytest.approx(2 / 3)


def test_many_slots_cannot_count_one_episode_repeatedly():
    report = request_capacity([(5, 55, 1), (56, 63, 1)], 65)
    assert report["imagery_overlapping_robot_horizon"] == 2
    assert report["max_distinct_matches"] == 2


def test_episodes_entirely_before_second_slot_cannot_use_it():
    report = request_capacity([(0, 4, 1), (12, 20, 1)], 40)
    assert report["max_distinct_matches"] == 1
