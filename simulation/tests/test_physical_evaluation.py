import math

import pytest

from fri_bci_sim.physical_evaluation import (
    PhysicalThresholds, evaluate_physical_rows, evaluate_request_target_delays,
)
from fri_bci_sim.replay_evaluation import AnnotatedTrial, LabelInterval


def row(t, *, q=0.0, target=0.0, z=0.28, force=0.0, skill="hold", finger=0.04):
    return {"time_s": t, "interval_start_s": max(0, t - 0.1),
            "joint_pos": [q], "joint_target": [target], "requested_skill": skill,
            "hand_pose_w": [q, 0, 0.4, 0, 0, 0, 1],
            "cube_pose_w": [0, 0, z, 0, 0, 0, 1],
            "finger_contact_n": [force, force], "finger_target_width_m": finger}


def test_drifting_hold_fails_motion_evidence_despite_stationary_cube():
    rows = [row(i / 10, q=i / 100, target=i / 100) for i in range(11)]
    trial = AnnotatedTrial("s", "s", 1, (LabelInterval(0.25, 0.75, "rest"),))
    score = evaluate_physical_rows(rows, trial=trial)
    assert not score["sampled_physical_lift_pass"]
    assert score["max_hand_displacement_from_start_m"] == pytest.approx(0.1)
    assert score["max_hold_target_excursion_rad"] == pytest.approx(0.1)
    assert score["rest_motion"]["observed_rest_duration_s"] == pytest.approx(0.5)
    assert score["rest_motion"]["moving_duration_s"] == pytest.approx(0.5)
    assert score["rest_motion"]["hand_path_length_m"] == pytest.approx(0.05)


def test_measured_contact_lift_then_drop_requires_duration_and_height():
    rows = [row(0)] + [row(i / 10, z=0.4, force=1) for i in range(1, 8)]
    rows += [row(i / 10) for i in range(8, 12)]
    score = evaluate_physical_rows(rows)
    assert score["sampled_physical_lift_pass"]
    assert score["first_sampled_success_time_s"] == pytest.approx(0.6)
    assert score["drops_after_sampled_acquisition"] == 1
    assert score["drop_detection_times_s"] == pytest.approx([1.1])
    # Height alone and a request cannot establish a grasp.
    no_contact = [dict(r, finger_contact_n=[0, 0], requested_skill="grasp_lift") for r in rows]
    assert not evaluate_physical_rows(no_contact)["sampled_physical_lift_pass"]


def test_missing_samples_break_sustained_contact_and_are_not_rest_observations():
    rows = [row(0), row(0.1, z=0.4, force=1), row(0.2, z=0.4, force=1),
            row(1.0, z=0.4, force=1), row(1.1, z=0.4, force=1)]
    trial = AnnotatedTrial("s", "s", 1.2, (LabelInterval(0, 1.2, "rest"),))
    score = evaluate_physical_rows(rows, trial=trial)
    assert not score["sampled_physical_lift_pass"]
    assert score["observed_interval_duration_s"] == pytest.approx(0.3)
    assert score["unobserved_duration_s"] == pytest.approx(0.9)
    assert score["rest_motion"]["observed_rest_duration_s"] == pytest.approx(0.3)


def test_dispatch_delay_includes_finger_only_action_and_censors_no_action():
    rows = [row(0), row(0.1, skill="grasp_lift"), row(0.2, skill="grasp_lift", finger=0.02),
            row(0.3, finger=0.02), row(0.4, skill="grasp_lift", finger=0.02)]
    events = [{"time_s": 0.05, "skill": "grasp_lift"}, {"time_s": 0.25, "skill": "hold"},
              {"time_s": 0.35, "skill": "grasp_lift"}]
    score = evaluate_request_target_delays(rows, events)
    assert score["spans_with_target_change"] == 1
    assert score["mean_target_change_delay_s"] == pytest.approx(0.05)
    assert score["events"][1]["right_censored"]


@pytest.mark.parametrize("bad", [
    [row(0), row(0)],
    [row(0), dict(row(0.1), joint_pos=[math.nan])],
    [row(0), dict(row(0.1), done=True)],
    [row(0), dict(row(0.1), joint_target=[0, 1])],
])
def test_rejects_invalid_or_reset_telemetry(bad):
    with pytest.raises(ValueError):
        evaluate_physical_rows(bad)


def test_hold_after_active_compares_previous_target_and_carry_contact_loss_is_not_drop_alone():
    rows = [row(0)] + [row(i / 10, z=0.4, force=1, skill="grasp_lift", target=0.2) for i in range(1, 8)]
    rows += [row(i / 10, z=0.4, force=0, target=0.2) for i in range(8, 12)]
    score = evaluate_physical_rows(rows)
    assert score["sampled_physical_lift_pass"]
    assert score["drops_after_sampled_acquisition"] == 0
    assert score["max_hold_target_excursion_rad"] == 0


def test_fall_below_height_is_drop_even_when_contact_remains():
    rows = [row(0)] + [row(i / 10, z=0.4, force=1, skill="grasp_lift") for i in range(1, 8)]
    rows += [row(i / 10, z=0.3, force=1, skill="grasp_lift") for i in range(8, 12)]
    score = evaluate_physical_rows(rows)
    assert score["drops_after_sampled_acquisition"] == 1
    assert score["drop_detection_times_s"] == pytest.approx([1.1])
