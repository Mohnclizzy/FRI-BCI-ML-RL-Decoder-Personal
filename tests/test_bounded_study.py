import json
import numpy as np
import pytest
from eeg_decoder.bounded_study import (schedule, matched_schedules, trace_for, aggregate,
                                       gate_specs, SUBJECTS, SEEDS)
from eeg_decoder.command_study_metrics import evaluate_gate_on_r08
from fri_bci_sim.robomimic_harness import CausalDecisionReplay, spans_to_active_steps


def test_causal_trace_matches_native_clock_and_stale_hold(tmp_path):
    spec = dict(threshold=.55, margin=.02, persistence_s=.25)
    trace = trace_for([.9, .9, .9], [2., 2.25, 2.5], spec, 1, 1.)
    path = tmp_path/'trace.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in trace))
    replay = CausalDecisionReplay(path, 'raw')
    assert replay.at(1.99)[0] == 'hold'
    assert replay.at(2.)[0] == 'hold'
    assert replay.at(2.25)[0] == 'grasp_lift'
    assert replay.at(3.51)[0] == 'hold'
    assert schedule(trace) == [s+1 for s in range(400) if replay.at(s/20)[0] == 'grasp_lift']


def test_rest_carryover_is_counted():
    result = evaluate_gate_on_r08(np.array([.9, .9, .1]), np.array([2.,2.25,2.5]),
        [[0.,2.3,1], [2.3,4.,0]], threshold=.5, margin=0., persistence_s=0.)
    assert result['rest_command_fraction'] == pytest.approx(.2/1.7)


@pytest.mark.parametrize('steps', [[], [1,2,9,200,399,400], list(range(1,401))])
def test_matched_controls_preserve_duration_and_shift(steps):
    controls = matched_schedules(steps)
    assert spans_to_active_steps(controls['contiguous'],400) == set(range(1,len(steps)+1))
    assert spans_to_active_steps(controls['shifted'],400) == {((s-1+200)%400)+1 for s in steps}


def test_seed_averaging_precedes_utility_and_missing_seeds_fail():
    rows=[]
    for subject in SUBJECTS:
        for seed, rest in zip(SEEDS, [.2,.2,.8]):
            rows.append(dict(name='eegnet/raw',subject=subject,seed=seed,
                metrics=dict(rest_command_fraction=rest, imagery_command_coverage=.8, enabled_seconds_20s=5.)))
    people, groups = aggregate(rows)
    assert len(people)==15
    assert groups[0]['pass_fraction']==0
    with pytest.raises(ValueError, match='seeds'):
        aggregate(rows[:-1])


def test_fixed_gate_candidates():
    specs=gate_specs(np.array([.6,.8,.2,.1]),np.array([2.,2.25,2.5,2.75]),[[0,2.5,0],[2.5,4,1]],
                     dict(max_age_s=1.,alpha=.1,hysteresis_margin=.05,persistence_s=.5))
    assert set(specs)=={'raw','confidence','temporal','moderate','rest_q80','crc'}
    assert specs['moderate']==dict(threshold=.55,margin=.02,persistence_s=.25)
    assert specs['crc']['threshold']==1.1
