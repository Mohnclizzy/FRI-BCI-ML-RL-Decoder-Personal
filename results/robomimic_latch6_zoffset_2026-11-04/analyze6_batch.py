"""Phase 6 audit: runs the IDENTICAL Phase 5 gate/invariants against the Phase 6
matched batch, then compares Phase 5 vs Phase 6 run-by-run. The single Phase 6
change is the +0.015 m world-z lift of the held goal at acquisition (latch capture).
Phase 5 dir is left untouched and used only as the read-only baseline for comparison."""
import hashlib, json
from pathlib import Path
import numpy as np

RAW6 = Path('/home/ubuntu/robomimic-native-phase6-20261104/matched_batch')
RAW5 = Path('/home/ubuntu/robomimic-native-phase5-20261104/matched_batch')
RISE, CONTACT = 0.08, 10

def audit(root):
    out = {}
    for path in sorted(root.glob('*/summary.json')):
        name = path.parent.name
        s = json.loads(path.read_text())
        trace = path.parent / 'telemetry.jsonl'
        rows = [json.loads(l) for l in trace.read_text().splitlines()]
        assert len(rows) == 401 and [r['step'] for r in rows] == list(range(401)), name
        assert all(abs(r['time_s'] - r['step'] / 20) < 1e-9 for r in rows), name
        assert all(r['bilateral_contact'] == r['builtin_grasp'] for r in rows), name
        base = rows[0]['cube_pos'][2]
        latch = s['latch_step']
        trigger = None; run = 0
        for r in rows[1:]:
            run = run + 1 if r['bilateral_contact'] else 0
            if run >= CONTACT and r['cube_pos'][2] - base >= RISE:
                trigger = r['step']; break
        assert latch == trigger, name
        if latch is not None:
            assert all(r['mode'] == 'latch' for r in rows[latch + 1:]), name
        losses = [r['step'] for r in rows[latch:] if (not r['bilateral_contact'] or r['cube_pos'][2] - base < RISE)] if latch is not None else []
        retained = bool(latch is not None and not losses and rows[-1]['bilateral_contact'] and rows[-1]['cube_pos'][2] - base >= RISE)
        assert retained == s['terminal_retained'], name
        hold = [r for r in rows if r['mode'] == 'hold']; locked = [r for r in rows if r['mode'] == 'latch']
        def maxnorm(kind, target, group):
            return max((float(np.linalg.norm(np.asarray(r[kind]) - np.asarray(r[target]))) for r in group), default=0.)
        hold_goal = maxnorm('goal_pos', 'target_pos', hold)
        latch_goal = maxnorm('goal_pos', 'latch_goal_pos', locked)
        hold_cmd = max((float(np.max(np.abs(np.asarray(r['gripper_command']) - np.asarray(r['target_grip'])))) for r in hold), default=0.)
        latch_cmd = max((float(np.max(np.abs(np.asarray(r['gripper_command']) - np.asarray([-1., 1.])))) for r in locked), default=0.)
        no_calls = all(r['policy_calls'] == rows[r['step'] - 1]['policy_calls'] and r['rnn_counter'] == rows[r['step'] - 1]['rnn_counter'] for r in hold + locked)
        close_actions = all(r['action'] == [0., 0., 0., 0., 0., 0., 1.] for r in locked)
        assert hold_goal <= 1e-9 and latch_goal <= 1e-9 and hold_cmd <= 1e-9 and latch_cmd <= 1e-9 and no_calls and close_actions, name
        out[name] = dict(seed=s['seed'], condition=s['condition'], pause_start=s['pause_start'], pause_steps=s['pause_steps'],
                         latch_step=latch, first_post_latch_loss_step=losses[0] if losses else None,
                         terminal_rise_m=round(s['terminal_rise_m'], 6), terminal_contact=rows[-1]['bilateral_contact'],
                         retained=retained, hold_steps=len(hold), latch_steps=len(locked), policy_calls=s['policy_calls'],
                         all_invariants_pass=True, deadline_pass=s['steps'] == 400,
                         trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    assert len(out) == 24, len(out)
    for seed in range(3):
        names = [n for n, v in out.items() if v['seed'] == seed]
        initials = [json.loads((root / n / 'telemetry.jsonl').open().readline())['cube_pos'] for n in names]
        assert all(p == initials[0] for p in initials), seed
    return out

p6 = audit(RAW6)
p5 = audit(RAW5)  # read-only baseline

metrics = {
    'protocol': 'protocol6.md',
    'intervention': {'name': 'latch_z_offset_m', 'value_m': 0.015, 'applied': 'once at acquisition, held goal frozen thereafter',
                     'applies_to': 'latch capture in latch6_rollout.py; all other logic byte-identical to Phase 5'},
    'runs': p6,
    'positive_acquired': sum(v['latch_step'] is not None for v in p6.values() if v['condition'] != 'always-hold'),
    'positive_retained': sum(v['retained'] for v in p6.values() if v['condition'] != 'always-hold'),
    'negative_acquired': sum(v['latch_step'] is not None for v in p6.values() if v['condition'] == 'always-hold'),
    'all_deadlines_pass': all(v['deadline_pass'] for v in p6.values()),
    'all_invariants_pass': all(v['all_invariants_pass'] for v in p6.values()),
    'checkpoint_sha256': json.loads((RAW6 / 'always-enable_seed0' / 'summary.json').read_text())['checkpoint_sha256'],
}

# Phase 5 -> Phase 6 per-run comparison (the decision table)
comparison = []
for name in sorted(p6):
    a, b = p5[name], p6[name]
    comparison.append({
        'run': name, 'condition': a['condition'], 'seed': a['seed'],
        'p5_latch_step': a['latch_step'], 'p5_retained': a['retained'], 'p5_terminal_rise_m': a['terminal_rise_m'],
        'p6_latch_step': b['latch_step'], 'p6_retained': b['retained'], 'p6_terminal_rise_m': b['terminal_rise_m'],
        'delta_retained': b['retained'] - a['retained'],
        'recovered': (not a['retained']) and b['retained'],
        'regressed': a['retained'] and (not b['retained']),
    })
recovered = [c['run'] for c in comparison if c['recovered']]
regressed = [c['run'] for c in comparison if c['regressed']]
still_fail = [c['run'] for c in comparison if c['condition'] != 'always-hold' and not c['p6_retained']]
comparison_summary = {
    'total_runs': len(comparison),
    'p5_positive_retained': metrics.get('p5_positive_retained', sum(c['p5_retained'] for c in comparison if c['condition'] != 'always-hold')),
    'p6_positive_retained': metrics['positive_retained'],
    'recovered': recovered, 'regressed': regressed, 'still_failing_positive': still_fail,
    'net_change_in_positive_retention': metrics['positive_retained'] - sum(c['p5_retained'] for c in comparison if c['condition'] != 'always-hold'),
}

result = {**metrics, 'comparison': comparison, 'comparison_summary': comparison_summary}
Path('results/robomimic_latch6_zoffset_2026-11-04/metrics6.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
