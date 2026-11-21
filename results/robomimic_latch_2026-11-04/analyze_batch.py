"""Independent trace-level audit of the frozen Phase 5 matched batch."""
import hashlib, json
from pathlib import Path
import numpy as np

root=Path('/home/ubuntu/robomimic-native-phase5-20261104/matched_batch')
out={}
for path in sorted(root.glob('*/summary.json')):
    name=path.parent.name
    s=json.loads(path.read_text())
    trace=path.parent/'telemetry.jsonl'
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows)==401 and [r['step'] for r in rows]==list(range(401)), name
    assert all(abs(r['time_s']-r['step']/20)<1e-9 for r in rows), name
    assert all(r['bilateral_contact']==r['builtin_grasp'] for r in rows), name
    base=rows[0]['cube_pos'][2]
    latch=s['latch_step']
    trigger=None; run=0
    for r in rows[1:]:
        run=run+1 if r['bilateral_contact'] else 0
        if run>=10 and r['cube_pos'][2]-base>=.08:
            trigger=r['step']; break
    assert latch==trigger, name
    if latch is not None:
        assert all(r['mode']=='latch' for r in rows[latch+1:]), name
    losses=[]
    if latch is not None:
        losses=[r['step'] for r in rows[latch:] if not r['bilateral_contact'] or r['cube_pos'][2]-base<.08]
    retained=bool(latch is not None and not losses and rows[-1]['bilateral_contact'] and rows[-1]['cube_pos'][2]-base>=.08)
    assert retained==s['terminal_retained'], name
    hold=[r for r in rows if r['mode']=='hold']; locked=[r for r in rows if r['mode']=='latch']
    def maxnorm(kind, target, group):
        return max((float(np.linalg.norm(np.asarray(r[kind])-np.asarray(r[target]))) for r in group),default=0.)
    hold_goal=maxnorm('goal_pos','target_pos',hold)
    latch_goal=maxnorm('goal_pos','latch_goal_pos',locked)
    hold_cmd=max((float(np.max(np.abs(np.asarray(r['gripper_command'])-np.asarray(r['target_grip'])))) for r in hold),default=0.)
    latch_cmd=max((float(np.max(np.abs(np.asarray(r['gripper_command'])-[-1.,1.]))) for r in locked),default=0.)
    no_calls=all(r['policy_calls']==rows[r['step']-1]['policy_calls'] and r['rnn_counter']==rows[r['step']-1]['rnn_counter'] for r in hold+locked)
    close_actions=all(r['action']==[0.,0.,0.,0.,0.,0.,1.] for r in locked)
    assert hold_goal<=1e-9 and latch_goal<=1e-9 and hold_cmd<=1e-9 and latch_cmd<=1e-9 and no_calls and close_actions, name
    pause_entry=next((r for r in hold if r['step']==s['pause_start']),None)
    out[name]={
        'seed':s['seed'],'condition':s['condition'],'pause_start':s['pause_start'],'pause_steps':s['pause_steps'],
        'pause_entry_rise_m':None if pause_entry is None else pause_entry['cube_pos'][2]-base,
        'pause_entry_bilateral_contact':None if pause_entry is None else pause_entry['bilateral_contact'],
        'latch_step':latch,'first_post_latch_loss_step':losses[0] if losses else None,
        'loss_type_at_first':None if not losses else {'contact':not rows[losses[0]]['bilateral_contact'],'rise':rows[losses[0]]['cube_pos'][2]-base<.08},
        'terminal_rise_m':s['terminal_rise_m'],'terminal_contact':s['terminal_contact'],'retained':retained,
        'max_hold_eef_drift_m':s['max_hold_eef_drift_m'],'max_hold_cube_drift_m':s['max_hold_cube_drift_m'],
        'hold_steps':len(hold),'latch_steps':len(locked),'policy_calls':s['policy_calls'],
        'all_invariants_pass':True,'deadline_pass':s['steps']==400,
        'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
    }
assert len(out)==24, len(out)
for seed in range(3):
    names=[name for name,v in out.items() if v['seed']==seed]
    initials=[json.loads((root/name/'telemetry.jsonl').open().readline())['cube_pos'] for name in names]
    assert all(p==initials[0] for p in initials), seed
metrics={'protocol':'protocol.md','runs':out,'positive_acquired':sum(v['latch_step'] is not None for v in out.values() if v['condition']!='always-hold'),
         'positive_retained':sum(v['retained'] for v in out.values() if v['condition']!='always-hold'),
         'negative_acquired':sum(v['latch_step'] is not None for v in out.values() if v['condition']=='always-hold'),
         'all_deadlines_pass':all(v['deadline_pass'] for v in out.values()),
         'all_invariants_pass':all(v['all_invariants_pass'] for v in out.values())}
Path('results/robomimic_latch_2026-11-04/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({k:v for k,v in metrics.items() if k!='runs'},indent=2))
