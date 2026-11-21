"""Independent checks on the frozen 24-run native continuous-control batch."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
records={}; errors=[]
for folder in sorted(a.run.iterdir()):
    if not folder.is_dir() or not (folder/'summary.json').exists(): continue
    s=json.loads((folder/'summary.json').read_text())
    rows=[json.loads(line) for line in (folder/'telemetry.jsonl').open()]
    if len(rows)!=401 or s['steps']!=400: errors.append(folder.name+': deadline')
    if any(r['step']!=i or abs(r['time_s']-i/20)>1e-10 for i,r in enumerate(rows)): errors.append(folder.name+': clock')
    if any(r['bilateral_contact']!=r['builtin_grasp'] for r in rows): errors.append(folder.name+': contact proxy')
    if any(not np.all(np.isfinite(r['cube_pos'])) for r in rows): errors.append(folder.name+': nonfinite cube')
    for i,r in enumerate(rows[1:],1):
        if r['mode']=='hold':
            if r['policy_calls']!=rows[i-1]['policy_calls'] or r['rnn_counter']!=rows[i-1]['rnn_counter']: errors.append(folder.name+': learned call in HOLD')
            if np.linalg.norm(np.array(r['goal_pos'])-r['target_pos'])>1e-9: errors.append(folder.name+': pose goal')
            if np.max(np.abs(np.array(r['goal_ori'])-r['target_ori']))>1e-9: errors.append(folder.name+': orientation goal')
            if np.max(np.abs(np.array(r['gripper_command'])-r['target_grip']))>1e-9: errors.append(folder.name+': gripper goal')
        elif r['policy_calls']!=rows[i-1]['policy_calls']+1: errors.append(folder.name+': resume/policy call')
    if s['condition']=='pause' and s['resumes']!=1: errors.append(folder.name+': resume count')
    if s['max_hold_eef_drift_m']>s['tolerances']['eef_drift_m'] or s['max_hold_cube_drift_m']>s['tolerances']['cube_drift_m']:
        errors.append(folder.name+': physical drift tolerance')
    initial=rows[0]['cube_pos'][2]
    acquired=next((i for i,r in enumerate(rows) if r['bilateral_contact'] and r['cube_pos'][2]-initial>=0.08),None)
    post=rows[acquired:] if acquired is not None else []
    loss=any(not r['bilateral_contact'] or r['cube_pos'][2]-initial<0.08 for r in post)
    run=longest=0
    for r in post:
        run=run+1 if r['bilateral_contact'] else 0; longest=max(longest,run)
    retained=bool(acquired is not None and longest>=10 and not loss and post[-1]['bilateral_contact'])
    if acquired!=s['first_acquisition_step'] or retained!=s['terminal_retained']: errors.append(folder.name+': independent endpoint')
    if s['condition']=='pause':
        before=rows[s['pause_start']-1]
        phase=dict(cube_rise_m=before['cube_pos'][2]-initial,bilateral_contact=before['bilateral_contact'])
    else: phase=None
    records[folder.name]=dict(seed=s['seed'],condition=s['condition'],pause_start=s['pause_start'],pause_steps=s['pause_steps'],
        first_acquisition_step=acquired,terminal_retained=retained,post_acquisition_loss=loss,
        terminal_rise_m=s['terminal_rise_m'],terminal_contact=s['terminal_contact'],
        max_hold_eef_drift_m=s['max_hold_eef_drift_m'],max_hold_cube_drift_m=s['max_hold_cube_drift_m'],
        hold_invariants_pass=s['hold_invariants_pass'],observed_pause_entry=phase,
        trace_sha256=hashlib.sha256((folder/'telemetry.jsonl').read_bytes()).hexdigest())
for seed in (0,1,2):
    baseline=[json.loads(x) for x in (a.run/('uninterrupted_seed%d'%seed)/'telemetry.jsonl').open()]
    for name,r in records.items():
        if r['seed']!=seed or r['condition']!='pause': continue
        trial=[json.loads(x) for x in (a.run/name/'telemetry.jsonl').open()]
        n=r['pause_start']
        if not np.allclose([x['cube_pos'] for x in baseline[:n]],[x['cube_pos'] for x in trial[:n]],atol=1e-10,rtol=0):
            errors.append(name+': unmatched prefix')
        if not np.allclose([x['action'] for x in baseline[1:n]],[x['action'] for x in trial[1:n]],atol=1e-10,rtol=0):
            errors.append(name+': unmatched action prefix')
by={}
for r in records.values():
    key=r['condition'] if r['condition']!='pause' else 'pause_%d_%d'%(r['pause_start'],r['pause_steps'])
    by.setdefault(key,[]).append(r)
compact={k:dict(runs=len(v),acquisitions=sum(x['first_acquisition_step'] is not None for x in v),
    retained=sum(x['terminal_retained'] for x in v),post_acquisition_losses=sum(x['post_acquisition_loss'] for x in v),
    max_hold_eef_drift_m=max(x['max_hold_eef_drift_m'] for x in v),max_hold_cube_drift_m=max(x['max_hold_cube_drift_m'] for x in v)) for k,v in sorted(by.items())}
result=dict(run_dir=str(a.run),run_count=len(records),audit_errors=errors,groups=compact,runs=records)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(run_count=len(records),audit_errors=errors,groups=compact),indent=2))
if len(records)!=24 or errors: raise SystemExit(1)
