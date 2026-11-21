"""Independent audit of corrective closure and one reset-on-resume intervention."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/home/ubuntu/robomimic-native-phase4-20261104')
OUT=Path(__file__).with_name('followup_metrics.json')
results={}; errors=[]
for group in ('closure_corrective','resume_reset_diagnostic'):
    results[group]={}
    for folder in sorted((ROOT/group).iterdir()):
        if not folder.is_dir(): continue
        s=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(x) for x in (folder/'telemetry.jsonl').open()]
        e=[]
        if len(rows)!=401 or s['steps']!=400 or s['resumes']!=1: e.append('clock/resume')
        if any(abs(r['time_s']-i/20)>1e-10 for i,r in enumerate(rows)): e.append('clock times')
        if any(r['bilateral_contact']!=r['builtin_grasp'] for r in rows): e.append('contact proxy')
        for i,r in enumerate(rows[1:],1):
            if r['mode']=='hold':
                if r['policy_calls']!=rows[i-1]['policy_calls'] or r['rnn_counter']!=rows[i-1]['rnn_counter']: e.append('learned call in hold')
                if np.linalg.norm(np.array(r['goal_pos'])-r['target_pos'])>1e-9: e.append('pose goal')
                if np.max(np.abs(np.array(r['goal_ori'])-r['target_ori']))>1e-9: e.append('orientation goal')
                if np.max(np.abs(np.array(r['gripper_command'])-r['target_grip']))>1e-9: e.append('gripper target')
            elif r['policy_calls']!=rows[i-1]['policy_calls']+1: e.append('policy calls')
        if s['max_hold_eef_drift_m']>0.035 or s['max_hold_cube_drift_m']>0.025: e.append('physical drift')
        baseline=rows[0]['cube_pos'][2]
        first=next((i for i,r in enumerate(rows) if r['bilateral_contact'] and r['cube_pos'][2]-baseline>=.08),None)
        post=rows[first:] if first is not None else []
        loss=any(not r['bilateral_contact'] or r['cube_pos'][2]-baseline<.08 for r in post)
        run=best=0
        for r in post: run=run+1 if r['bilateral_contact'] else 0; best=max(best,run)
        retained=bool(first is not None and best>=10 and not loss and post[-1]['bilateral_contact'])
        if retained!=s['terminal_retained'] or first!=s['first_acquisition_step']: e.append('endpoint')
        results[group][folder.name]=dict(seed=s['seed'],pause_start=s['pause_start'],pause_steps=s['pause_steps'],
            resume_state=s.get('resume_state','freeze'),retained=retained,acquired=first is not None,post_acquisition_loss=loss,
            max_hold_eef_drift_m=s['max_hold_eef_drift_m'],max_hold_cube_drift_m=s['max_hold_cube_drift_m'],audit_errors=e)
        errors.extend(folder.name+':'+x for x in e)
result=dict(run_count=sum(map(len,results.values())),audit_errors=errors,groups=results)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(run_count=result['run_count'],audit_errors=errors),indent=2))
if result['run_count']!=12 or errors: raise SystemExit(1)
