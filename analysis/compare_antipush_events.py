"""Audit saved trace timing and reset matching; event definitions are descriptive."""
import hashlib, json, math
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R/'outputs/isaac'

def read(path):
    return [json.loads(s) for s in path.open()]

def event(rows, ep, label, seed):
    rows=[x for x in rows if x['episode']==ep]
    start=next(x for x in rows if x['record_type']=='episode_start')
    s=[x for x in rows if x['record_type']=='physics_step']
    end=next(x for x in rows if x['record_type'] in ('terminal_pre_reset','episode_incomplete_end'))
    base=start['cube_pose_w']
    def first(fn):
        row=next((x for x in s if fn(x)),None)
        if row is None: return None
        return {'time_s':row['episode_time_s'], 'cube_xy_m':round(math.dist(base[:2],row['cube_pose_w'][:2]),5), 'rise_m':round(row['cube_height_m']-base[2],5), 'contacts_n':row['finger_contact_n'], 'finger_positions_m':row['robot_joint_pos'][-2:]}
    return {'condition':label,'seed':seed,'episode':ep,'start_cube_pose':base,'start_joints':start['robot_joint_pos'],'right_only':first(lambda x:x['finger_contact_n'][1]>=0.1 and x['finger_contact_n'][0]<0.1),'xy_2cm_below_4cm':first(lambda x:math.dist(base[:2],x['cube_pose_w'][:2])>=.02 and x['cube_height_m']<base[2]+.04),'either_finger_le_20mm':first(lambda x:min(x['robot_joint_pos'][-2:])<=.02),'both_fingers_le_20mm':first(lambda x:max(x['robot_joint_pos'][-2:])<=.02),'lift_8cm':first(lambda x:x['cube_height_m']>=base[2]+.08),'end_type':end['record_type'],'end_time_s':end['episode_time_s']}

out=[]
ap=P/'table-approach-eval-5x12s-seed8-20261102T202013Z/ppo-telemetry.jsonl'
r=read(ap)
for i in range(1,6):
    x=event(r,i,'approach',8)
    x['source_sha256']=hashlib.sha256(ap.read_bytes()).hexdigest()
    out.append(x)
for label,path in [('progress','near-table-progress-eval-5x12s-seed8-12-20261102T2330Z'),('antipush','antipush-eval-5x12s-20261103T2305Z')]:
    for seed in range(8,13):
        f=P/path/f'seed{seed}'/'ppo-telemetry.jsonl'
        x=event(read(f),1,label,seed)
        summary=json.loads((f.parent/'ppo-telemetry.summary.json').read_text())['episode_results'][0]
        x.update({'acquisition_s':summary['acquisition_time_s'],'drop_s':summary['drop_times_s'], 'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
        out.append(x)
res=R/'results/trace_comparison_2026-11-03.json'
res.write_text(json.dumps(out,indent=2)+'\n')
for x in out:
    print(x['condition'],x['seed'],x['episode'],*[None if x[k] is None else x[k]['time_s'] for k in ('right_only','xy_2cm_below_4cm','either_finger_le_20mm','both_fingers_le_20mm','lift_8cm')],x.get('acquisition_s'),x.get('drop_s'),x['end_time_s'])
print('pose match progress/anti', [out[5+i]['start_joints']==out[10+i]['start_joints'] and out[5+i]['start_cube_pose']==out[10+i]['start_cube_pose'] for i in range(5)])
print('approach/anti joint equal',out[0]['start_joints']==out[10]['start_joints'])
