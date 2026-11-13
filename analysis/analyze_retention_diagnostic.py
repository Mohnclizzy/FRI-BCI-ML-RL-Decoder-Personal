"""Review seed-11 physics trace and compare exact-condition replay."""
import csv,hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
a=root/'outputs/isaac/unilateral-eval-5x12s-20261103T2310Z/seed11/ppo-telemetry.jsonl'
b=root/'outputs/isaac/retention-diagnostic-seed11-20261103T2333Z/ppo-telemetry.jsonl'
out=root/'results/retention_diagnostic_2026-11-03';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 rows=[json.loads(l) for l in p.open()];return rows,[x for x in rows if x['record_type']=='physics_step']
x,xs=load(a);y,ys=load(b)
start=next(r for r in y if r['record_type']=='episode_start');first=lambda rows,k:next(r for r in rows if r['record_type']==k)
assert len(xs)==len(ys)==1440 and all(r['contact_valid'] for r in ys)
assert all(abs((v['episode_time_s']-u['episode_time_s'])-1/120)<2e-6 for u,v in zip(ys,ys[1:]))
cond=lambda r:r['cube_height_m']>=start['baseline_cube_height_m']+.08 and min(r['finger_contact_n'])>=.1
streaks=[];beg=None
for i,r in enumerate(ys):
 if cond(r) and beg is None:beg=i
 if beg is not None and (not cond(r) or i==len(ys)-1):
  end=i-1 if not cond(r) else i
  streaks.append((beg,end));beg=None
longest=max(streaks,key=lambda z:z[1]-z[0]);loss=min(longest[1]+1,len(ys)-1)
peak=max(range(len(ys)),key=lambda i:ys[i]['finger_contact_n'][1])
right=next(i for i,r in enumerate(ys) if r['finger_contact_n'][1]>=.1 and r['finger_contact_n'][0]<.1)
push=next(i for i,r in enumerate(ys) if math.dist(r['cube_pose_w'][:2],start['cube_pose_w'][:2])>=.02 and r['cube_height_m']<start['baseline_cube_height_m']+.04)
def detail(i):
 r=ys[i]; prev=ys[max(0,i-1)];dt=r['episode_time_s']-prev['episode_time_s'];
 return {'time_s':r['episode_time_s'],'cube_pose_w':r['cube_pose_w'],'hand_pose_w':r['hand_pose_w'],'cube_minus_hand_m':[r['cube_pose_w'][j]-r['hand_pose_w'][j] for j in range(3)],'cube_speed_m_s':math.dist(r['cube_pose_w'][:3],prev['cube_pose_w'][:3])/dt if dt else None,'contacts_n':r['finger_contact_n'],'finger_positions_m':r['robot_joint_pos'][-2:],'finger_targets_m':r['processed_finger_targets']}
with (out/'trace_30hz.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s','cube_x','cube_y','cube_z','hand_x','hand_y','hand_z','left_force_n','right_force_n','left_joint_m','right_joint_m','left_target_m','right_target_m'])
 for i,r in enumerate(ys):
  if i%4==3:w.writerow([r['episode_time_s'],*r['cube_pose_w'][:3],*r['hand_pose_w'][:3],*r['finger_contact_n'],*r['robot_joint_pos'][-2:],*r['processed_finger_targets'].values()])
summary=json.loads((b.parent/'ppo-telemetry.summary.json').read_text())
result={'baseline_trace_path':str(a),'baseline_sha256':sha(a),'replay_trace_path':str(b),'replay_sha256':sha(b),'checkpoint_sha256':sha(b.parent/'input/model.pt'),'reset_cube_equal':first(x,'episode_start')['cube_pose_w']==start['cube_pose_w'],'reset_joints_equal':first(x,'episode_start')['robot_joint_pos']==start['robot_joint_pos'],'first_action_equal':first(x,'policy_step_start')['policy_action']==first(y,'policy_step_start')['policy_action'],'sample_count':len(ys),'max_cube_position_difference_m':max(math.dist(u['cube_pose_w'][:3],v['cube_pose_w'][:3]) for u,v in zip(xs,ys)),'max_contact_difference_n':max(abs(u['finger_contact_n'][j]-v['finger_contact_n'][j]) for u,v in zip(xs,ys) for j in (0,1)),'right_only':detail(right),'low_push_2cm':detail(push),'peak_right_force':detail(peak),'longest_lift_bilateral_start':detail(longest[0]),'longest_lift_bilateral_end':detail(longest[1]),'first_sample_after_longest':detail(loss),'longest_samples':longest[1]-longest[0]+1,'summary':summary,'trace_30hz_sha256':sha(out/'trace_30hz.csv')}
(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('summary','baseline_trace_path','replay_trace_path')},indent=2))
