"""Native BC-RNN continuous enable/HOLD/resume adapter and measured rollout."""
import argparse, bisect, hashlib, json, math
from pathlib import Path
import numpy as np, torch, imageio
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

class DecisionReplay:
    """Python 3.8 port of the existing causal DecisionReplay contract."""
    def __init__(self, path, condition, max_age_s=1.0):
        self.events=[]; self.times=[]; self.max_age_s=max_age_s
        previous=None; key=None
        for number,line in enumerate(path.open(),1):
            if not line.strip(): continue
            row=json.loads(line); est=row['estimate']; decision=row['decisions'][condition]
            source=(est['source'],est['episode']); end=float(est['window_end_s']); available=float(est['available_at_s']); issued=float(decision['issued_at_s'])
            if not all(map(math.isfinite,(end,available,issued))) or end<0 or available<end or issued<available:
                raise ValueError('Causality violation at decision line %d' % number)
            if key is not None and source!=key: raise ValueError('Mixed EEG source/episode')
            if previous and (end<=previous[0] or available<previous[1] or issued<previous[2]):
                raise ValueError('Out-of-order decision at line %d' % number)
            skill=decision['skill']
            if skill not in ('hold','grasp_lift'): raise ValueError('Unknown decision skill')
            self.events.append((issued,end,skill,decision.get('reason','recorded_replay'))); self.times.append(issued)
            previous=(end,available,issued); key=source
        if not self.events: raise ValueError('Empty EEG decision trace')
    def at(self,time_s):
        index=bisect.bisect_right(self.times,time_s)-1
        if index<0: return 'hold','awaiting_estimate'
        issued,end,skill,reason=self.events[index]
        if time_s-end>self.max_age_s: return 'hold','stale'
        return skill,reason

CKPT = Path('/home/ubuntu/robomimic-native-phase2-20261104/artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth')
TOLERANCES = dict(rise_m=0.08, contact_steps=10, goal_pos_m=1e-9, goal_ori_matrix=1e-9,
                  gripper_command=1e-9, eef_drift_m=0.035, cube_drift_m=0.025)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition',choices=['always-hold','uninterrupted','pause','eeg'],required=True)
    p.add_argument('--pause-start',type=int); p.add_argument('--pause-steps',type=int)
    p.add_argument('--decisions',type=Path); p.add_argument('--gate',choices=['raw','confidence','temporal'],default='raw')
    p.add_argument('--resume-state',choices=['freeze','reset'],default='freeze')
    p.add_argument('--video',action='store_true'); a=p.parse_args()
    if a.condition=='pause' and (not a.pause_start or not a.pause_steps): p.error('pause schedule required')
    if a.condition=='eeg' and a.decisions is None: p.error('decision trace required')
    a.out.mkdir(parents=True,exist_ok=False)
    replay=DecisionReplay(a.decisions,a.gate) if a.condition=='eeg' else None
    policy,c=FileUtils.policy_from_checkpoint(ckpt_path=str(CKPT),device=TorchUtils.get_torch_device(try_to_use_cuda=False),verbose=False)
    config,_=FileUtils.config_from_checkpoint(ckpt_dict=c); horizon=int(config.experiment.rollout.horizon)
    env,_=FileUtils.env_from_checkpoint(ckpt_dict=c,render=False,render_offscreen=a.video,verbose=False)
    np.random.seed(a.seed); torch.manual_seed(a.seed); policy.start_episode()
    obs=env.reset(); obs=env.reset_to(env.get_state())
    base=env.env; sim=base.sim; robot=base.robots[0]; ctrl=robot.controller; original=ctrl.set_goal
    left=set(robot.gripper.important_geoms['left_fingerpad']); right=set(robot.gripper.important_geoms['right_fingerpad']); obj=set(base.cube.contact_geoms)
    rows=[]; target_pos=target_ori=target_grip=target_cube=None; calls=entries=resumes=0; previous=None
    writer=imageio.get_writer(str(a.out/'video.mp4'),fps=4) if a.video else None
    def held_set_goal(action): original(action,set_pos=target_pos,set_ori=target_ori)
    def measure(step,mode,reason,action,reward,done):
        lh=rh=False; pairs=[]
        for i in range(sim.data.ncon):
            con=sim.data.contact[i]; n1=sim.model.geom_id2name(int(con.geom1)); n2=sim.model.geom_id2name(int(con.geom2))
            if (n1 in obj and n2 in left) or (n2 in obj and n1 in left): lh=True
            if (n1 in obj and n2 in right) or (n2 in obj and n1 in right): rh=True
            if n1 in obj or n2 in obj: pairs.append([n1,n2])
        rows.append(dict(step=step,time_s=step/base.control_freq,mode=mode,reason=reason,
            action=None if action is None else np.asarray(action).tolist(),cube_pos=sim.data.body_xpos[base.cube_body_id].tolist(),
            eef_pos=np.asarray(obs['robot0_eef_pos']).tolist(),goal_pos=np.asarray(ctrl.goal_pos).tolist(),goal_ori=np.asarray(ctrl.goal_ori).tolist(),
            gripper_command=np.asarray(robot.gripper.current_action).tolist(),
            target_pos=None if target_pos is None else target_pos.tolist(),target_ori=None if target_ori is None else target_ori.tolist(),
            target_grip=None if target_grip is None else target_grip.tolist(),target_cube=None if target_cube is None else target_cube.tolist(),
            bilateral_contact=lh and rh,builtin_grasp=bool(base._check_grasp(gripper=robot.gripper,object_geoms=base.cube)),contact_pairs=pairs,
            native_success=bool(env.is_success()['task']),reward=float(reward),done=bool(done),policy_calls=calls,rnn_counter=int(policy.policy._rnn_counter)))
    measure(0,'initial','reset',None,0,False)
    try:
        for step in range(1,horizon+1):
            if a.condition=='always-hold': mode,reason='hold','always_hold_control'
            elif a.condition=='uninterrupted': mode,reason='policy','uninterrupted_control'
            elif a.condition=='pause':
                mode='hold' if a.pause_start<=step<a.pause_start+a.pause_steps else 'policy'; reason='scripted_'+mode
            else:
                skill,reason=replay.at((step-1)/base.control_freq); mode='hold' if skill=='hold' else 'policy'
            if mode!=previous:
                if mode=='hold':
                    entries+=1; ctrl.update(); target_pos=np.array(ctrl.ee_pos,copy=True); target_ori=np.array(ctrl.ee_ori_mat,copy=True)
                    target_grip=np.array(robot.gripper.current_action,copy=True); target_cube=np.array(sim.data.body_xpos[base.cube_body_id],copy=True)
                    ctrl.set_goal=held_set_goal
                else:
                    if previous=='hold':
                        resumes+=1
                        if a.resume_state=='reset': policy.policy.reset()
                    ctrl.set_goal=original
            if mode=='hold': action=np.zeros(7)
            else: action=policy(ob=obs); calls+=1
            obs,reward,done,_=env.step(action); measure(step,mode,reason,action,reward,done)
            if writer and (step-1)%5==0: writer.append_data(env.render(mode='rgb_array',height=256,width=256,camera_name='agentview'))
            previous=mode
            if done: break
    finally:
        ctrl.set_goal=original
        if writer: writer.close()
        base.close()
    with (a.out/'telemetry.jsonl').open('x') as stream:
        for row in rows: stream.write(json.dumps(row,separators=(',',':'))+'\n')
    base_z=rows[0]['cube_pos'][2]
    first=next((i for i,r in enumerate(rows) if r['bilateral_contact'] and r['cube_pos'][2]-base_z>=TOLERANCES['rise_m']),None)
    drop=first is not None and any(r['cube_pos'][2]-base_z<TOLERANCES['rise_m'] or not r['bilateral_contact'] for r in rows[first:])
    held=[r for r in rows if r['mode']=='hold']
    def maximum(xs): return max(xs,default=0.0)
    gp=maximum(float(np.linalg.norm(np.array(r['goal_pos'])-r['target_pos'])) for r in held)
    go=maximum(float(np.max(np.abs(np.array(r['goal_ori'])-r['target_ori']))) for r in held)
    gr=maximum(float(np.max(np.abs(np.array(r['gripper_command'])-r['target_grip']))) for r in held)
    ed=maximum(float(np.linalg.norm(np.array(r['eef_pos'])-r['target_pos'])) for r in held)
    cd=maximum(float(np.linalg.norm(np.array(r['cube_pos'])-r['target_cube'])) for r in held)
    call_violations=sum(r['mode']=='hold' and r['policy_calls']!=rows[i-1]['policy_calls'] for i,r in enumerate(rows) if i)
    rnn_violations=sum(r['mode']=='hold' and r['rnn_counter']!=rows[i-1]['rnn_counter'] for i,r in enumerate(rows) if i)
    run=best=0
    for r in rows: run=run+1 if r['bilateral_contact'] else 0; best=max(best,run)
    retained=bool(first is not None and best>=TOLERANCES['contact_steps'] and not drop and rows[-1]['bilateral_contact'])
    summary=dict(seed=a.seed,condition=a.condition,gate=a.gate if replay else None,pause_start=a.pause_start,pause_steps=a.pause_steps,
        resume_state=a.resume_state,
        steps=len(rows)-1,horizon=horizon,control_hz=float(base.control_freq),physics_dt_s=float(sim.model.opt.timestep),tolerances=TOLERANCES,
        initial_cube_pos=rows[0]['cube_pos'],terminal_cube_pos=rows[-1]['cube_pos'],terminal_rise_m=rows[-1]['cube_pos'][2]-base_z,
        max_rise_m=maximum(r['cube_pos'][2]-base_z for r in rows),first_acquisition_step=first,longest_bilateral_contact_steps=best,
        contact_proxy_validated=all(r['bilateral_contact']==r['builtin_grasp'] for r in rows),post_acquisition_loss=bool(drop),
        terminal_contact=rows[-1]['bilateral_contact'],terminal_retained=retained,native_success_any=any(r['native_success'] for r in rows),
        hold_steps=len(held),hold_entries=entries,resumes=resumes,policy_calls=calls,hold_policy_call_violations=call_violations,
        hold_rnn_counter_violations=rnn_violations,max_hold_goal_pos_error_m=gp,max_hold_goal_ori_error=go,
        max_hold_gripper_command_error=gr,max_hold_eef_drift_m=ed,max_hold_cube_drift_m=cd,
        hold_invariants_pass=bool(gp<=TOLERANCES['goal_pos_m'] and go<=TOLERANCES['goal_ori_matrix'] and
            gr<=TOLERANCES['gripper_command'] and call_violations==0 and rnn_violations==0),
        completion_state='complete_retained' if retained else 'complete_unretained',checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),
        decisions_sha256=hashlib.sha256(a.decisions.read_bytes()).hexdigest() if a.decisions else None)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
