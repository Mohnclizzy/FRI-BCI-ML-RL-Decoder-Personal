import argparse, json, hashlib, os
from pathlib import Path
import numpy as np, torch, imageio
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--out',required=True); p.add_argument('--video',action='store_true'); p.add_argument('--resume-step',type=int,default=0); a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
ckpt='/home/ubuntu/robomimic-native-phase2-20261104/artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth'
policy, c=FileUtils.policy_from_checkpoint(ckpt_path=ckpt,device=TorchUtils.get_torch_device(try_to_use_cuda=False),verbose=False)
config,_=FileUtils.config_from_checkpoint(ckpt_dict=c)
horizon=int(config.experiment.rollout.horizon)
env,_=FileUtils.env_from_checkpoint(ckpt_dict=c,render=False,render_offscreen=a.video,verbose=False)
np.random.seed(a.seed); torch.manual_seed(a.seed)
policy.start_episode(); obs=env.reset(); start_state=env.get_state(); obs=env.reset_to(start_state)
base=env.env; sim=base.sim; table=float(base.model.mujoco_arena.table_offset[2]); dt=float(sim.model.opt.timestep)
control=float(base.control_freq); cube_id=base.cube_body_id
left=base.robots[0].gripper.important_geoms['left_fingerpad']; right=base.robots[0].gripper.important_geoms['right_fingerpad']; obj=base.cube.contact_geoms
left=set(left if isinstance(left,(list,tuple)) else [left]); right=set(right if isinstance(right,(list,tuple)) else [right]); obj=set(obj)
ids={n:sim.model.geom_name2id(n) for n in left|right|obj}
writer=imageio.get_writer(str(out/'video.mp4'),fps=max(1,int(round(control/5)))) if a.video else None
rows=[]
controller=base.robots[0].controller
original_set_goal=controller.set_goal
hold_active=False; acquired_step=None; target_pos=None; target_ori=None; held_steps=0

def held_set_goal(action):
    original_set_goal(action,set_pos=target_pos,set_ori=target_ori)

def measure(step,reward,action,done):
    contacts=[]; lhit=rhit=False
    for i in range(sim.data.ncon):
        con=sim.data.contact[i]; x,y=int(con.geom1),int(con.geom2)
        n1=sim.model.geom_id2name(x); n2=sim.model.geom_id2name(y)
        if (n1 in obj and n2 in left) or (n2 in obj and n1 in left): lhit=True
        if (n1 in obj and n2 in right) or (n2 in obj and n1 in right): rhit=True
        if n1 in obj or n2 in obj: contacts.append([n1,n2])
    built=bool(base._check_grasp(gripper=base.robots[0].gripper,object_geoms=base.cube))
    row=dict(step=step,time_s=round(step/control,6),cube_pos=sim.data.body_xpos[cube_id].tolist(),cube_height_m=float(sim.data.body_xpos[cube_id][2]),table_height_m=table,left_contact=lhit,right_contact=rhit,bilateral_contact=lhit and rhit,builtin_grasp=built,contact_pairs=contacts,native_success=bool(env.is_success()['task']),reward=float(reward),done=bool(done),action=None if action is None else np.asarray(action).tolist(),gripper_qpos=np.asarray(obs['robot0_gripper_qpos']).tolist(),mode='hold' if hold_active else 'policy',target_pos=None if target_pos is None else target_pos.tolist(),eef_pos=np.asarray(obs['robot0_eef_pos']).tolist(),goal_pos=np.asarray(controller.goal_pos).tolist(),goal_ori=np.asarray(controller.goal_ori).tolist(),target_ori=None if target_ori is None else target_ori.tolist())
    rows.append(row)
measure(0,0,None,False)
try:
    for step in range(1,horizon+1):
        if hold_active and a.resume_step and step >= a.resume_step:
            hold_active=False; controller.set_goal=original_set_goal
        if hold_active:
            act=np.array([0.,0.,0.,0.,0.,0.,1.]); held_steps+=1
        else:
            act=policy(ob=obs)
        obs,reward,done,_=env.step(act); measure(step,reward,act,done)
        if acquired_step is None and rows[-1]['bilateral_contact'] and rows[-1]['cube_height_m']-rows[0]['cube_height_m']>=0.08:
            acquired_step=step; controller.update()
            target_pos=np.array(controller.ee_pos,copy=True); target_ori=np.array(controller.ee_ori_mat,copy=True)
            controller.set_goal=held_set_goal; hold_active=True
        if writer and (step-1)%5==0:
            frame=env.render(mode='rgb_array',height=256,width=256,camera_name='agentview'); writer.append_data(frame)
        if done: break
finally:
    if writer: writer.close()
    controller.set_goal=original_set_goal
    base.close()
with (out/'telemetry.jsonl').open('w') as f:
    for row in rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
# Contact metric validation is prerequisite to outcome scoring.
valid=all(r['bilateral_contact']==r['builtin_grasp'] for r in rows)
base_h=rows[0]['cube_height_m']; elevated=[r for r in rows if r['cube_height_m']-base_h>=0.08]
acq=[r for r in rows if r['bilateral_contact'] and r['cube_height_m']-base_h>=0.08]
first_acq=acq[0]['step'] if acq else None
post_drop=bool(first_acq is not None and any(r['cube_height_m']-base_h<0.08 for r in rows[first_acq:]))
summary=dict(seed=a.seed,steps=len(rows)-1,horizon=horizon,control_hz=control,physics_dt_s=dt,physics_hz=1/dt,table_height_m=table,initial_cube_pos=rows[0]['cube_pos'],terminal_cube_pos=rows[-1]['cube_pos'],max_rise_m=max(r['cube_height_m']-base_h for r in rows),native_success_any=any(r['native_success'] for r in rows),native_success_terminal=rows[-1]['native_success'],bilateral_contact_steps=sum(r['bilateral_contact'] for r in rows),contact_proxy_validated=valid,first_acquisition_step=first_acq,post_acquisition_drop=post_drop,terminal_bilateral_contact=rows[-1]['bilateral_contact'],terminal_retained=bool(first_acq is not None and rows[-1]['cube_height_m']-base_h>=0.08 and rows[-1]['bilateral_contact'] and not post_drop),video=a.video,intervention='fixed_acquisition_eef_pose_plus_close_gripper',resume_step=a.resume_step,held_steps=held_steps,acquired_step=acquired_step,target_pos=None if target_pos is None else target_pos.tolist(),max_hold_goal_pos_error_m=max((float(np.linalg.norm(np.array(r['goal_pos'])-target_pos)) for r in rows if r['mode']=='hold'),default=None) if target_pos is not None else None,max_hold_goal_ori_error=max((float(np.max(np.abs(np.array(r['goal_ori'])-target_ori))) for r in rows if r['mode']=='hold'),default=None) if target_ori is not None else None,checkpoint_sha256=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),left_geoms=sorted(left),right_geoms=sorted(right),object_geoms=sorted(obj))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
